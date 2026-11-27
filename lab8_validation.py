"""Evaluate out-of-time threshold selection and class imbalance treatments."""
import argparse
import gzip
import json
from pathlib import Path

from imblearn.over_sampling import SMOTENC
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from experiment_plan import (
    BASE_CATEGORICAL, BASE_NUMERIC, LIGHTGBM, PLAN, experiment_folds, model_probabilities, snapshot,
)
from temporal_modeling import best_threshold

GRID = np.arange(2, 99) / 100
GROUPS = PLAN["M4"][2]
ADDITIONS = {
    "B1": ("LightGBM", "M4 + balanced class weights", GROUPS),
    "B2": ("LightGBM", "M4 + random undersampling", GROUPS),
    "B3": ("LightGBM", "M4 + SMOTENC", GROUPS),
    "R0": ("Random Forest", "M4 features", GROUPS),
    "R1": ("Random Forest", "M4 features + balanced bootstrap weights", GROUPS),
}
CONFIGS = {**PLAN, **ADDITIONS}
CURVE_RUNS = ["M4", "B1", "B2", "B3", "R0", "R1"]


def select_threshold(data):
    return best_threshold(data, GRID)


def matrices(train, val):
    cats, nums = BASE_CATEGORICAL, BASE_NUMERIC
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                             min_frequency=30, max_categories=64)
    encoded_train = encoder.fit_transform(train[cats])
    encoded_val = encoder.transform(val[cats])

    def joined(frame, encoded):
        numeric = (
            frame[nums].astype(float).fillna(-1).to_numpy()
            if frame is not None else np.empty((0, len(nums)))
        )
        return np.column_stack([encoded, numeric])

    return joined(train, encoded_train), joined(val, encoded_val)


def fit_variant(code, train, val):
    cats = BASE_CATEGORICAL
    y = train["y"].to_numpy(dtype=int)
    x_train, x_val = matrices(train, val)
    # Retain raw-rate missingness for the standard LightGBM representation.
    if code.startswith("B"):
        raw_rate = len(cats) + BASE_NUMERIC.index("raw_rate")
        x_train[:, raw_rate] = train["raw_rate"].to_numpy(dtype=float, na_value=np.nan)
        x_val[:, raw_rate] = val["raw_rate"].to_numpy(dtype=float, na_value=np.nan)
    if code == "B2":
        rng = np.random.default_rng(42)
        fail_rows = np.flatnonzero(y == 1)
        nonfail_rows = rng.choice(np.flatnonzero(y == 0), size=len(fail_rows), replace=False)
        rows = np.sort(np.concatenate([fail_rows, nonfail_rows]))
        x_train, y = x_train[rows], y[rows]
    if code == "B3":
        # Scale distances for count/timing columns; preserve category IDs for SMOTENC.
        numeric = x_train[:, len(cats):]
        raw_missing = np.isnan(numeric)
        numeric[raw_missing] = -1
        scale = numeric.std(axis=0)
        scale[scale == 0] = 1
        numeric /= scale
        x_train, y = SMOTENC(
            categorical_features=list(range(len(cats))), random_state=42,
        ).fit_resample(x_train, y)
        x_train[:, len(cats):] *= scale
        x_train[x_train[:, raw_rate] < 0, raw_rate] = np.nan

    columns = BASE_CATEGORICAL + BASE_NUMERIC
    x_train = pd.DataFrame(x_train, columns=columns)
    x_val = pd.DataFrame(x_val, columns=columns)
    if code.startswith("B"):
        model = LGBMClassifier(**LIGHTGBM)
        weights = compute_sample_weight("balanced", y) if code == "B1" else None
        model.fit(x_train, y, sample_weight=weights, categorical_feature=cats)
    else:
        # Nominal categories are one-hot; ordinal codes do not become ordered splits.
        cat_train = pd.get_dummies(x_train[cats].astype(str), prefix=cats)
        cat_val = pd.get_dummies(x_val[cats].astype(str), prefix=cats)
        x_train = pd.concat([cat_train, x_train[BASE_NUMERIC]], axis=1)
        x_val = pd.concat([cat_val.reindex(columns=cat_train.columns, fill_value=False),
                           x_val[BASE_NUMERIC]], axis=1)
        model = RandomForestClassifier(n_estimators=160, min_samples_leaf=15, max_features="sqrt",
                                       random_state=42, n_jobs=2,
                                       class_weight="balanced_subsample" if code == "R1" else None)
        model.fit(x_train, y)
    return model.predict_proba(x_val)[:, 1]


def calibration_curve(data):
    threshold_grid = np.unique(np.r_[GRID[::2], select_threshold(data)])
    scores = data["probability"].to_numpy()
    labels = data["y"].to_numpy(dtype=int)
    order = scores.argsort()
    excluded = np.searchsorted(scores[order], threshold_grid)
    fn = np.r_[0, labels[order].cumsum()][excluded]
    tn = excluded - fn
    tp = labels.sum() - fn
    fp = len(labels) - excluded - tp
    errors = fp + fn
    def divide(top, bottom):
        return np.divide(top, bottom, out=np.zeros_like(threshold_grid), where=bottom > 0)
    macro = (divide(2 * tp, 2 * tp + errors) + divide(2 * tn, 2 * tn + errors)) / 2
    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    return {float(t): dict(macro_f1=m, fail_precision=p, fail_recall=r)
            for t, m, p, r in zip(threshold_grid, macro, precision, recall)}


def run():
    periods = [(str(year), f"{year}-01-01", f"{year+1}-01-01") for year in range(2022, 2026)]
    curves = {}
    prior = {code: [] for code in CURVE_RUNS}

    def predict(code, train, val, cutoff):
        p = (fit_variant(code, train, val) if code in ADDITIONS
             else model_probabilities(code, train, val, cutoff))
        if code in prior:
            if prior[code]:
                calibration = pd.concat(prior[code], ignore_index=True)
                curves.setdefault(code, {})[cutoff[:4]] = calibration_curve(calibration)
            prior[code].append(val[["y"]].assign(probability=p))
        return p

    frame = snapshot()
    with threadpool_limits(limits=2):
        folds = experiment_folds(frame, periods, plan=CONFIGS, predictive=predict,
                                 select_threshold=select_threshold, record_default=True)
    return {"protocol": "annual-past-only-threshold-v3",
            "folds": [fold for fold in folds if fold["fold"] != "2022"], "curves": curves}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("lab8_results.json.gz"))
    args = parser.parse_args()
    payload = json.dumps(run(), default=np.generic.item, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    args.output.write_bytes(gzip.compress(payload, mtime=0))


if __name__ == "__main__":
    main()
