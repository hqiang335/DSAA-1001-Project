"""Offline Phase 2 experiments for the frozen Chicago inspection snapshot."""
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from threadpoolctl import threadpool_limits

CONTEXT = ["Inspection Type", "Facility Type", "Risk"]
HISTORY = ["visits", "fails", "judged", "smoothed_rate", "has_history"]
TIMING = ["days_since_visit", "days_since_fail"]
LABELS = {"fail": 1, "pass": 0, "pass w/ conditions": 0}
MISSING = "__missing__"
THRESHOLDS = np.arange(10, 61, 2) / 100


def read_snapshot(*, extra_columns=(), end="2026-07-01"):
    metadata = json.loads(Path("data/snapshot_metadata.json").read_text())
    columns = CONTEXT + ["Inspection ID", "Inspection Date", "License #", "Results"] + list(extra_columns)
    with ZipFile("data/food_inspections_snapshot.zip") as archive:
        name = metadata["csv_file"]
        with archive.open(name) as raw:
            digest = sha256()
            for block in iter(lambda: raw.read(2**20), b""):
                digest.update(block)
        if digest.hexdigest() != metadata["csv_sha256"]:
            raise ValueError("Snapshot checksum differs from the project metadata")
        with archive.open(name) as raw:
            frame = pd.read_csv(raw, usecols=columns, dtype="string")
    if len(frame) != metadata["row_count"]:
        raise ValueError("Snapshot row count differs from the project metadata")
    frame["date"] = pd.to_datetime(frame["Inspection Date"], format="%m/%d/%Y")
    frame = frame.loc[frame["date"].lt(end)].drop_duplicates()
    if frame["Inspection ID"].isna().any() or frame["Inspection ID"].duplicated().any():
        raise ValueError("Missing or conflicting inspection IDs")
    for column in CONTEXT:
        frame[column] = (frame[column].str.strip().str.lower()
                         .str.replace(r"\s+", " ", regex=True)
                         .replace("", MISSING).fillna(MISSING))
    license_id = pd.to_numeric(frame["License #"], errors="coerce")
    frame["license"] = license_id.where((license_id > 0) & (license_id % 1 == 0))
    frame["y"] = frame["Results"].str.strip().str.lower().map(LABELS)
    return frame


def add_history(source, visits):
    """Join to the latest strictly earlier license-day in the allowed source."""
    events = source.dropna(subset=["license"]).assign(
        fails=lambda d: d["y"].eq(1).astype(int),
        judged=lambda d: d["y"].notna().astype(int),
    )
    daily = (events.groupby(["license", "date"], as_index=False)
             .agg(visits=("date", "size"), fails=("fails", "sum"),
                  judged=("judged", "sum"),
                  previous_result=("y", lambda y: (
                      "mixed" if y.nunique(dropna=False) > 1 else
                      "unjudged" if y.isna().all() else
                      "fail" if y.iloc[0] == 1 else "non_fail"))))
    daily = daily.sort_values("date")
    daily["fail_date"] = daily["date"].where(daily["fails"].gt(0))
    daily["fail_date"] = daily.groupby("license")["fail_date"].ffill()
    counts = ["visits", "fails", "judged"]
    daily[counts] = daily.groupby("license")[counts].cumsum()
    daily = daily.rename(columns={"date": "history_date"})
    known = visits.loc[visits["license"].notna(), ["license", "date"]].sort_values("date")
    joined = pd.merge_asof(
        known.reset_index(names="row"), daily, by="license",
        left_on="date", right_on="history_date", allow_exact_matches=False,
    ).set_index("row").reindex(visits.index)
    out = visits.copy()
    out[counts] = joined[counts].fillna(0)
    out["has_history"] = out["visits"].gt(0).astype(int)
    daily_pool = (source.groupby("date")["y"].agg(["sum", "count"])
                  .sort_index().cumsum())
    daily_pool["prior"] = daily_pool["sum"].div(daily_pool["count"].replace(0, np.nan))
    prior = daily_pool["prior"].reindex(
        pd.DatetimeIndex(out["date"]) - pd.Timedelta(days=1), method="ffill",
    ).fillna(0.5).to_numpy()
    out["smoothed_rate"] = (out["fails"] + 5 * prior) / (out["judged"] + 5)
    out["previous_result"] = joined["previous_result"].fillna("no_history")
    out["process_x_previous"] = out["Inspection Type"] + " | " + out["previous_result"]
    out["days_since_visit"] = (out["date"] - joined["history_date"]).dt.days.fillna(-1)
    out["days_since_fail"] = (out["date"] - joined["fail_date"]).dt.days.fillna(-1)
    return out


def make_model(algorithm, feature_set):
    categories = CONTEXT.copy()
    numbers = []
    if feature_set != "M0":
        categories += ["previous_result", "process_x_previous"]
        numbers += HISTORY
    if feature_set == "M2":
        numbers += TIMING
    if algorithm == "logistic":
        encoder = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30)
        transform = ColumnTransformer([
            ("context", encoder, categories), ("numeric", StandardScaler(), numbers),
        ])
        estimator = LogisticRegression(max_iter=500)
    else:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1,
            min_frequency=30, max_categories=64,
        )
        transform = ColumnTransformer([
            ("context", encoder, categories), ("numeric", "passthrough", numbers),
        ])
        estimator = HistGradientBoostingClassifier(
            categorical_features=list(range(len(categories))), max_iter=150,
            max_leaf_nodes=15, early_stopping=False, random_state=42,
        )
    return make_pipeline(transform, estimator)


def fold_data(frame, cutoff, end):
    source = frame.loc[frame["date"].lt(cutoff)]
    train = source.dropna(subset=["y"])
    val = frame.loc[frame["date"].ge(cutoff) & frame["date"].lt(end)]
    val = val.dropna(subset=["y"])
    if train.empty or val.empty:
        raise ValueError(f"Empty train or validation data at {cutoff}")
    return add_history(source, train), add_history(source, val)


def metrics(data, threshold):
    pred = data["probability"].ge(threshold).astype(int)
    y = data["y"]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, labels=[0, 1], zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(n=len(y), fail_rate=y.mean(), macro_f1=f1.mean(),
                pass_precision=precision[0], pass_recall=recall[0], pass_f1=f1[0],
                fail_precision=precision[1], fail_recall=recall[1], fail_f1=f1[1],
                tn=tn, fp=fp, fn=fn, tp=tp)


def best_threshold(data):
    y = data["y"].to_numpy(dtype=bool)
    p = data["probability"].to_numpy()
    pred = p[:, None] >= THRESHOLDS
    tp = np.count_nonzero(pred[y], axis=0)
    tn = np.count_nonzero(~pred[~y], axis=0)
    errors = len(y) - tp - tn
    pass_f1 = np.divide(2 * tn, 2 * tn + errors, out=np.zeros_like(THRESHOLDS),
                        where=(2 * tn + errors) > 0)
    fail_f1 = np.divide(2 * tp, 2 * tp + errors, out=np.zeros_like(THRESHOLDS),
                        where=(2 * tp + errors) > 0)
    order = np.lexsort((abs(THRESHOLDS - 0.5), -(pass_f1 + fail_f1)))
    return float(THRESHOLDS[order[0]])


def predict_fold(train, val, algorithm, feature_set, start_year):
    fit = train.loc[train["date"].dt.year.ge(start_year)]
    model = make_model(algorithm, feature_set)
    model.fit(fit, fit["y"].astype(int))
    predictions = val[["y", "has_history"]].copy()
    predictions["probability"] = model.predict_proba(val)[:, 1]
    return predictions


def run():
    frame = read_snapshot()
    configs = [(algorithm, features, year)
               for algorithm in ("logistic", "boosting")
               for features in ("M0", "M1", "M2") for year in (2016, 2019)]
    results = []
    historical = {config: [] for config in configs}
    with threadpool_limits(limits=2):
        for year in (2023, 2024, 2025):
            train, val = fold_data(frame, f"{year}-01-01", f"{year + 1}-01-01")
            for config in configs:
                data = predict_fold(train, val, *config)
                earlier = historical[config]
                threshold = best_threshold(pd.concat(earlier)) if earlier else 0.5
                earlier.append(data)
                results.append(dict(year=year, algorithm=config[0], features=config[1],
                                    start_year=config[2], threshold=threshold,
                                    **metrics(data, threshold)))
        history = pd.DataFrame(results)
        # Rank on 2024/25, when thresholds can be set on earlier folds.
        ranking = (history.loc[history["year"].gt(2023)]
                   .groupby(["algorithm", "features", "start_year"])["macro_f1"]
                   .agg(["mean", "min"]).sort_values(["mean", "min"], ascending=False))
        baseline = ("logistic", "M0", 2016)
        contenders = list(dict.fromkeys([baseline, *ranking.index[:2]]))
        train, val = fold_data(frame, "2026-01-01", "2026-07-01")
        validation = []
        for config in contenders:
            data = predict_fold(train, val, *config)
            threshold = best_threshold(pd.concat(historical[config]))
            for group, subset in (("all", data),
                                  ("with_history", data.loc[data["has_history"].eq(1)]),
                                  ("cold_start", data.loc[data["has_history"].eq(0)])):
                validation.append(dict(algorithm=config[0], features=config[1],
                                       start_year=config[2], threshold=threshold, group=group,
                                       **metrics(subset, threshold)))
    return history, ranking, pd.DataFrame(validation)


if __name__ == "__main__":
    historical, ranking, validation = run()
    print(historical.to_string(index=False))
    print(ranking.to_string())
    print(validation.to_string(index=False))
