"""Run the IDSA workbench plan on the frozen historical folds."""
import argparse
from gzip import compress
import json
from pathlib import Path
from time import perf_counter

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import OrdinalEncoder
from threadpoolctl import threadpool_limits

from temporal_modeling import (
    CONTEXT, TIMING, add_history, best_threshold, make_model, metrics,
    read_snapshot,
)

RAW = ["visits", "fails", "judged", "has_history", "raw_rate"]
PROCESS = ["previous_result", "process_x_previous"]
BASE_NUMERIC = RAW + ["smoothed_rate"] + TIMING
BASE_CATEGORICAL = CONTEXT + PROCESS
SPLIT_NUMERIC = [f"{kind}_{field}" for kind in ("routine", "recheck")
                 for field in ("visits", "judged", "fails", "fail_rate")]
PLAN = {
    "D0": ("Dummy", "训练期多数类", []),
    "M0": ("Logistic Regression", "检查背景", ["context"]),
    "M1": ("LightGBM", "检查背景", ["context"]),
    "M2": ("LightGBM", "加入原始门店历史", ["context", "history"]),
    "M3": ("LightGBM", "加入平滑历史", ["context", "history", "smoothing"]),
    "M4": ("LightGBM", "流程与检查间隔", ["context", "history", "smoothing", "process", "timing"]),
    "M5": ("LightGBM", "时间衰减 τ=365 天", ["context", "history", "smoothing", "process", "timing"]),
    "M6": ("LightGBM", "最近 3 年拟合", ["context", "history", "smoothing", "process", "timing"]),
    "M7": ("LightGBM", "最近 5 年拟合", ["context", "history", "smoothing", "process", "timing"]),
    "M8": ("CatBoost", "固定 M4 特征对照", ["context", "history", "smoothing", "process", "timing"]),
    "M9": ("LightGBM", "许可证＋地址的历史", ["context", "history", "smoothing", "process", "timing", "identity"]),
    "M10": ("LightGBM", "时间目标编码", ["context", "history", "smoothing", "process", "timing", "target_encoding"]),
    "M11": ("LightGBM × 2", "复查与非复查独立拟合", ["context", "history", "smoothing", "process", "timing"]),
    "M12": ("LightGBM + CatBoost", "验证折学习融合权重", ["context", "history", "smoothing", "process", "timing"]),
    "A0": ("HistGradientBoosting", "第一轮 Boosting M1 对照", ["context", "history", "smoothing", "process"]),
    "A1": ("HistGradientBoosting", "门店历史按复查拆分", ["context", "history", "smoothing", "process"]),
    "A2": ("HistGradientBoosting", "按当前检查类型平滑", ["context", "history", "smoothing", "process"]),
    "A3": ("HistGradientBoosting", "无匹配历史单独阈值", ["context", "history", "smoothing", "process"]),
    "A4": ("HistGradientBoosting + Logistic", "无匹配历史回退 M0", ["context", "history", "smoothing", "process"]),
    "A5": ("HistGradientBoosting", "上次失败且本次复查的间隔", ["context", "history", "smoothing", "process", "timing"]),
}
WINDOWS = {"M6": 3, "M7": 5}
LIGHTGBM = dict(n_estimators=220, learning_rate=0.05, num_leaves=15,
                min_child_samples=40, reg_lambda=1, random_state=42, n_jobs=2,
                verbosity=-1, deterministic=True, force_col_wise=True)
CATBOOST = dict(iterations=300, learning_rate=0.06, depth=5, l2_leaf_reg=5,
                random_seed=42, thread_count=2, verbose=False,
                allow_writing_files=False, one_hot_max_size=64)


def dated_counts(source, visits, keys, *, visit_counts=False):
    """Use only source days earlier than each target day for grouped statistics."""
    observed = source.dropna(subset=keys).assign(
        count=1, judged=source["y"].notna().astype(int), fails=source["y"].eq(1).astype(int),
    )
    fields = ["count", "judged", "fails"] if visit_counts else ["judged", "fails"]
    daily = observed.groupby(keys + ["date"], as_index=False)[fields].sum().sort_values("date")
    daily[fields] = daily.groupby(keys, sort=False)[fields].cumsum()
    query = visits.dropna(subset=keys)[keys + ["date"]].sort_values("date")
    return (pd.merge_asof(query.reset_index(names="row"), daily, on="date", by=keys,
                          allow_exact_matches=False).set_index("row")[fields]
            .reindex(visits.index).fillna(0))


def enrich(source, targets):
    result = add_history(source, targets)
    result["raw_rate"] = result["fails"].div(result["judged"].replace(0, np.nan)).astype(float)
    type_counts = dated_counts(source, result, ["Inspection Type"])
    global_counts = dated_counts(source, result, ["global_group"])
    global_rate = global_counts["fails"].div(global_counts["judged"].replace(0, np.nan)).fillna(0.5)
    type_rate = (type_counts["fails"] + 20 * global_rate) / (type_counts["judged"] + 20)
    result["type_smoothed_rate"] = (result["fails"] + 5 * type_rate) / (result["judged"] + 5)
    encoded = dated_counts(source, result, ["Facility Type", "Inspection Type"])
    result["facility_type_rate"] = (encoded["fails"] + 20 * type_rate) / (encoded["judged"] + 20)
    result["facility_type_support"] = encoded["judged"]
    for label, is_recheck in [("routine", False), ("recheck", True)]:
        subset = source.loc[source["reinspection"].eq(is_recheck)]
        counts = dated_counts(subset, result, ["license"], visit_counts=True)
        result[[f"{label}_visits", f"{label}_judged", f"{label}_fails"]] = counts
        result[f"{label}_fail_rate"] = (counts["fails"] + 5 * global_rate) / (counts["judged"] + 5)
    result["recheck_after_fail_days"] = result["days_since_visit"].where(
        result["reinspection"] & result["previous_result"].eq("fail"), -1,
    )
    return result


def snapshot():
    frame = read_snapshot(extra_columns=["Address"], end="2026-01-01")
    frame["reinspection"] = frame["Inspection Type"].str.contains(r"\bre[\s-]*inspection\b")
    frame["global_group"] = "all"
    address = (frame["Address"].str.lower().str.replace(r"[^a-z0-9 ]", " ", regex=True)
               .str.replace(r"\s+", " ", regex=True).str.strip().replace("", pd.NA))
    frame["address_license"] = frame["license"].astype("string") + "|" + address
    return frame


def fold_features(frame, start, end):
    source = frame.loc[frame["date"].lt(start)]
    labeled = frame.loc[frame["date"].lt(end)].dropna(subset=["y"])
    # A single join uses the same frozen history source for both partitions.
    result = enrich(source, labeled)
    composite = add_history(
        source.assign(license=source["address_license"]),
        labeled.assign(license=labeled["address_license"]),
    )
    composite["raw_rate"] = composite["fails"].div(composite["judged"].replace(0, np.nan))
    composite_columns = BASE_CATEGORICAL + BASE_NUMERIC
    composite = result.assign(**{key: composite[key] for key in composite_columns})
    training = result["date"].lt(start)
    return (result.loc[training], result.loc[~training],
            composite.loc[training], composite.loc[~training])


def features(code):
    if code == "M1":
        return CONTEXT, []
    if code in ("M2", "M3"):
        return CONTEXT, RAW + (["smoothed_rate"] if code == "M3" else [])
    if code == "M10":
        return BASE_CATEGORICAL, BASE_NUMERIC + ["facility_type_rate", "facility_type_support"]
    return BASE_CATEGORICAL, BASE_NUMERIC


def model_probabilities(code, train, val, cutoff):
    if code == "D0":
        model = DummyClassifier(strategy="most_frequent").fit(train[CONTEXT], train["y"].astype(int))
        return model.predict_proba(val[CONTEXT])[:, 1]
    if code == "M0" or code.startswith("A"):
        if code == "M0":
            model = make_model("logistic", "M0")
        else:
            model = make_model("boosting", "M1")
            transformer = model.steps[0][1]
            columns = transformer.transformers[1][2]
            if code == "A1":
                columns.extend(SPLIT_NUMERIC)
            if code == "A2":
                columns[columns.index("smoothed_rate")] = "type_smoothed_rate"
            if code == "A5":
                columns.append("recheck_after_fail_days")
        model.fit(train, train["y"].astype(int))
        return model.predict_proba(val)[:, 1]
    cats, nums = features(code)
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                             min_frequency=30, max_categories=64)
    def encoded(frame, categories):
        numeric = frame[nums].to_numpy(dtype=float, na_value=np.nan)
        return pd.DataFrame(np.column_stack([categories, numeric]), columns=cats + nums)
    fit_data = encoded(train, encoder.fit_transform(train[cats]))
    val_data = encoded(val, encoder.transform(val[cats]))
    if code == "M8":
        # Force one-hot categories so internal target statistics cannot use same-day labels.
        category_types = dict.fromkeys(cats, str)
        model = CatBoostClassifier(**CATBOOST, cat_features=cats)
        model.fit(fit_data.astype(category_types), train["y"].astype(int))
        return model.predict_proba(val_data.astype(category_types))[:, 1]
    weights = None
    if code == "M5":
        weights = np.exp(-(pd.Timestamp(cutoff) - train["date"]).dt.days.to_numpy() / 365)
        weights /= weights.mean()
    model = LGBMClassifier(**LIGHTGBM)
    model.fit(fit_data, train["y"].astype(int), sample_weight=weights, categorical_feature=cats)
    return model.predict_proba(val_data)[:, 1]


def choose_blend(left, right, select_threshold=best_threshold):
    """Require unique correct classifications from both parents on earlier folds."""
    if left.empty:
        return 1.0, 0.5, None
    a = left["probability"].ge(select_threshold(left))
    b = right["probability"].ge(select_threshold(right))
    y = left["y"].astype(bool)
    unique = (int((a.eq(y) & b.ne(y)).sum()), int((b.eq(y) & a.ne(y)).sum()))
    options = (0, .25, .5, .75, 1) if all(unique) else (0, 1)
    scored = []
    for weight in options:
        blend = left.assign(probability=weight * left["probability"] + (1 - weight) * right["probability"])
        threshold = select_threshold(blend)
        scored.append((metrics(blend, threshold)["macro_f1"], weight in (0, 1), weight, threshold))
    _, _, weight, threshold = max(scored)
    return weight, threshold, unique


def subset_metrics(data, thresholds):
    scored = data.assign(predicted=data["probability"].ge(thresholds).astype(int))
    groups = {
        "all": pd.Series(True, index=data.index),
        "with_history": data["has_history"].eq(1),
        "cold_start": data["has_history"].eq(0),
        "invalid_license": data["invalid_license"],
        "valid_no_history": data["has_history"].eq(0) & ~data["invalid_license"],
        "history_1_2": data["visits"].between(1, 2),
        "history_3_plus": data["visits"].ge(3),
        "reinspection": data["reinspection"],
        "non_reinspection": ~data["reinspection"],
    }
    return {key: metrics(scored.loc[mask].assign(probability=scored.loc[mask, "predicted"]), .5)
            for key, mask in groups.items() if mask.any()}


def as_validation(val, probabilities):
    return val[["y", "reinspection"]].assign(
        has_history=val["has_history"], visits=val["visits"],
        invalid_license=val["license"].isna(), probability=probabilities,
    )


def experiment_folds(frame, periods, *, plan=PLAN, predictive=model_probabilities,
                     select_threshold=best_threshold, record_default=False):
    earlier = {key: [] for key in plan}
    results = []
    for fold, cutoff, end in periods:
        train, val, composite_train, composite_val = fold_features(frame, cutoff, end)
        current = {}
        for code in plan:
            duration = perf_counter()
            start = str(frame["date"].min().date())
            selected_train = train
            if code in WINDOWS:
                start = str((pd.Timestamp(cutoff) - pd.DateOffset(years=WINDOWS[code])).date())
                selected_train = train.loc[train["date"].ge(start)]
            prior = pd.concat(earlier[code], ignore_index=True) if earlier[code] else pd.DataFrame()
            threshold = select_threshold(prior) if len(prior) and code != "D0" else .5
            extras = {}
            if code == "M9":
                data = as_validation(val, predictive(code, composite_train, composite_val, cutoff))
                extras["composite_history_count"] = int(composite_val["has_history"].sum())
            elif code == "M11":
                data = as_validation(val, np.nan)
                for group in (False, True):
                    fit = train.loc[train["reinspection"].eq(group)]
                    subset = val.loc[val["reinspection"].eq(group)]
                    if len(subset):
                        data.loc[subset.index, "probability"] = model_probabilities("M4", fit, subset, cutoff)
            elif code == "M12":
                left = pd.concat(earlier["M4"][:-1], ignore_index=True) if len(earlier["M4"]) > 1 else pd.DataFrame()
                right = pd.concat(earlier["M8"][:-1], ignore_index=True) if len(earlier["M8"]) > 1 else pd.DataFrame()
                weight, threshold, unique = choose_blend(left, right, select_threshold)
                extras = dict(lightgbm_weight=weight, earlier_unique_correct=unique)
                data = current["M4"].assign(
                    probability=weight * current["M4"]["probability"] + (1 - weight) * current["M8"]["probability"],
                )
            elif code == "A3":
                data = current["A0"]
                cold = prior.loc[prior["has_history"].eq(0)] if len(prior) else prior
                extras["cold_threshold"] = select_threshold(cold) if len(cold) else threshold
            elif code == "A4":
                data = current["A0"].assign(
                    probability=current["A0"]["probability"].where(
                        current["A0"]["has_history"].eq(1), current["M0"]["probability"],
                    ),
                )
            else:
                data = as_validation(val, predictive(code, selected_train, val, cutoff))
            if data["probability"].isna().any():
                raise ValueError(f"Missing predictions for {code}, {fold}")
            current[code] = data
            earlier[code].append(data)
            if code == "A3":
                threshold_by_row = pd.Series(threshold, index=data.index).where(
                    data["has_history"].eq(1), extras["cold_threshold"],
                )
            elif code == "A4" and len(prior):
                # Use the parent models' own earlier validation thresholds unchanged.
                m1 = pd.concat(earlier["A0"][:-1], ignore_index=True)
                m0 = pd.concat(earlier["M0"][:-1], ignore_index=True)
                threshold = select_threshold(m1)
                extras["cold_threshold"] = select_threshold(m0)
                threshold_by_row = pd.Series(threshold, index=data.index).where(
                    data["has_history"].eq(1), extras["cold_threshold"],
                )
            else:
                threshold_by_row = threshold
            if code == "A4" and not len(prior):
                extras["cold_threshold"] = .5
            if record_default and len(prior):
                extras["default_metrics"] = metrics(data, .5)
            results.append(dict(code=code, fold=fold, train_start=start,
                                train_end=str((pd.Timestamp(cutoff) - pd.Timedelta(days=1)).date()),
                                val_start=cutoff, val_end=str((pd.Timestamp(end) - pd.Timedelta(days=1)).date()),
                                n_train=len(selected_train), threshold=threshold,
                                fit_and_score_seconds=round(perf_counter() - duration, 3),
                                subgroups=subset_metrics(data, threshold_by_row), **extras))
        print(f"Completed {fold}: {len(train):,} training and {len(val):,} validation inspections", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiment_plan_results.json.gz"))
    args = parser.parse_args()
    frame = snapshot()
    with threadpool_limits(limits=2):
        results = experiment_folds(frame, [(str(year), f"{year}-01-01", f"{year + 1}-01-01")
                                           for year in (2023, 2024, 2025)])
    payload = json.dumps({"protocol": "historical-frozen-v2", "folds": results},
                         ensure_ascii=False, allow_nan=False, separators=(",", ":"),
                         default=np.generic.item) + "\n"
    args.output.write_bytes(compress(payload.encode("utf-8"), mtime=0))


if __name__ == "__main__":
    main()
