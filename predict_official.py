"""Score the course-provided unlabeled CSV with a previously locked configuration."""
import argparse
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from experiment_plan import model_probabilities
from lab8_validation import ADDITIONS, fit_variant
from temporal_modeling import CONTEXT, MISSING, add_history, read_snapshot

TEST_START = "2026-10-31"
TEST_END = "2026-11-27"
FINAL_MODELS = ("M0", "M3", "M4", "A0", *ADDITIONS)


def read_test(path):
    frame = pd.read_csv(path, dtype="string")
    required = ["Inspection ID", "Inspection Date", "License #", *CONTEXT]
    if "Results" in frame or frame.empty or set(required) - set(frame):
        raise ValueError("Expected a nonempty course CSV with pre-inspection columns and no Results")
    frame = frame[required].copy()
    ids = frame["Inspection ID"].str.strip()
    if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Inspection IDs must be present and unique")
    frame["Inspection ID"] = ids
    frame["date"] = pd.to_datetime(frame["Inspection Date"], format="mixed").dt.normalize()
    if not frame["date"].between(TEST_START, TEST_END).all():
        raise ValueError(f"All test dates must be between {TEST_START} and {TEST_END}")
    for column in CONTEXT:
        frame[column] = (frame[column].str.strip().str.lower()
                         .str.replace(r"\s+", " ", regex=True)
                         .replace("", MISSING).fillna(MISSING))
    license_id = pd.to_numeric(frame["License #"], errors="coerce")
    frame["license"] = license_id.where((license_id > 0) & (license_id % 1 == 0))
    return frame


def predict(test_path, model, threshold):
    if model not in FINAL_MODELS or not 0 <= threshold <= 1:
        raise ValueError("Expected a supported locked model and a threshold in [0, 1]")
    test = read_test(test_path)
    history = read_snapshot(end=TEST_START)
    if test["Inspection ID"].isin(history["Inspection ID"]).any():
        raise ValueError("Official test IDs overlap the training snapshot")
    train = add_history(history, history.dropna(subset=["y"]))
    rows = add_history(history, test)
    for dataset in (train, rows):
        dataset["raw_rate"] = dataset["fails"].div(dataset["judged"].replace(0, float("nan")))
    with threadpool_limits(limits=2):
        p = (fit_variant(model, train, rows) if model in ADDITIONS
             else model_probabilities(model, train, rows, TEST_START))
    return pd.DataFrame({"Inspection ID": rows["Inspection ID"], "pred_fail": (p >= threshold).astype(int)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_csv", type=Path)
    parser.add_argument("--model", choices=FINAL_MODELS, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()
    if args.test_csv.resolve() == args.output.resolve():
        parser.error("Output must differ from input")
    predict(args.test_csv, args.model, args.threshold).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
