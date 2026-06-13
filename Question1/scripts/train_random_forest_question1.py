"""Train a random forest model for Question 1 NTU prediction."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


QUESTION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = QUESTION_ROOT / "train_for_question1.csv"
DEFAULT_VALIDATION_FILE = QUESTION_ROOT / "2026年1-3月_拼接排序.xlsx"
DEFAULT_OUTPUT_DIR = QUESTION_ROOT / "outputs" / "random_forest_sorted_validation"
TARGET_COLUMN = "NTU"
DATETIME_COLUMN = "DATETIME"
LAG_SOURCE_COLUMNS = ["R/W NTU", "FILT. NTU", "ALUM"]
LAG_HOURS = [2, 4, 6]
BASE_FEATURE_COLUMNS = [
    "RIVER LEVEL",
    "R/W FLOW",
    "R/W NTU",
    "R/W CLR",
    "R/W PH",
    "FILT. NTU",
    "C/W WELL LEVEL",
    "ALUM",
]
KEEP_COLUMNS = ["DATE", "DATE_INT", "TIME", DATETIME_COLUMN, *BASE_FEATURE_COLUMNS, TARGET_COLUMN]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a random forest regressor for Question 1."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input training CSV. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for model and reports. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=DEFAULT_VALIDATION_FILE,
        help=f"Sorted 2026 validation Excel file. Default: {DEFAULT_VALIDATION_FILE}",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=500,
        help="Number of trees. Default: 500",
    )
    return parser.parse_args()


def add_lag_features(df: pd.DataFrame, group_column: str | None = None) -> pd.DataFrame:
    df = df.sort_values(DATETIME_COLUMN, kind="stable").reset_index(drop=True).copy()
    for column in LAG_SOURCE_COLUMNS:
        for hours in LAG_HOURS:
            lag_column = f"{column}_LAG_{hours}H"
            if group_column:
                df[lag_column] = df.groupby(group_column, sort=False)[column].shift(hours // 2)
            else:
                df[lag_column] = df[column].shift(hours // 2)
    return df


def validate_time_step(df: pd.DataFrame) -> None:
    deltas = df[DATETIME_COLUMN].diff().dropna()
    bad_deltas = deltas[deltas != pd.Timedelta(hours=2)]
    if not bad_deltas.empty:
        examples = df.loc[bad_deltas.index[:5], [DATETIME_COLUMN]].copy()
        examples["delta"] = bad_deltas.head(5).to_numpy()
        raise ValueError(
            "Training data is not a continuous 2-hour sequence. Examples:\n"
            f"{examples.to_string(index=False)}"
        )


def clean_model_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).strip().upper() for column in df.columns]
    df = df.replace({"-": pd.NA, "": pd.NA, " ": pd.NA})
    for column in df.columns:
        if column in {"DATE", "DATE_INT", "TIME", DATETIME_COLUMN, "REMARKS", "SOURCE_FILE"}:
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def load_training_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN])
    df = df.sort_values(DATETIME_COLUMN, kind="stable").reset_index(drop=True)
    validate_time_step(df)
    return df[KEEP_COLUMNS].copy()


def load_validation_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0)
    df = clean_model_values(df)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["DATE_INT"] = df["DATE"].dt.strftime("%Y%m%d").astype("Int64")
    df["TIME"] = pd.to_numeric(df["TIME"], errors="coerce").astype("Int64")
    time_text = df["TIME"].astype("string").str.zfill(4)
    hour = pd.to_numeric(time_text.str[:2], errors="coerce")
    minute = pd.to_numeric(time_text.str[2:], errors="coerce")
    df[DATETIME_COLUMN] = df["DATE"] + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m")
    df = df.drop_duplicates(subset=[DATETIME_COLUMN], keep="first")
    df = df.sort_values(DATETIME_COLUMN, kind="stable").reset_index(drop=True)
    validate_time_step(df)
    df["VALIDATION_MONTH"] = df[DATETIME_COLUMN].dt.strftime("%Y-%m")
    return df[[*KEEP_COLUMNS, "VALIDATION_MONTH"]]


def build_feature_importance(
    model: RandomForestRegressor, feature_columns: list[str]
) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    train_df = load_training_data(args.input)
    valid_df = load_validation_data(args.validation_file)

    train_df = add_lag_features(train_df)
    valid_df = add_lag_features(valid_df)
    lag_feature_columns = [
        f"{column}_LAG_{hours}H"
        for column in LAG_SOURCE_COLUMNS
        for hours in LAG_HOURS
    ]
    feature_columns = [*BASE_FEATURE_COLUMNS, *lag_feature_columns]
    train_model_df = train_df.dropna(subset=[*feature_columns, TARGET_COLUMN]).copy()
    valid_model_df = valid_df.dropna(subset=[*feature_columns, TARGET_COLUMN]).copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_model_df[
        ["DATE", "DATE_INT", "TIME", DATETIME_COLUMN, *feature_columns, TARGET_COLUMN]
    ].to_csv(
        args.output_dir / "training_data_with_lags.csv",
        index=False,
        encoding="utf-8-sig",
    )
    valid_model_df[
        [
            "VALIDATION_MONTH",
            "DATE",
            "DATE_INT",
            "TIME",
            DATETIME_COLUMN,
            *feature_columns,
            TARGET_COLUMN,
        ]
    ].to_csv(
        args.output_dir / "validation_data_with_lags.csv",
        index=False,
        encoding="utf-8-sig",
    )

    x_train = train_model_df[feature_columns]
    y_train = train_model_df[TARGET_COLUMN]
    x_valid = valid_model_df[feature_columns]
    y_valid = valid_model_df[TARGET_COLUMN]

    model = RandomForestRegressor(
        n_estimators=800,
        random_state=42,
        max_depth=6,
        n_jobs=-1,
        min_samples_leaf=4,
        max_features="sqrt"
    )
    model.fit(x_train, y_train)
    train_predictions = model.predict(x_train)
    predictions = model.predict(x_valid)
    predictions_df = pd.DataFrame(
        {
            "VALIDATION_MONTH": valid_model_df["VALIDATION_MONTH"].to_numpy(),
            "DATETIME": valid_model_df[DATETIME_COLUMN].to_numpy(),
            "actual_NTU": y_valid.to_numpy(),
            "predicted_NTU": predictions,
        }
    )
    metrics_by_month = {}
    for month_name, month_predictions in predictions_df.groupby("VALIDATION_MONTH", sort=True):
        actual = month_predictions["actual_NTU"]
        predicted = month_predictions["predicted_NTU"]
        metrics_by_month[month_name] = {
            "rows": int(len(month_predictions)),
            "mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
            "r2": float(r2_score(actual, predicted)),
        }

    metrics = {
        "train_period_start": str(train_model_df[DATETIME_COLUMN].min()),
        "train_period_end": str(train_model_df[DATETIME_COLUMN].max()),
        "valid_period_start": str(valid_model_df[DATETIME_COLUMN].min()),
        "valid_period_end": str(valid_model_df[DATETIME_COLUMN].max()),
        "train_rows_after_lag": int(len(train_model_df)),
        "valid_rows_after_lag": int(len(valid_model_df)),
        "train_rows": int(len(x_train)),
        "valid_rows": int(len(x_valid)),
        "train_mae": float(mean_absolute_error(y_train, train_predictions)),
        "train_rmse": float(math.sqrt(mean_squared_error(y_train, train_predictions))),
        "train_r2": float(r2_score(y_train, train_predictions)),
        "mae": float(mean_absolute_error(y_valid, predictions)),
        "rmse": float(math.sqrt(mean_squared_error(y_valid, predictions))),
        "r2": float(r2_score(y_valid, predictions)),
        "by_validation_month": metrics_by_month,
    }

    impurity_importance = build_feature_importance(model, feature_columns)
    permutation = permutation_importance(
        model,
        x_valid,
        y_valid,
        n_repeats=20,
        random_state=42,
        n_jobs=-1,
        scoring="neg_root_mean_squared_error",
    )
    permutation_importance_df = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance_mean": permutation.importances_mean,
                "importance_std": permutation.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    joblib.dump(model, args.output_dir / "random_forest_question1.joblib")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    impurity_importance.to_csv(
        args.output_dir / "feature_importance_impurity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    permutation_importance_df.to_csv(
        args.output_dir / "feature_importance_permutation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions_df.to_csv(args.output_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("\nTop impurity importance:")
    print(impurity_importance.head(10).to_string(index=False))
    print("\nTop permutation importance:")
    print(permutation_importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
