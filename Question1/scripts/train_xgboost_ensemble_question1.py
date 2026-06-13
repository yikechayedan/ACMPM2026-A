"""Train XGBoost and RF+XGBoost ensemble models for Question 1."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


QUESTION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_FILE = QUESTION_ROOT / "train_for_question1.csv"
DEFAULT_VALIDATION_FILE = QUESTION_ROOT / "2026年1-3月_拼接排序.xlsx"
DEFAULT_OUTPUT_DIR = QUESTION_ROOT / "outputs" / "xgboost_ensemble"

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
        description="Train XGBoost and RF+XGBoost ensemble models for Question 1."
    )
    parser.add_argument(
        "--training-file",
        type=Path,
        default=DEFAULT_TRAINING_FILE,
        help=f"2025 training CSV. Default: {DEFAULT_TRAINING_FILE}",
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=DEFAULT_VALIDATION_FILE,
        help=f"Sorted 2026 validation Excel file. Default: {DEFAULT_VALIDATION_FILE}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--rf-estimators",
        type=int,
        default=500,
        help="Random forest tree count. Default: 500",
    )
    parser.add_argument(
        "--xgb-estimators",
        type=int,
        default=600,
        help="XGBoost boosting rounds. Default: 600",
    )
    return parser.parse_args()


def validate_time_step(df: pd.DataFrame) -> None:
    deltas = df[DATETIME_COLUMN].diff().dropna()
    bad_deltas = deltas[deltas != pd.Timedelta(hours=2)]
    if not bad_deltas.empty:
        examples = df.loc[bad_deltas.index[:5], [DATETIME_COLUMN]].copy()
        examples["delta"] = bad_deltas.head(5).to_numpy()
        raise ValueError(
            "Data is not a continuous 2-hour sequence. Examples:\n"
            f"{examples.to_string(index=False)}"
        )


def clean_model_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).strip().upper() for column in df.columns]
    df = df.replace({"-": pd.NA, "": pd.NA, " ": pd.NA})
    for column in df.columns:
        if column in {"DATE", "DATE_INT", "TIME", DATETIME_COLUMN, "REMARKS"}:
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
    df = df[df[DATETIME_COLUMN].dt.month != 3].copy()
    return df[[*KEEP_COLUMNS, "VALIDATION_MONTH"]]


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(DATETIME_COLUMN, kind="stable").reset_index(drop=True).copy()
    for column in LAG_SOURCE_COLUMNS:
        for hours in LAG_HOURS:
            df[f"{column}_LAG_{hours}H"] = df[column].shift(hours // 2)
    return df


def build_model_frames(
    training_file: Path, validation_file: Path
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_df = add_lag_features(load_training_data(training_file))
    valid_df = add_lag_features(load_validation_data(validation_file))

    lag_feature_columns = [
        f"{column}_LAG_{hours}H"
        for column in LAG_SOURCE_COLUMNS
        for hours in LAG_HOURS
    ]
    feature_columns = [*BASE_FEATURE_COLUMNS, *lag_feature_columns]

    train_model_df = train_df.dropna(subset=[*feature_columns, TARGET_COLUMN]).copy()
    valid_model_df = valid_df.dropna(subset=[*feature_columns, TARGET_COLUMN]).copy()
    return train_model_df, valid_model_df, feature_columns


def evaluate_predictions(actual: pd.Series, predicted) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def evaluate_by_month(predictions_df: pd.DataFrame, prediction_column: str) -> dict[str, dict]:
    metrics = {}
    for month_name, month_predictions in predictions_df.groupby("VALIDATION_MONTH", sort=True):
        metrics[month_name] = {
            "rows": int(len(month_predictions)),
            **evaluate_predictions(month_predictions["actual_NTU"], month_predictions[prediction_column]),
        }
    return metrics


def build_metrics_summary(metrics: dict) -> pd.DataFrame:
    rows = []
    for model_name, model_metrics in metrics["models"].items():
        for dataset_name in ["train", "validation"]:
            dataset_metrics = model_metrics[dataset_name]
            rows.append(
                {
                    "model": model_name,
                    "dataset": dataset_name,
                    "mae": dataset_metrics["mae"],
                    "rmse": dataset_metrics["rmse"],
                    "r2": dataset_metrics["r2"],
                }
            )
    return pd.DataFrame(rows)


def build_xgboost_model(n_estimators: int):
    try:
        from xgboost import XGBRegressor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xgboost is not installed. Install it with: "
            ".\\.venv\\Scripts\\python.exe -m pip install xgboost"
        ) from exc

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=3,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )


def main() -> None:
    args = parse_args()
    train_df, valid_df, feature_columns = build_model_frames(
        args.training_file, args.validation_file
    )

    x_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]
    x_valid = valid_df[feature_columns]
    y_valid = valid_df[TARGET_COLUMN]

    rf_model = RandomForestRegressor(
        n_estimators=800,
        random_state=42,
        max_depth=6,
        n_jobs=-1,
        min_samples_leaf=4,
        max_features="sqrt"
    )
    xgb_model = build_xgboost_model(args.xgb_estimators)

    rf_model.fit(x_train, y_train)
    xgb_model.fit(x_train, y_train)

    rf_train_predictions = rf_model.predict(x_train)
    xgb_train_predictions = xgb_model.predict(x_train)
    ensemble_train_predictions = (rf_train_predictions + xgb_train_predictions) / 2

    rf_predictions = rf_model.predict(x_valid)
    xgb_predictions = xgb_model.predict(x_valid)
    ensemble_predictions = (rf_predictions + xgb_predictions) / 2

    predictions_df = pd.DataFrame(
        {
            "VALIDATION_MONTH": valid_df["VALIDATION_MONTH"].to_numpy(),
            "DATETIME": valid_df[DATETIME_COLUMN].to_numpy(),
            "actual_NTU": y_valid.to_numpy(),
            "rf_predicted_NTU": rf_predictions,
            "xgb_predicted_NTU": xgb_predictions,
            "ensemble_predicted_NTU": ensemble_predictions,
        }
    )

    metrics = {
        "train_period_start": str(train_df[DATETIME_COLUMN].min()),
        "train_period_end": str(train_df[DATETIME_COLUMN].max()),
        "valid_period_start": str(valid_df[DATETIME_COLUMN].min()),
        "valid_period_end": str(valid_df[DATETIME_COLUMN].max()),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "models": {
            "random_forest": {
                "train": evaluate_predictions(y_train, rf_train_predictions),
                "validation": evaluate_predictions(y_valid, rf_predictions),
                **evaluate_predictions(y_valid, rf_predictions),
                "by_validation_month": evaluate_by_month(predictions_df, "rf_predicted_NTU"),
            },
            "xgboost": {
                "train": evaluate_predictions(y_train, xgb_train_predictions),
                "validation": evaluate_predictions(y_valid, xgb_predictions),
                **evaluate_predictions(y_valid, xgb_predictions),
                "by_validation_month": evaluate_by_month(predictions_df, "xgb_predicted_NTU"),
            },
            "ensemble_average": {
                "train": evaluate_predictions(y_train, ensemble_train_predictions),
                "validation": evaluate_predictions(y_valid, ensemble_predictions),
                **evaluate_predictions(y_valid, ensemble_predictions),
                "by_validation_month": evaluate_by_month(
                    predictions_df, "ensemble_predicted_NTU"
                ),
            },
        },
    }

    xgb_importance = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": xgb_model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_df[["DATE", "DATE_INT", "TIME", DATETIME_COLUMN, *feature_columns, TARGET_COLUMN]].to_csv(
        args.output_dir / "training_data_with_lags.csv",
        index=False,
        encoding="utf-8-sig",
    )
    valid_df[
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
    predictions_df.to_csv(
        args.output_dir / "validation_predictions_ensemble.csv",
        index=False,
        encoding="utf-8-sig",
    )
    xgb_importance.to_csv(
        args.output_dir / "feature_importance_xgboost.csv",
        index=False,
        encoding="utf-8-sig",
    )
    joblib.dump(rf_model, args.output_dir / "random_forest_question1.joblib")
    joblib.dump(xgb_model, args.output_dir / "xgboost_question1.joblib")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("\nTrain/validation metrics summary:")
    print(build_metrics_summary(metrics).to_string(index=False))
    print("\nTop XGBoost importance:")
    print(xgb_importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
