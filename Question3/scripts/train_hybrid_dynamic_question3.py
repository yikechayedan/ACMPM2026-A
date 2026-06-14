"""Hybrid dynamic NTU forecasting model for Question 3.

The model combines a clear-well residence-time-distribution (RTD) feature layer
with horizon-specific XGBoost residual models. The source data are sampled every
2 hours, so the direct forecast horizons are 2, 4, 6, 8, 10, and 12 hours.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


QUESTION_ROOT = Path(__file__).resolve().parents[1]
TRAINING_FILE = QUESTION_ROOT / "train.csv"
VALIDATION_FILE = QUESTION_ROOT / "validate.csv"
OUTPUT_DIR = QUESTION_ROOT / "outputs" / "hybrid_dynamic_model"

DATETIME_COLUMN = "DATETIME"
TARGET_COLUMN = "NTU"
OBSERVED_TARGET_COLUMN = "NTU_OBSERVED"
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
RTD_SOURCE_COLUMNS = ["R/W NTU", "FILT. NTU", "ALUM", "NTU"]
RTD_MAX_LAG_STEPS = 12
LAG_STEPS = [1, 2, 3, 4, 5, 6, 9, 12]
HORIZON_STEPS = [1, 2, 3, 4, 5, 6]
TARGET_DATES = ["2026-02-01", "2026-02-10", "2026-02-20"]
TARGET_TIMES = [700, 900, 1100, 1300, 1500, 1700, 1900]


@dataclass(frozen=True)
class ModelBundle:
    horizon_hours: int
    model: object
    feature_columns: list[str]


def build_xgboost_model():
    try:
        from xgboost import XGBRegressor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xgboost is required. Run: .\\.venv\\Scripts\\python.exe -m pip install xgboost"
        ) from exc

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.035,
        max_depth=3,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.02,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1,
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).strip().upper() for column in df.columns]
    return df


def add_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    time_text = pd.to_numeric(df["TIME"], errors="coerce").astype("Int64").astype("string").str.zfill(4)
    hour = pd.to_numeric(time_text.str[:2], errors="coerce")
    minute = pd.to_numeric(time_text.str[2:], errors="coerce")
    df[DATETIME_COLUMN] = df["DATE"] + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m")
    df["DATE_INT"] = df["DATE"].dt.strftime("%Y%m%d").astype("Int64")
    df["TIME"] = pd.to_numeric(df["TIME"], errors="coerce").astype("Int64")
    for column in BASE_FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.sort_values(DATETIME_COLUMN, kind="stable").reset_index(drop=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = add_datetime(pd.read_csv(TRAINING_FILE))
    valid_df = add_datetime(pd.read_csv(VALIDATION_FILE))
    combined = pd.concat(
        [
            train_df.assign(DATASET="train"),
            valid_df.assign(DATASET="validation"),
        ],
        ignore_index=True,
    ).sort_values(DATETIME_COLUMN, kind="stable").reset_index(drop=True)
    combined[OBSERVED_TARGET_COLUMN] = combined[TARGET_COLUMN]
    combined["NTU_WAS_RECURSIVELY_FILLED"] = False
    return train_df, valid_df, combined


def gamma_rtd_weights(tau_steps: float, max_lag_steps: int = RTD_MAX_LAG_STEPS) -> np.ndarray:
    lag_steps = np.arange(1, max_lag_steps + 1, dtype=float)
    shape = 3.0
    scale = max(tau_steps / shape, 0.2)
    raw = (lag_steps ** (shape - 1.0)) * np.exp(-lag_steps / scale)
    if not np.isfinite(raw).all() or raw.sum() <= 0:
        raw = np.ones(max_lag_steps)
    return raw / raw.sum()


def add_dynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(DATETIME_COLUMN, kind="stable").reset_index(drop=True).copy()
    median_flow = float(df["R/W FLOW"].median())
    median_level = float(df["C/W WELL LEVEL"].median())

    flow = df["R/W FLOW"].replace(0, np.nan).fillna(median_flow).to_numpy(dtype=float)
    level = df["C/W WELL LEVEL"].fillna(median_level).to_numpy(dtype=float)
    tau_hours = 6.0 * (level / median_level) * (median_flow / flow)
    tau_hours = np.clip(tau_hours, 2.0, 12.0)
    df["RTD_TAU_HOURS"] = tau_hours

    for source in RTD_SOURCE_COLUMNS:
        values = df[source].to_numpy(dtype=float)
        rtd_values = np.full(len(df), np.nan)
        for i in range(len(df)):
            if i < RTD_MAX_LAG_STEPS:
                continue
            weights = gamma_rtd_weights(tau_hours[i] / 2.0)
            history = values[i - RTD_MAX_LAG_STEPS : i][::-1]
            if np.isnan(history).any():
                continue
            safe_name = source.replace("/", "_").replace(".", "").replace(" ", "_")
            rtd_values[i] = float(np.dot(weights, history))
        df[f"RTD_{safe_name}"] = rtd_values

    for source in ["R/W NTU", "FILT. NTU", "ALUM", "NTU", "R/W FLOW"]:
        safe_name = source.replace("/", "_").replace(".", "").replace(" ", "_")
        for step in LAG_STEPS:
            df[f"{safe_name}_LAG_{step * 2}H"] = df[source].shift(step)
        df[f"{safe_name}_DIFF_2H"] = df[source] - df[source].shift(1)
        df[f"{safe_name}_ROLL_MEAN_12H"] = df[source].shift(1).rolling(6).mean()
        df[f"{safe_name}_ROLL_STD_12H"] = df[source].shift(1).rolling(6).std()

    df["HOUR"] = df[DATETIME_COLUMN].dt.hour
    df["MONTH"] = df[DATETIME_COLUMN].dt.month
    df["SIN_HOUR"] = np.sin(2 * np.pi * df["HOUR"] / 24.0)
    df["COS_HOUR"] = np.cos(2 * np.pi * df["HOUR"] / 24.0)
    df["SIN_MONTH"] = np.sin(2 * np.pi * df["MONTH"] / 12.0)
    df["COS_MONTH"] = np.cos(2 * np.pi * df["MONTH"] / 12.0)

    df["PHYS_BASE_NTU"] = (
        0.45 * df["RTD_FILT_NTU"]
        + 0.35 * df["RTD_NTU"]
        + 0.15 * df["NTU_LAG_2H"]
        + 0.05 * df["FILT. NTU"]
    )
    return df


def make_feature_columns(df: pd.DataFrame) -> list[str]:
    generated = [
        column
        for column in df.columns
        if column.startswith(("RTD_", "R_W_", "FILT_", "ALUM_", "NTU_", "SIN_", "COS_"))
    ]
    excluded = {"NTU", OBSERVED_TARGET_COLUMN}
    return [*BASE_FEATURE_COLUMNS, *[column for column in generated if column not in excluded], "PHYS_BASE_NTU"]


def evaluate(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def evaluate_by_month(rows: pd.DataFrame, target_column: str, predictions: np.ndarray) -> dict[str, dict]:
    if rows.empty:
        return {}
    month_frame = pd.DataFrame(
        {
            "target_month": rows[DATETIME_COLUMN].shift(0).to_numpy(),
            "target_datetime": rows["_TARGET_DATETIME_FOR_METRICS"].to_numpy(),
            "actual": rows[target_column].to_numpy(),
            "predicted": predictions,
        }
    ).dropna(subset=["actual"])
    if month_frame.empty:
        return {}
    output = {}
    for month, month_rows in month_frame.groupby(month_frame["target_datetime"].dt.strftime("%Y-%m")):
        output[month] = {
            "rows": int(len(month_rows)),
            **evaluate(month_rows["actual"], month_rows["predicted"].to_numpy()),
        }
    return output


def train_models(feature_df: pd.DataFrame, feature_columns: list[str]) -> tuple[list[ModelBundle], dict, pd.DataFrame]:
    bundles: list[ModelBundle] = []
    metrics: dict[str, dict] = {}
    prediction_frames = []

    for step in HORIZON_STEPS:
        horizon_hours = step * 2
        target_future = f"TARGET_NTU_PLUS_{horizon_hours}H"
        residual_target = f"TARGET_RESIDUAL_PLUS_{horizon_hours}H"
        feature_df[target_future] = feature_df[OBSERVED_TARGET_COLUMN].shift(-step)
        feature_df[f"TARGET_DATETIME_PLUS_{horizon_hours}H"] = feature_df[DATETIME_COLUMN].shift(-step)
        feature_df[residual_target] = feature_df[target_future] - feature_df["PHYS_BASE_NTU"]

        train_rows = feature_df[
            (feature_df["DATASET"] == "train")
            & feature_df[feature_columns].notna().all(axis=1)
            & feature_df[residual_target].notna()
        ].copy()
        holdout_rows = feature_df[
            (feature_df["DATASET"] == "validation")
            & feature_df[feature_columns].notna().all(axis=1)
        ].copy()

        model = build_xgboost_model()
        model.fit(train_rows[feature_columns], train_rows[residual_target])
        bundles.append(ModelBundle(horizon_hours, model, feature_columns))

        train_pred = train_rows["PHYS_BASE_NTU"].to_numpy() + model.predict(train_rows[feature_columns])
        horizon_metrics = {
            "train_rows": int(len(train_rows)),
            "validation_prediction_rows": int(len(holdout_rows)),
            "validation_labeled_rows": int(holdout_rows[target_future].notna().sum()),
            "train": evaluate(train_rows[target_future], train_pred),
        }
        labeled_holdout_rows = holdout_rows.dropna(subset=[target_future]).copy()
        if labeled_holdout_rows.empty:
            horizon_metrics["validation_labeled"] = None
        else:
            labeled_holdout_pred = labeled_holdout_rows["PHYS_BASE_NTU"].to_numpy() + model.predict(labeled_holdout_rows[feature_columns])
            horizon_metrics["validation_labeled"] = evaluate(labeled_holdout_rows[target_future], labeled_holdout_pred)
            month_metric_rows = labeled_holdout_rows.copy()
            month_metric_rows["_TARGET_DATETIME_FOR_METRICS"] = month_metric_rows[f"TARGET_DATETIME_PLUS_{horizon_hours}H"]
            horizon_metrics["validation_labeled_by_target_month"] = evaluate_by_month(
                month_metric_rows,
                target_future,
                labeled_holdout_pred,
            )
        if not holdout_rows.empty:
            holdout_pred = holdout_rows["PHYS_BASE_NTU"].to_numpy() + model.predict(holdout_rows[feature_columns])
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "origin_datetime": holdout_rows[DATETIME_COLUMN].to_numpy(),
                        "target_datetime": holdout_rows[DATETIME_COLUMN].to_numpy()
                        + np.timedelta64(horizon_hours, "h"),
                        "horizon_hours": horizon_hours,
                        "actual_NTU": holdout_rows[target_future].to_numpy(),
                        "predicted_NTU": holdout_pred,
                        "physical_base_NTU": holdout_rows["PHYS_BASE_NTU"].to_numpy(),
                    }
                )
            )
        metrics[f"{horizon_hours}h"] = horizon_metrics

    holdout_predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return bundles, metrics, holdout_predictions


def predict_for_rows(df: pd.DataFrame, rows: pd.DataFrame, bundles: list[ModelBundle]) -> pd.DataFrame:
    output_rows = []
    indexed = df.set_index(DATETIME_COLUMN, drop=False)
    for _, origin in rows.iterrows():
        for bundle in bundles:
            if origin[bundle.feature_columns].isna().any() or pd.isna(origin["PHYS_BASE_NTU"]):
                continue
            predicted = float(origin["PHYS_BASE_NTU"] + bundle.model.predict(feature_frame_from_row(origin, bundle.feature_columns))[0])
            target_datetime = origin[DATETIME_COLUMN] + pd.Timedelta(hours=bundle.horizon_hours)
            actual = indexed.loc[target_datetime, TARGET_COLUMN] if target_datetime in indexed.index else np.nan
            output_rows.append(
                {
                    "origin_datetime": origin[DATETIME_COLUMN],
                    "target_datetime": target_datetime,
                    "horizon_hours": bundle.horizon_hours,
                    "predicted_NTU": predicted,
                    "actual_NTU": actual,
                    "physical_base_NTU": origin["PHYS_BASE_NTU"],
                }
            )
    return pd.DataFrame(output_rows)


def feature_frame_from_row(row: pd.Series, feature_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame([row[feature_columns].astype(float).to_dict()], columns=feature_columns)


def safe_feature_name(source: str) -> str:
    return source.replace("/", "_").replace(".", "").replace(" ", "_")


def build_single_feature_row(
    df: pd.DataFrame,
    idx: int,
    feature_columns: list[str],
    median_flow: float,
    median_level: float,
) -> pd.Series:
    row = df.loc[idx].copy()
    flow = float(row["R/W FLOW"]) if pd.notna(row["R/W FLOW"]) and float(row["R/W FLOW"]) != 0 else median_flow
    level = float(row["C/W WELL LEVEL"]) if pd.notna(row["C/W WELL LEVEL"]) else median_level
    tau_hours = float(np.clip(6.0 * (level / median_level) * (median_flow / flow), 2.0, 12.0))
    row["RTD_TAU_HOURS"] = tau_hours

    for source in RTD_SOURCE_COLUMNS:
        safe_name = safe_feature_name(source)
        if idx < RTD_MAX_LAG_STEPS:
            row[f"RTD_{safe_name}"] = np.nan
            continue
        history = df.loc[idx - RTD_MAX_LAG_STEPS : idx - 1, source].to_numpy(dtype=float)[::-1]
        if np.isnan(history).any():
            row[f"RTD_{safe_name}"] = np.nan
        else:
            row[f"RTD_{safe_name}"] = float(np.dot(gamma_rtd_weights(tau_hours / 2.0), history))

    for source in ["R/W NTU", "FILT. NTU", "ALUM", "NTU", "R/W FLOW"]:
        safe_name = safe_feature_name(source)
        current = float(row[source]) if pd.notna(row[source]) else np.nan
        for step in LAG_STEPS:
            row[f"{safe_name}_LAG_{step * 2}H"] = (
                float(df.loc[idx - step, source]) if idx >= step and pd.notna(df.loc[idx - step, source]) else np.nan
            )
        previous = float(df.loc[idx - 1, source]) if idx >= 1 and pd.notna(df.loc[idx - 1, source]) else np.nan
        row[f"{safe_name}_DIFF_2H"] = current - previous if pd.notna(current) and pd.notna(previous) else np.nan
        rolling_values = df.loc[max(0, idx - 6) : idx - 1, source].to_numpy(dtype=float) if idx >= 1 else np.array([])
        row[f"{safe_name}_ROLL_MEAN_12H"] = float(np.mean(rolling_values)) if len(rolling_values) == 6 and not np.isnan(rolling_values).any() else np.nan
        row[f"{safe_name}_ROLL_STD_12H"] = float(np.std(rolling_values, ddof=1)) if len(rolling_values) == 6 and not np.isnan(rolling_values).any() else np.nan

    dt = row[DATETIME_COLUMN]
    row["HOUR"] = dt.hour
    row["MONTH"] = dt.month
    row["SIN_HOUR"] = np.sin(2 * np.pi * row["HOUR"] / 24.0)
    row["COS_HOUR"] = np.cos(2 * np.pi * row["HOUR"] / 24.0)
    row["SIN_MONTH"] = np.sin(2 * np.pi * row["MONTH"] / 12.0)
    row["COS_MONTH"] = np.cos(2 * np.pi * row["MONTH"] / 12.0)
    row["PHYS_BASE_NTU"] = (
        0.45 * row["RTD_FILT_NTU"]
        + 0.35 * row["RTD_NTU"]
        + 0.15 * row["NTU_LAG_2H"]
        + 0.05 * row["FILT. NTU"]
    )
    return row[list(dict.fromkeys([*feature_columns, "PHYS_BASE_NTU"]))]


def recursive_fill_missing_ntu(
    combined: pd.DataFrame,
    bundles: list[ModelBundle],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filled = combined.sort_values(DATETIME_COLUMN, kind="stable").reset_index(drop=True).copy()
    two_hour_bundle = next(bundle for bundle in bundles if bundle.horizon_hours == 2)
    datetime_to_index = {dt: idx for idx, dt in filled[DATETIME_COLUMN].items()}
    median_flow = float(filled["R/W FLOW"].median())
    median_level = float(filled["C/W WELL LEVEL"].median())
    fill_rows = []

    missing_indices = filled.index[
        (filled["DATASET"] == "validation") & filled[OBSERVED_TARGET_COLUMN].isna()
    ].tolist()
    for idx in missing_indices:
        target_datetime = filled.loc[idx, DATETIME_COLUMN]
        origin_datetime = target_datetime - pd.Timedelta(hours=2)
        if origin_datetime not in datetime_to_index:
            continue
        origin_idx = datetime_to_index[origin_datetime]
        origin = build_single_feature_row(
            filled,
            origin_idx,
            two_hour_bundle.feature_columns,
            median_flow,
            median_level,
        )
        if origin[two_hour_bundle.feature_columns].isna().any() or pd.isna(origin["PHYS_BASE_NTU"]):
            continue
        predicted = float(
            origin["PHYS_BASE_NTU"]
            + two_hour_bundle.model.predict(feature_frame_from_row(origin, two_hour_bundle.feature_columns))[0]
        )
        predicted = max(predicted, 0.0)
        filled.loc[idx, TARGET_COLUMN] = predicted
        filled.loc[idx, "NTU_WAS_RECURSIVELY_FILLED"] = True
        fill_rows.append(
            {
                "target_datetime": target_datetime,
                "origin_datetime": origin_datetime,
                "filled_NTU": predicted,
                "physical_base_NTU": float(origin["PHYS_BASE_NTU"]),
            }
        )

    return filled, pd.DataFrame(fill_rows)


def build_required_answer(feature_df: pd.DataFrame, bundles: list[ModelBundle]) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = feature_df[feature_df["DATASET"] == "validation"].copy()
    rolling_rows = []
    from_7_rows = []
    bundle_by_horizon = {bundle.horizon_hours: bundle for bundle in bundles}
    indexed = valid.set_index(DATETIME_COLUMN, drop=False)

    for date_text in TARGET_DATES:
        for time_value in TARGET_TIMES:
            time_text = str(time_value).zfill(4)
            target_dt = pd.Timestamp(f"{date_text} {time_text[:2]}:{time_text[2:]}:00")
            origin_dt = target_dt - pd.Timedelta(hours=2)
            if origin_dt not in indexed.index:
                continue
            origin = indexed.loc[origin_dt]
            bundle = bundle_by_horizon[2]
            predicted = float(origin["PHYS_BASE_NTU"] + bundle.model.predict(feature_frame_from_row(origin, bundle.feature_columns))[0])
            rolling_rows.append(
                {
                    "date": date_text,
                    "time": time_value,
                    "target_datetime": target_dt,
                    "origin_datetime": origin_dt,
                    "horizon_hours": 2,
                    "predicted_NTU": predicted,
                    "actual_NTU": indexed.loc[target_dt, OBSERVED_TARGET_COLUMN] if target_dt in indexed.index else np.nan,
                    "target_NTU_recursively_filled": indexed.loc[target_dt, TARGET_COLUMN] if target_dt in indexed.index else np.nan,
                }
            )

        origin_dt = pd.Timestamp(f"{date_text} 07:00:00")
        if origin_dt not in indexed.index:
            continue
        origin = indexed.loc[origin_dt]
        for horizon_hours in [2, 4, 6, 8, 10, 12]:
            target_dt = origin_dt + pd.Timedelta(hours=horizon_hours)
            bundle = bundle_by_horizon[horizon_hours]
            predicted = float(origin["PHYS_BASE_NTU"] + bundle.model.predict(feature_frame_from_row(origin, bundle.feature_columns))[0])
            from_7_rows.append(
                {
                    "date": date_text,
                    "origin_time": 700,
                    "target_time": int(target_dt.strftime("%H%M")),
                    "origin_datetime": origin_dt,
                    "target_datetime": target_dt,
                    "horizon_hours": horizon_hours,
                    "predicted_NTU": predicted,
                    "actual_NTU": indexed.loc[target_dt, OBSERVED_TARGET_COLUMN] if target_dt in indexed.index else np.nan,
                    "target_NTU_recursively_filled": indexed.loc[target_dt, TARGET_COLUMN] if target_dt in indexed.index else np.nan,
                }
            )

    return pd.DataFrame(rolling_rows), pd.DataFrame(from_7_rows)


def build_hourly_answer(rolling_answer: pd.DataFrame, from_7_answer: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date_text in TARGET_DATES:
        rolling_day = rolling_answer[rolling_answer["date"] == date_text]
        from_7_day = from_7_answer[from_7_answer["date"] == date_text]
        known_points = {}
        row_7 = rolling_day[rolling_day["time"] == 700]
        if not row_7.empty:
            known_points[7] = float(row_7.iloc[0]["predicted_NTU"])
        for _, row in from_7_day.iterrows():
            hour = int(str(int(row["target_time"])).zfill(4)[:2])
            known_points[hour] = float(row["predicted_NTU"])
        if not known_points:
            continue
        series = pd.Series(known_points).sort_index()
        hourly_index = pd.Index(range(7, 20), name="hour")
        hourly = series.reindex(hourly_index).interpolate(method="linear")
        for hour, predicted in hourly.items():
            rows.append(
                {
                    "date": date_text,
                    "time": f"{hour:02d}:00",
                    "target_datetime": pd.Timestamp(f"{date_text} {hour:02d}:00:00"),
                    "horizon_hours_from_07": hour - 7,
                    "predicted_NTU": float(predicted),
                    "source": "direct_2h_model" if hour in known_points else "linear_interpolation",
                }
            )
    return pd.DataFrame(rows)


def build_sensitivity(feature_df: pd.DataFrame, bundles: list[ModelBundle]) -> pd.DataFrame:
    valid = feature_df[feature_df["DATASET"] == "validation"].copy()
    origin_rows = valid[
        valid[DATETIME_COLUMN].isin([pd.Timestamp(f"{date} 07:00:00") for date in TARGET_DATES])
    ].copy()

    scenarios = [
        ("raw_NTU_plus_20pct", "R/W NTU", "R_W_NTU", 1.20),
        ("raw_NTU_plus_50pct", "R/W NTU", "R_W_NTU", 1.50),
        ("alum_plus_10pct", "ALUM", "ALUM", 1.10),
        ("alum_minus_10pct", "ALUM", "ALUM", 0.90),
    ]

    rows = []
    for _, origin in origin_rows.iterrows():
        for bundle in bundles:
            if origin[bundle.feature_columns].isna().any():
                continue
            base_pred = float(origin["PHYS_BASE_NTU"] + bundle.model.predict(feature_frame_from_row(origin, bundle.feature_columns))[0])
            rows.append(
                {
                    "origin_datetime": origin[DATETIME_COLUMN],
                    "horizon_hours": bundle.horizon_hours,
                    "scenario": "baseline",
                    "predicted_NTU": base_pred,
                    "delta_vs_baseline": 0.0,
                    "elasticity": np.nan,
                }
            )
            for scenario_name, column, feature_prefix, factor in scenarios:
                scenario_origin = origin.copy()
                original_value = float(scenario_origin[column])
                scenario_origin[column] = original_value * factor
                scenario_feature_columns = [
                    feature
                    for feature in bundle.feature_columns
                    if feature == column
                    or feature.startswith(f"{feature_prefix}_")
                    or feature.startswith(f"RTD_{feature_prefix}")
                ]
                for feature in scenario_feature_columns:
                    if pd.notna(scenario_origin[feature]):
                        scenario_origin[feature] = float(scenario_origin[feature]) * factor
                scenario_pred = float(
                    scenario_origin["PHYS_BASE_NTU"]
                    + bundle.model.predict(feature_frame_from_row(scenario_origin, bundle.feature_columns))[0]
                )
                input_delta_pct = factor - 1.0
                rows.append(
                    {
                        "origin_datetime": origin[DATETIME_COLUMN],
                        "horizon_hours": bundle.horizon_hours,
                        "scenario": scenario_name,
                        "predicted_NTU": scenario_pred,
                        "delta_vs_baseline": scenario_pred - base_pred,
                        "elasticity": ((scenario_pred - base_pred) / max(abs(base_pred), 1e-9)) / input_delta_pct,
                    }
                )
    return pd.DataFrame(rows)


def write_outputs(
    feature_df: pd.DataFrame,
    bundles: list[ModelBundle],
    metrics: dict,
    holdout_predictions: pd.DataFrame,
    recursive_fill: pd.DataFrame,
    rolling_answer: pd.DataFrame,
    from_7_answer: pd.DataFrame,
    hourly_answer: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_columns = bundles[0].feature_columns

    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    holdout_predictions.to_csv(OUTPUT_DIR / "validation_multihorizon_predictions.csv", index=False, encoding="utf-8-sig")
    recursive_fill.to_csv(OUTPUT_DIR / "recursive_filled_february_ntu.csv", index=False, encoding="utf-8-sig")
    rolling_answer.to_csv(OUTPUT_DIR / "question3_required_predictions_rolling_2h.csv", index=False, encoding="utf-8-sig")
    from_7_answer.to_csv(OUTPUT_DIR / "question3_forecast_from_07.csv", index=False, encoding="utf-8-sig")
    hourly_answer.to_csv(OUTPUT_DIR / "question3_hourly_7_to_19_interpolated.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(OUTPUT_DIR / "sensitivity_scenarios.csv", index=False, encoding="utf-8-sig")

    importance_rows = []
    for bundle in bundles:
        if hasattr(bundle.model, "feature_importances_"):
            for feature, importance in zip(feature_columns, bundle.model.feature_importances_):
                importance_rows.append(
                    {
                        "horizon_hours": bundle.horizon_hours,
                        "feature": feature,
                        "importance": float(importance),
                    }
                )
    importance = pd.DataFrame(importance_rows).sort_values(["horizon_hours", "importance"], ascending=[True, False])
    importance.to_csv(OUTPUT_DIR / "feature_importance_by_horizon.csv", index=False, encoding="utf-8-sig")

    model_dir = OUTPUT_DIR / "models"
    model_dir.mkdir(exist_ok=True)
    for bundle in bundles:
        joblib.dump(bundle.model, model_dir / f"xgboost_residual_plus_{bundle.horizon_hours}h.joblib")

    dynamic_columns = [
        "DATASET",
        "DATE",
        "DATE_INT",
        "TIME",
        DATETIME_COLUMN,
        *BASE_FEATURE_COLUMNS,
        OBSERVED_TARGET_COLUMN,
        TARGET_COLUMN,
        "NTU_WAS_RECURSIVELY_FILLED",
        *feature_columns,
    ]
    dynamic_columns = list(dict.fromkeys(dynamic_columns))
    feature_df[dynamic_columns].to_csv(
        OUTPUT_DIR / "dynamic_training_table.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows = []
    month_summary_rows = []
    for horizon, horizon_metrics in metrics.items():
        if "train" not in horizon_metrics:
            continue
        validation = horizon_metrics["validation_labeled"] or {}
        summary_rows.append(
            {
                "horizon": horizon,
                "train_mae": horizon_metrics["train"]["mae"],
                "train_rmse": horizon_metrics["train"]["rmse"],
                "train_r2": horizon_metrics["train"]["r2"],
                "validation_mae": validation.get("mae"),
                "validation_rmse": validation.get("rmse"),
                "validation_r2": validation.get("r2"),
            }
        )
        for month, month_metrics in horizon_metrics.get("validation_labeled_by_target_month", {}).items():
            month_summary_rows.append(
                {
                    "horizon": horizon,
                    "target_month": month,
                    "rows": month_metrics["rows"],
                    "validation_mae": month_metrics["mae"],
                    "validation_rmse": month_metrics["rmse"],
                    "validation_r2": month_metrics["r2"],
                }
            )
    summary = pd.DataFrame(summary_rows)
    month_summary = pd.DataFrame(month_summary_rows)

    with pd.ExcelWriter(OUTPUT_DIR / "question3_answer.xlsx", engine="openpyxl") as writer:
        rolling_answer.to_excel(writer, sheet_name="rolling_7_19", index=False)
        from_7_answer.to_excel(writer, sheet_name="forecast_from_07", index=False)
        hourly_answer.to_excel(writer, sheet_name="hourly_interpolated", index=False)
        recursive_fill.to_excel(writer, sheet_name="recursive_fill_feb", index=False)
        summary.to_excel(writer, sheet_name="metrics", index=False)
        month_summary.to_excel(writer, sheet_name="metrics_by_month", index=False)
        sensitivity.to_excel(writer, sheet_name="sensitivity", index=False)
        importance.groupby("horizon_hours").head(12).to_excel(writer, sheet_name="feature_importance", index=False)

    report = [
        "# Question 3 Hybrid Dynamic Model",
        "",
        "Data are sampled every 2 hours, so direct forecast horizons are 2, 4, 6, 8, 10, and 12 hours.",
        "The workbook also contains an hourly 7:00-19:00 table, linearly interpolated between direct",
        "2-hour-grid forecasts.",
        "Historical treated-water NTU is used as an autoregressive state. Missing February 2026 NTU",
        "values are filled chronologically by the 2-hour model before the 2-12 hour forecasts are made.",
        "The physical layer uses a Gamma residence-time-distribution over the previous 24 hours, with",
        "the mean residence time adjusted by clear-well level and raw-water flow. XGBoost models then",
        "learn the residual between the physical baseline and each future NTU target.",
        "",
        "Main output: `question3_answer.xlsx`.",
    ]
    (OUTPUT_DIR / "question3_method_summary.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    _, _, combined = load_data()
    initial_feature_df = add_dynamic_features(combined)
    feature_columns = make_feature_columns(initial_feature_df)
    initial_bundles, _, _ = train_models(initial_feature_df, feature_columns)

    filled_combined, recursive_fill = recursive_fill_missing_ntu(combined, initial_bundles)
    feature_df = add_dynamic_features(filled_combined)
    feature_columns = make_feature_columns(feature_df)
    bundles, metrics, holdout_predictions = train_models(feature_df, feature_columns)
    metrics["recursive_fill"] = {
        "filled_rows": int(len(recursive_fill)),
        "filled_period_start": str(recursive_fill["target_datetime"].min()) if not recursive_fill.empty else None,
        "filled_period_end": str(recursive_fill["target_datetime"].max()) if not recursive_fill.empty else None,
        "filled_ntu_min": float(recursive_fill["filled_NTU"].min()) if not recursive_fill.empty else None,
        "filled_ntu_max": float(recursive_fill["filled_NTU"].max()) if not recursive_fill.empty else None,
    }
    rolling_answer, from_7_answer = build_required_answer(feature_df, bundles)
    hourly_answer = build_hourly_answer(rolling_answer, from_7_answer)
    sensitivity = build_sensitivity(feature_df, bundles)
    write_outputs(
        feature_df,
        bundles,
        metrics,
        holdout_predictions,
        recursive_fill,
        rolling_answer,
        from_7_answer,
        hourly_answer,
        sensitivity,
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nWrote outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
