"""Preprocess Question 1 data for NTU modeling.

This script applies the current feature and missing-value strategy:
- drop downstream/companion-state variables not used as candidate features;
- remove REMARKS from model features;
- read outlier points from outputs/potential_outliers.xlsx;
- set outlier values to missing before interpolation;
- when R/W NTU is marked as an outlier, also set R/W CLR missing for that row;
- time-interpolate feature variables and target NTU;
- output a continuous model-data CSV without dropping missing target rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "data_for_question1.xlsx"
DEFAULT_OUTLIERS = Path(__file__).resolve().parents[1] / "outputs" / "potential_outliers.xlsx"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "train_for_question1.csv"

TARGET_COLUMN = "NTU"

MODEL_FEATURE_COLUMNS = [
    "RIVER LEVEL",
    "R/W FLOW",
    "R/W NTU",
    "R/W CLR",
    "R/W PH",
    "FILT. NTU",
    "C/W WELL LEVEL",
    "ALUM",
]

INTERPOLATE_COLUMNS = [*MODEL_FEATURE_COLUMNS, TARGET_COLUMN]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess data_for_question1.xlsx for Question 1 modeling."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input Excel path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path for model data. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--outliers",
        type=Path,
        default=DEFAULT_OUTLIERS,
        help=f"Outlier Excel file. Default: {DEFAULT_OUTLIERS}",
    )
    return parser.parse_args()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().upper() for col in df.columns]
    return df


def build_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_text = (
        df["DATE"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    date_part = pd.to_datetime(date_text, format="%Y%m%d", errors="coerce")
    fallback = pd.to_datetime(df["DATE"], errors="coerce")
    date_part = date_part.fillna(fallback)

    time_numeric = pd.to_numeric(df["TIME"], errors="coerce").astype("Int64")
    # The source file has a few 17:00 records recorded as 5. The sampling
    # schedule is every two hours from 01:00 to 23:00, so normalize them here.
    time_numeric = time_numeric.mask(time_numeric == 5, 1700)
    time_text = time_numeric.astype("string").str.zfill(4)
    hour = pd.to_numeric(time_text.str[:2], errors="coerce")
    minute = pd.to_numeric(time_text.str[2:], errors="coerce")

    df["DATE"] = date_part
    df["DATE_INT"] = df["DATE"].dt.strftime("%Y%m%d").astype("Int64")
    df["TIME"] = time_numeric
    df["DATETIME"] = df["DATE"] + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m")
    return df


def clean_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace({"-": pd.NA, "": pd.NA, " ": pd.NA})
    for col in df.columns:
        if col in {"DATE", "DATE_INT", "TIME", "DATETIME", "REMARKS"}:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_outliers(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Outlier file not found: {path}")

    outliers = pd.read_excel(path, sheet_name=0)
    outliers.columns = [str(col).strip().upper() for col in outliers.columns]
    required = {"DATE", "TIME", "COLUMN"}
    missing = sorted(required - set(outliers.columns))
    if missing:
        raise ValueError(f"Outlier file is missing required columns: {missing}")

    outliers["DATE_INT"] = pd.to_numeric(outliers["DATE"], errors="coerce").astype("Int64")
    outliers["TIME"] = pd.to_numeric(outliers["TIME"], errors="coerce").astype("Int64")
    outliers["COLUMN"] = outliers["COLUMN"].astype("string").str.strip().str.upper()
    return outliers


def apply_outliers(
    df: pd.DataFrame, outliers: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    records: list[dict] = []

    for _, outlier in outliers.iterrows():
        column = outlier["COLUMN"]
        mask = (df["DATE_INT"] == outlier["DATE_INT"]) & (df["TIME"] == outlier["TIME"])
        matching_indices = list(df.index[mask])
        if not matching_indices:
            continue

        for row_index in matching_indices:
            columns_to_clear = [column]
            if column == "R/W NTU" and "R/W CLR" in df.columns:
                columns_to_clear.append("R/W CLR")

            for clear_column in columns_to_clear:
                if clear_column not in df.columns:
                    continue
                old_value = df.at[row_index, clear_column]
                df.at[row_index, clear_column] = pd.NA
                records.append(
                    {
                        "row_index": int(row_index),
                        "DATE_INT": df.at[row_index, "DATE_INT"],
                        "TIME": df.at[row_index, "TIME"],
                        "DATETIME": df.at[row_index, "DATETIME"],
                        "source_column": column,
                        "cleared_column": clear_column,
                        "old_value": old_value,
                        "new_value": pd.NA,
                        "action": "set_missing_before_interpolation",
                    }
                )

    return df, pd.DataFrame(records)


def interpolate_model_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    records: list[dict] = []

    df = df.sort_values("DATETIME", kind="stable").reset_index(drop=True)
    df = df.set_index("DATETIME")

    for col in INTERPOLATE_COLUMNS:
        if col not in df.columns:
            continue
        before_missing = int(df[col].isna().sum())
        df[col] = df[col].interpolate(method="time", limit_direction="both")
        after_missing = int(df[col].isna().sum())
        records.append(
            {
                "column": col,
                "method": "time_linear_interpolation",
                "missing_before": before_missing,
                "missing_after": after_missing,
            }
        )

    df = df.reset_index()
    return df, pd.DataFrame(records)


def build_outputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep_columns = ["DATE", "DATE_INT", "TIME", "DATETIME", *MODEL_FEATURE_COLUMNS, TARGET_COLUMN]
    missing_columns = [col for col in keep_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns after preprocessing: {missing_columns}")

    processed = df[keep_columns].copy()
    processed["TARGET_NTU_MISSING"] = processed[TARGET_COLUMN].isna().astype(int)

    model_data = processed.copy()
    return processed, model_data


def main() -> None:
    args = parse_args()
    df = pd.read_excel(args.input)
    df = normalize_columns(df)
    df = build_datetime(df)
    df = clean_values(df)
    df = df.sort_values("DATETIME", kind="stable").reset_index(drop=True)
    input_rows = len(df)

    outliers = load_outliers(args.outliers)
    df, outlier_changes = apply_outliers(df, outliers)
    df, interpolation_summary = interpolate_model_columns(df)
    processed, model_data = build_outputs(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model_data.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Input rows: {input_rows}")
    print(f"Output rows: {len(model_data)}")
    print(f"Outlier values set missing before interpolation: {len(outlier_changes)}")
    print("Missing values after interpolation:")
    print(interpolation_summary[["column", "missing_after"]].to_string(index=False))
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
