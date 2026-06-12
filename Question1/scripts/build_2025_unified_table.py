"""Build one unified table from all 2025 Excel files.

The source files are monthly workbooks with inconsistent headers and some dates
that Excel/Pandas read as month/day instead of day/month. This script uses the
month encoded in each file name and reconstructs dates as YYYYMMDD integers.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = (
    PROJECT_ROOT / "A题 自来水厂水质预测与评估" / "附件" / "附件1  2025数据集"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "Question1" / "data_2025_unified.xlsx"

MONTH_NAME_TO_NUMBER = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUNE": 6,
    "JUL": 7,
    "JULY": 7,
    "AUG": 8,
    "SEP": 9,
    "SEPT": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}

CANONICAL_COLUMNS = [
    "DATE",
    "DATE_INT",
    "TIME",
    "DATETIME",
    "SOURCE_FILE",
    "RIVER LEVEL",
    "R/W PUMP DUTY",
    "R/W FLOW",
    "R/W NTU",
    "R/W CLR",
    "R/W PH",
    "FILT. NTU",
    "C/W WELL LEVEL",
    "PH",
    "NTU",
    "CLR",
    "CL2",
    "F/RIDE",
    "ALUM",
    "T/W PUMP DUTY",
    "T/W FLOW",
    "18ML LEVEL",
    "18ML FLOW",
    "REMARKS",
]

POSITIONAL_COLUMNS_21 = [
    "DATE",
    "TIME",
    "RIVER LEVEL",
    "R/W PUMP DUTY",
    "R/W FLOW",
    "R/W NTU",
    "R/W CLR",
    "R/W PH",
    "FILT. NTU",
    "C/W WELL LEVEL",
    "PH",
    "NTU",
    "CLR",
    "CL2",
    "F/RIDE",
    "ALUM",
    "T/W PUMP DUTY",
    "T/W FLOW",
    "18ML LEVEL",
    "18ML FLOW",
    "REMARKS",
]

POSITIONAL_COLUMNS_15 = [
    "DATE",
    "TIME",
    "RIVER LEVEL",
    "R/W PUMP DUTY",
    "R/W FLOW",
    "R/W NTU",
    "R/W CLR",
    "R/W PH",
    "FILT. NTU",
    "C/W WELL LEVEL",
    "PH",
    "NTU",
    "CLR",
    "CL2",
    "T/W FLOW",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge all 2025 monthly workbooks into one normalized table."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"Directory containing 2025 Excel files. Default: {DEFAULT_SOURCE_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output Excel path. Default: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def month_from_filename(path: Path) -> int:
    upper_name = path.stem.upper()
    for month_name, month_number in MONTH_NAME_TO_NUMBER.items():
        if month_name in upper_name:
            return month_number
    raise ValueError(f"Cannot infer month from file name: {path.name}")


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    if len(df.columns) == len(POSITIONAL_COLUMNS_21):
        df = df.copy()
        df.columns = POSITIONAL_COLUMNS_21
        return df

    if len(df.columns) == len(POSITIONAL_COLUMNS_15):
        df = df.copy()
        df.columns = POSITIONAL_COLUMNS_15
        return df

    df = df.copy()
    rename_map = {
        "DATA": "DATE",
        "DATE": "DATE",
        "TIME": "TIME",
        "RIVER LEVEL": "RIVER LEVEL",
    }
    df.columns = [rename_map.get(str(col).strip().upper(), str(col).strip().upper()) for col in df.columns]
    return df


def parse_excel_serial(value) -> pd.Timestamp | pd.NaT:
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value):
        return pd.NaT
    return pd.to_datetime(numeric_value, unit="D", origin="1899-12-30", errors="coerce")


def extract_day(value, file_month: int) -> int | None:
    if isinstance(value, (pd.Timestamp, datetime, date)):
        parsed = pd.Timestamp(value)
    else:
        numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(numeric_value):
            parsed = parse_excel_serial(value)
        else:
            parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return None

    parsed = pd.Timestamp(parsed)

    # Several source files store Apr 1 as 2025-01-04, Sep 1 as 2025-01-09,
    # etc. In that case, the source month appears in the day component and
    # the real day appears in the month component.
    if parsed.year == 2025 and parsed.day == file_month and parsed.month != file_month:
        return int(parsed.month)

    return int(parsed.day)


def normalize_time(value) -> int | None:
    if pd.isna(value):
        return None
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value):
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        if not digits:
            return None
        numeric_value = int(digits)
    return int(numeric_value)


def build_dates(df: pd.DataFrame, file_month: int) -> pd.DataFrame:
    df = df.copy()
    days = df["DATE"].map(lambda value: extract_day(value, file_month))
    df["DATE"] = pd.to_datetime(
        {
            "year": 2025,
            "month": file_month,
            "day": days,
        },
        errors="coerce",
    )
    df["DATE_INT"] = df["DATE"].dt.strftime("%Y%m%d").astype("Int64")

    df["TIME"] = df["TIME"].map(normalize_time).astype("Int64")
    time_text = df["TIME"].astype("string").str.zfill(4)
    hour = pd.to_numeric(time_text.str[:2], errors="coerce")
    minute = pd.to_numeric(time_text.str[2:], errors="coerce")
    df["DATETIME"] = df["DATE"] + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m")
    return df


def clean_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace({"-": pd.NA, "": pd.NA, " ": pd.NA})
    for col in df.columns:
        if col in {"DATE", "DATE_INT", "TIME", "DATETIME", "SOURCE_FILE", "REMARKS"}:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_month_file(path: Path) -> pd.DataFrame:
    file_month = month_from_filename(path)
    df = pd.read_excel(path)
    df = normalize_headers(df)
    df = clean_values(df)
    df = build_dates(df, file_month)
    df["SOURCE_FILE"] = path.name

    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    return df[CANONICAL_COLUMNS]


def main() -> None:
    args = parse_args()
    files = sorted(
        path
        for path in args.source_dir.glob("*.xlsx")
        if not path.name.startswith("~$")
    )
    if not files:
        raise FileNotFoundError(f"No .xlsx files found in {args.source_dir}")

    frames = [read_month_file(path) for path in files]
    unified = pd.concat(frames, ignore_index=True)
    unified = unified.sort_values(["DATETIME", "SOURCE_FILE"], kind="stable").reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        unified.to_excel(writer, sheet_name="data_2025", index=False)
        (
            unified.groupby("SOURCE_FILE", dropna=False)
            .agg(
                rows=("SOURCE_FILE", "size"),
                min_date=("DATE_INT", "min"),
                max_date=("DATE_INT", "max"),
                missing_datetime=("DATETIME", lambda s: int(s.isna().sum())),
            )
            .reset_index()
            .to_excel(writer, sheet_name="source_summary", index=False)
        )

    print(f"Source files: {len(files)}")
    print(f"Output rows: {len(unified)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
