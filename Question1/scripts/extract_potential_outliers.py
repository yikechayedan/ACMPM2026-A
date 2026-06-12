"""Extract potential abnormal points from Question 1 data.

The script reports candidate abnormal observations instead of modifying the
source data. It combines three checks:
1. physical range rules;
2. global IQR fences;
3. rolling-window robust z-scores for isolated spikes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "data_for_question1.xlsx"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / "potential_outliers.xlsx"

NON_NEGATIVE_COLUMNS = {
    "RIVER LEVEL",
    "R/W FLOW",
    "R/W NTU",
    "R/W CLR",
    "FILT. NTU",
    "C/W WELL LEVEL",
    "NTU",
    "CLR",
    "CL2",
    "F/RIDE",
    "ALUM",
}

PH_COLUMNS = {"R/W PH", "PH"}

PHYSICAL_RANGES = {
    "R/W PH": (5.0, 10.0),
    "PH": (5.0, 10.0),
}

EXCLUDED_FROM_NUMERIC_CHECKS = {"DATE", "DATE_INT", "TIME", "DATETIME", "SOURCE_FILE", "REMARKS"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract potential abnormal points from data_for_question1.xlsx."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input Excel file. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output Excel file. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--iqr-multiplier",
        type=float,
        default=1.5,
        help="IQR fence multiplier. Default: 1.5",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=12,
        help="Rolling window size for robust z-score. Default: 12 rows.",
    )
    parser.add_argument(
        "--rolling-z-threshold",
        type=float,
        default=10.0,
        help="Robust rolling z-score threshold. Default: 10.0",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Also output missing numeric values as candidate records.",
    )
    return parser.parse_args()


def build_datetime(df: pd.DataFrame) -> pd.Series:
    date_text = (
        df["DATE"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    date_part = pd.to_datetime(date_text, format="%Y%m%d", errors="coerce")
    fallback = pd.to_datetime(df["DATE"], errors="coerce")
    date_part = date_part.fillna(fallback)
    time_text = (
        df["TIME"]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(4)
    )
    hour = pd.to_numeric(time_text.str[:2], errors="coerce")
    minute = pd.to_numeric(time_text.str[2:], errors="coerce")

    return date_part + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m")


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if col not in EXCLUDED_FROM_NUMERIC_CHECKS and pd.api.types.is_numeric_dtype(df[col])
    ]


def add_record(
    records: list[dict],
    df: pd.DataFrame,
    row_index: int,
    column: str,
    value,
    method: str,
    reason: str,
    lower=None,
    upper=None,
    score=None,
) -> None:
    records.append(
        {
            "excel_row": row_index + 2,
            "row_index": row_index,
            "datetime": df.at[row_index, "datetime"],
            "date": df.at[row_index, "DATE"] if "DATE" in df else None,
            "time": df.at[row_index, "TIME"] if "TIME" in df else None,
            "column": column,
            "value": value,
            "method": method,
            "reason": reason,
            "lower_bound": lower,
            "upper_bound": upper,
            "score": score,
            "remarks": df.at[row_index, "REMARKS"] if "REMARKS" in df else None,
        }
    )


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [str(col).strip().upper() for col in df.columns]
    df = df.replace({"-": pd.NA, "": pd.NA, " ": pd.NA})

    for col in df.columns:
        if col not in {"DATE", "REMARKS"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["datetime"] = build_datetime(df)
    df = df.sort_values("datetime", kind="stable").reset_index(drop=True)
    return df


def detect_physical_rule_outliers(df: pd.DataFrame, columns: list[str]) -> list[dict]:
    records: list[dict] = []

    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce")

        if col in NON_NEGATIVE_COLUMNS:
            mask = series < 0
            for row_index, value in series[mask].items():
                add_record(
                    records,
                    df,
                    row_index,
                    col,
                    value,
                    "physical_rule",
                    "negative value is not physically meaningful",
                    lower=0,
                )

        if col in PHYSICAL_RANGES:
            lower, upper = PHYSICAL_RANGES[col]
            mask = series.notna() & ((series < lower) | (series > upper))
            for row_index, value in series[mask].items():
                add_record(
                    records,
                    df,
                    row_index,
                    col,
                    value,
                    "physical_rule",
                    "outside expected pH range",
                    lower=lower,
                    upper=upper,
                )

    bad_datetime = df["datetime"].isna()
    for row_index in df.index[bad_datetime]:
        add_record(
            records,
            df,
            row_index,
            "datetime",
            None,
            "physical_rule",
            "DATE/TIME cannot be parsed",
        )

    duplicate_datetime = df["datetime"].notna() & df["datetime"].duplicated(keep=False)
    for row_index in df.index[duplicate_datetime]:
        add_record(
            records,
            df,
            row_index,
            "datetime",
            df.at[row_index, "datetime"],
            "physical_rule",
            "duplicated datetime",
        )

    return records


def detect_iqr_outliers(
    df: pd.DataFrame, columns: list[str], multiplier: float
) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    stats: list[dict] = []

    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce")
        clean = series.dropna()
        if clean.empty:
            continue

        q1 = clean.quantile(0.25)
        q3 = clean.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        mask = series.notna() & ((series < lower) | (series > upper))

        stats.append(
            {
                "column": col,
                "count": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "mean": clean.mean(),
                "std": clean.std(),
                "q1": q1,
                "median": clean.median(),
                "q3": q3,
                "iqr": iqr,
                "iqr_lower": lower,
                "iqr_upper": upper,
                "iqr_outlier_count": int(mask.sum()),
            }
        )

        if iqr == 0:
            continue

        for row_index, value in series[mask].items():
            add_record(
                records,
                df,
                row_index,
                col,
                value,
                "iqr",
                f"outside [{lower:.6g}, {upper:.6g}]",
                lower=lower,
                upper=upper,
            )

    return records, stats


def detect_rolling_spikes(
    df: pd.DataFrame, columns: list[str], window: int, threshold: float
) -> list[dict]:
    records: list[dict] = []
    window = max(3, window)

    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce")
        rolling_median = series.rolling(window=window, center=True, min_periods=3).median()
        residual = (series - rolling_median).abs()
        rolling_mad = residual.rolling(window=window, center=True, min_periods=3).median()
        robust_z = 0.6745 * residual / rolling_mad.replace(0, pd.NA)
        mask = robust_z.notna() & (robust_z > threshold)

        for row_index, value in series[mask].items():
            add_record(
                records,
                df,
                row_index,
                col,
                value,
                "rolling_robust_z",
                f"isolated spike versus local median, threshold={threshold}",
                score=robust_z.at[row_index],
            )

    return records


def detect_missing_values(df: pd.DataFrame, columns: list[str]) -> list[dict]:
    records: list[dict] = []

    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce")
        for row_index in series.index[series.isna()]:
            add_record(
                records,
                df,
                row_index,
                col,
                None,
                "missing",
                "numeric value is missing",
            )

    return records


def summarize_records(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["column", "method", "count"])

    return (
        pd.DataFrame(records)
        .groupby(["column", "method"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["column", "method"])
    )


def build_iqr_and_rolling_intersection(outliers: pd.DataFrame) -> pd.DataFrame:
    if outliers.empty:
        return pd.DataFrame(
            columns=[
                "excel_row",
                "row_index",
                "datetime",
                "date",
                "time",
                "column",
                "value",
                "methods",
                "iqr_lower_bound",
                "iqr_upper_bound",
                "rolling_score",
                "remarks",
            ]
        )

    target = outliers[outliers["method"].isin(["iqr", "rolling_robust_z"])].copy()
    if target.empty:
        return pd.DataFrame()

    method_sets = (
        target.groupby(["row_index", "column"])["method"]
        .agg(lambda values: set(values))
        .reset_index(name="method_set")
    )
    both_keys = method_sets[
        method_sets["method_set"].map(lambda methods: {"iqr", "rolling_robust_z"}.issubset(methods))
    ][["row_index", "column"]]

    if both_keys.empty:
        return pd.DataFrame()

    both_records = target.merge(both_keys, on=["row_index", "column"], how="inner")

    def first_non_missing(series: pd.Series):
        non_missing = series.dropna()
        if non_missing.empty:
            return pd.NA
        return non_missing.iloc[0]

    intersection = (
        both_records.groupby(["row_index", "column"], as_index=False)
        .agg(
            excel_row=("excel_row", "first"),
            datetime=("datetime", "first"),
            date=("date", "first"),
            time=("time", "first"),
            value=("value", "first"),
            methods=("method", lambda values: ",".join(sorted(set(values)))),
            iqr_lower_bound=("lower_bound", first_non_missing),
            iqr_upper_bound=("upper_bound", first_non_missing),
            rolling_score=("score", first_non_missing),
            remarks=("remarks", first_non_missing),
        )
        .sort_values(["datetime", "column"])
        .reset_index(drop=True)
    )

    return intersection[
        [
            "excel_row",
            "row_index",
            "datetime",
            "date",
            "time",
            "column",
            "value",
            "methods",
            "iqr_lower_bound",
            "iqr_upper_bound",
            "rolling_score",
            "remarks",
        ]
    ]


def build_notes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item": "purpose",
                "description": (
                    "This workbook lists candidate abnormal points for manual review; "
                    "it does not prove that the values are wrong."
                ),
            },
            {
                "item": "physical_rule",
                "description": (
                    "Flags impossible or suspicious records, such as negative values, "
                    "pH outside [5, 10], unparseable datetime, or duplicated datetime."
                ),
            },
            {
                "item": "iqr",
                "description": (
                    "Flags values outside Q1 - k*IQR and Q3 + k*IQR. For skewed water "
                    "quality variables, true high-turbidity operating periods can also "
                    "be flagged and should not be deleted mechanically."
                ),
            },
            {
                "item": "rolling_robust_z",
                "description": (
                    "Flags local spikes compared with a centered rolling median. These "
                    "points are useful for finding isolated sensor noise."
                ),
            },
            {
                "item": "recommended_use",
                "description": (
                    "Review candidates with surrounding time points and REMARKS. Treat "
                    "obvious data-entry or sensor errors as missing before interpolation; "
                    "keep continuous abnormal periods that match real operating conditions."
                ),
            },
            {
                "item": "iqr_and_rolling",
                "description": (
                    "The iqr_and_rolling sheet keeps only points flagged by both IQR and "
                    "rolling robust z-score. These are higher-priority candidates for "
                    "manual review."
                ),
            },
        ]
    )


def main() -> None:
    args = parse_args()
    df = load_data(args.input)
    columns = numeric_columns(df)

    records: list[dict] = []
    records.extend(detect_physical_rule_outliers(df, columns))
    iqr_records, column_stats = detect_iqr_outliers(df, columns, args.iqr_multiplier)
    records.extend(iqr_records)
    records.extend(
        detect_rolling_spikes(
            df,
            columns,
            window=args.rolling_window,
            threshold=args.rolling_z_threshold,
        )
    )
    if args.include_missing:
        records.extend(detect_missing_values(df, columns))

    outliers = pd.DataFrame(records)
    if not outliers.empty:
        outliers = outliers.sort_values(["datetime", "column", "method"]).reset_index(drop=True)

    iqr_and_rolling = build_iqr_and_rolling_intersection(outliers)
    summary = summarize_records(records)
    stats = pd.DataFrame(column_stats).sort_values("column")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        outliers.to_excel(writer, sheet_name="potential_outliers", index=False)
        iqr_and_rolling.to_excel(writer, sheet_name="iqr_and_rolling", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
        stats.to_excel(writer, sheet_name="column_stats", index=False)
        build_notes().to_excel(writer, sheet_name="notes", index=False)

    print(f"Input rows: {len(df)}")
    print(f"Checked numeric columns: {len(columns)}")
    print(f"Potential outlier records: {len(outliers)}")
    print(f"IQR and rolling robust z-score intersection: {len(iqr_and_rolling)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
