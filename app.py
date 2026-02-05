import re
from typing import Optional

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Call Insights Dashboard", page_icon="📊", layout="wide")


KEYWORD_MAP = {
    "country": ["country", "pais", "país", "nation", "region", "market"],
    "reason": ["reason", "motivo", "causa", "failure_reason", "disposition", "result_reason"],
    "status": ["status", "estado", "outcome", "result", "call_result", "call status", "disposition"],
    "datetime": ["datetime", "timestamp", "fecha", "date", "call_time", "time", "created"],
}

FAIL_PATTERNS = [
    "fail",
    "failed",
    "error",
    "drop",
    "busy",
    "no answer",
    "not answered",
    "unreachable",
    "declined",
    "cancel",
]


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def detect_column(columns: list[str], targets: list[str]) -> Optional[str]:
    normalized_cols = {col: normalize(col) for col in columns}

    for target in targets:
        for col, ncol in normalized_cols.items():
            if target in ncol:
                return col

    return None


def parse_datetime_column(df: pd.DataFrame, dt_col: Optional[str]) -> Optional[pd.Series]:
    if not dt_col:
        return None

    parsed = pd.to_datetime(df[dt_col], errors="coerce", dayfirst=True)
    if parsed.notna().sum() == 0:
        return None
    return parsed


def infer_failed_mask(df: pd.DataFrame, status_col: Optional[str]) -> pd.Series:
    if not status_col:
        return pd.Series([False] * len(df), index=df.index)

    status_text = df[status_col].astype(str).str.lower()
    pattern = "|".join([re.escape(item) for item in FAIL_PATTERNS])
    return status_text.str.contains(pattern, na=False)


def top_counts(df: pd.DataFrame, column: str, top_n: int = 10) -> pd.DataFrame:
    return (
        df[column]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .head(top_n)
        .reset_index()
        .rename(columns={"index": column, column: "count"})
    )


st.title("📞 Call Failure Insights Dashboard")
st.caption("Upload your Excel file and get automatic insights for failed calls.")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

if not uploaded_file:
    st.info("Upload a call report to start analyzing failures, trends, and root causes.")
    st.stop()

# Read file
try:
    excel_data = pd.read_excel(uploaded_file, sheet_name=None)
except Exception as exc:
    st.error(f"Could not read the Excel file: {exc}")
    st.stop()

sheet_names = list(excel_data.keys())
selected_sheet = st.selectbox("Sheet", options=sheet_names, index=0)
df = excel_data[selected_sheet].copy()

if df.empty:
    st.warning("The selected sheet is empty.")
    st.stop()

# Drop fully empty columns and rows
initial_shape = df.shape
df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

st.write(f"Rows/columns before cleanup: **{initial_shape[0]} / {initial_shape[1]}**")
st.write(f"Rows/columns after cleanup: **{df.shape[0]} / {df.shape[1]}**")

all_columns = df.columns.tolist()
country_col = detect_column(all_columns, KEYWORD_MAP["country"])
reason_col = detect_column(all_columns, KEYWORD_MAP["reason"])
status_col = detect_column(all_columns, KEYWORD_MAP["status"])
datetime_col = detect_column(all_columns, KEYWORD_MAP["datetime"])

with st.sidebar:
    st.header("Column mapping")
    status_col = st.selectbox("Status / outcome column", [None] + all_columns, index=([None] + all_columns).index(status_col) if status_col in all_columns else 0)
    country_col = st.selectbox("Country column", [None] + all_columns, index=([None] + all_columns).index(country_col) if country_col in all_columns else 0)
    reason_col = st.selectbox("Failure reason column", [None] + all_columns, index=([None] + all_columns).index(reason_col) if reason_col in all_columns else 0)
    datetime_col = st.selectbox("Date/time column", [None] + all_columns, index=([None] + all_columns).index(datetime_col) if datetime_col in all_columns else 0)

parsed_dt = parse_datetime_column(df, datetime_col)
if parsed_dt is not None:
    df["_parsed_datetime"] = parsed_dt
    df["_date"] = parsed_dt.dt.date
    df["_hour"] = parsed_dt.dt.hour

failed_mask = infer_failed_mask(df, status_col)
failed_df = df[failed_mask].copy()

if status_col:
    st.caption(f"Detected failure rows using patterns in **{status_col}**: {', '.join(FAIL_PATTERNS)}")
else:
    st.warning("No status column selected. Failure insights are disabled until you map a status column.")

# KPIs
total_calls = len(df)
failed_calls = len(failed_df)
failed_rate = (failed_calls / total_calls * 100) if total_calls else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Calls", f"{total_calls:,}")
col2.metric("Failed Calls", f"{failed_calls:,}")
col3.metric("Failure Rate", f"{failed_rate:.2f}%")
col4.metric("Unique Countries", f"{df[country_col].nunique() if country_col else 0:,}")

st.subheader("Core failure insights")

if failed_calls == 0:
    st.success("No failed calls were detected with current settings.")
else:
    chart_col1, chart_col2 = st.columns(2)

    if country_col:
        top_countries = top_counts(failed_df, country_col)
        chart_col1.markdown("**Top countries with failed calls**")
        chart_col1.bar_chart(top_countries.set_index(country_col)["count"])
    else:
        chart_col1.info("Map a country column to see country-level failure analysis.")

    if reason_col:
        top_reasons = top_counts(failed_df, reason_col)
        chart_col2.markdown("**Top failure reasons**")
        chart_col2.bar_chart(top_reasons.set_index(reason_col)["count"])
    else:
        chart_col2.info("Map a reason column to see main failure reasons.")

    trend_col1, trend_col2 = st.columns(2)

    if "_date" in failed_df:
        by_date = failed_df.groupby("_date").size().reset_index(name="failed_calls")
        trend_col1.markdown("**Failed calls by date**")
        trend_col1.line_chart(by_date.set_index("_date")["failed_calls"])
    else:
        trend_col1.info("Map a date/time column to view failure trend by date.")

    if "_hour" in failed_df:
        by_hour = failed_df.groupby("_hour").size().reset_index(name="failed_calls")
        trend_col2.markdown("**Failed calls by hour**")
        trend_col2.bar_chart(by_hour.set_index("_hour")["failed_calls"])
    else:
        trend_col2.info("Map a date/time column to view failure concentration by hour.")

    st.subheader("Operational diagnostics")
    diag_col1, diag_col2 = st.columns(2)

    if country_col and reason_col:
        pivot = (
            failed_df.groupby([country_col, reason_col])
            .size()
            .reset_index(name="failed_calls")
            .sort_values("failed_calls", ascending=False)
            .head(20)
        )
        diag_col1.markdown("**Failure reason mix by country (top combinations)**")
        diag_col1.dataframe(pivot, use_container_width=True)
    else:
        diag_col1.info("Map both country and reason columns to see reason mix by country.")

    failed_preview = failed_df.head(100)
    diag_col2.dataframe(failed_preview, use_container_width=True)

st.subheader("Data quality summary")
missing_pct = (df.isna().mean() * 100).sort_values(ascending=False).reset_index()
missing_pct.columns = ["column", "missing_percent"]
st.dataframe(missing_pct, use_container_width=True)

st.download_button(
    "Download cleaned dataset as CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="cleaned_calls.csv",
    mime="text/csv",
)
