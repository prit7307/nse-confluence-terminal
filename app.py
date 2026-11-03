import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import os

st.set_page_config(
    page_title="NSE Institutional Confluence & Research Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# DATA ENDPOINTS
# ==============================================================================
GITHUB_PARQUET_URL = "https://raw.githubusercontent.com/prit7307/nse-working-scanner/main/data/nse_rolling_60d.parquet"
SECTOR_MASTER_URL = "https://raw.githubusercontent.com/prit7307/nse-working-scanner/main/nse_sector_master.csv"
FUNDAMENTAL_CSV_FILE = "nse_confluence_picks_20261030.csv"

# Replace with your published Google Sheet CSV URL from Part 1
REPORTS_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRipXwixKWlGix_0O58Vq5kp-l5W8TubMbWaSeVf9P6j-JtT3GhpajftV4OK9kcl7cl6WJkfzGY1JRO/pub?output=csv"


def clean_header(name: str) -> str:
    """Normalizes Screener multi-space column names."""
    c = re.sub(r"\s+", " ", str(name)).strip().upper()
    return c.replace(" RS.CR.", " (CR)").replace(" RS.", "")


@st.cache_data(ttl=60)
def fetch_live_reports_catalog(url: str):
    """Fetches the auto-updated Google Sheet catalog."""
    try:
        df = pd.read_csv(url)
        df.columns = [str(c).strip().upper() for c in df.columns]
        return df
    except Exception as e:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_unified_dataset():
    # 1. Ingest Technical Dataset
    try:
        raw_tech = pd.read_parquet(GITHUB_PARQUET_URL)
        raw_tech["DATE"] = pd.to_datetime(raw_tech["DATE"])
        latest_date = raw_tech["DATE"].max()
        tech_df = raw_tech[raw_tech["DATE"] == latest_date].copy()

        for c in ["CLOSE", "PREV_CLOSE", "HIGH", "LOW", "DELIV_QTY", "TOTTRDQTY"]:
            if c in tech_df.columns:
                tech_df[c] = pd.to_numeric(tech_df[c], errors="coerce").fillna(0.0)

        # DSI Calculation
        rolling_deliv = raw_tech.groupby("SYMBOL")["DELIV_QTY"].transform(
            lambda s: s.rolling(50, min_periods=1).mean()
        ).replace(0, np.nan)
        tech_df["DSI"] = (tech_df["DELIV_QTY"] / rolling_deliv.loc[tech_df.index]).fillna(1.0).round(2)

        # Mansfield RS Calculation
        benchmark = raw_tech.groupby("DATE")["CLOSE"].transform("median").replace(0, np.nan)
        raw_tech["RAW_RS"] = (raw_tech["CLOSE"] / benchmark).fillna(1.0)
        raw_tech["SMA50_RS"] = raw_tech.groupby("SYMBOL")["RAW_RS"].transform(
            lambda s: s.rolling(50, min_periods=1).mean()
        ).replace(0, np.nan)
        tech_df["MANSFIELD_RS"] = (((raw_tech.loc[tech_df.index, "RAW_RS"] / raw_tech.loc[tech_df.index, "SMA50_RS"]) - 1.0) * 100.0).fillna(0.0).round(2)

        # NR7 Contraction
        raw_tech["RANGE"] = (raw_tech["HIGH"] - raw_tech["LOW"]).round(2)
        min_7d = raw_tech.groupby("SYMBOL")["RANGE"].transform(lambda s: s.rolling(7, min_periods=1).min())
        tech_df["NR7"] = (raw_tech.loc[tech_df.index, "RANGE"] <= min_7d.loc[tech_df.index]) & (raw_tech.loc[tech_df.index, "RANGE"] > 0)

        # Price Change %
        if "PCT_CHANGE" not in tech_df.columns or tech_df["PCT_CHANGE"].abs().sum() == 0:
            tech_df["PCT_CHANGE"] = np.where(
                tech_df["PREV_CLOSE"] > 0,
                ((tech_df["CLOSE"] - tech_df["PREV_CLOSE"]) / tech_df["PREV_CLOSE"] * 100.0).round(2),
                0.0
            )

        # Setup Labels
        tech_df["Accumulation_Alert"] = np.where(
            (tech_df["DSI"] >= 1.5) & (tech_df["PCT_CHANGE"] > 0) & (tech_df["NR7"]),
            "STRONG ACCUMULATION",
            np.where((tech_df["DSI"] >= 1.5) & (tech_df["PCT_CHANGE"] > 0), "DELIVERY SURGE",
            np.where(tech_df["NR7"], "NR7 SQUEEZE", "NEUTRAL"))
        )
        tech_df["SYMBOL"] = tech_df["SYMBOL"].astype(str).str.strip().str.upper()

    except Exception as e:
        st.error(f"Error fetching technical dataset: {e}")
        return pd.DataFrame(), None

    # 2. Ingest Sector Mapping
    try:
        sec_df = pd.read_csv(SECTOR_MASTER_URL)
        sec_df["SYMBOL"] = sec_df["SYMBOL"].astype(str).str.strip().str.upper()
        tech_df = tech_df.merge(sec_df[["SYMBOL", "SECTOR"]], on="SYMBOL", how="left")
        tech_df["SECTOR"] = tech_df["SECTOR"].fillna("Broad Market / Small & Mid Cap")
    except Exception:
        tech_df["SECTOR"] = "Broad Market / Small & Mid Cap"

    # 3. Merge Confluence Fundamental Metrics
    if os.path.exists(FUNDAMENTAL_CSV_FILE):
        fund_df = pd.read_csv(FUNDAMENTAL_CSV_FILE)
        fund_df.columns = [clean_header(c) for c in fund_df.columns]
        fund_df["SYMBOL"] = fund_df["SYMBOL"].astype(str).str.strip().str.upper()
        overlap = [c for c in ["CLOSE", "PCT_CHANGE", "DELIV_QTY", "DSI", "MANSFIELD_RS"] if c in fund_df.columns]
        fund_df = fund_df.drop(columns=overlap)
        unified_df = tech_df.merge(fund_df, on="SYMBOL", how="left")
    else:
        unified_df = tech_df

    return unified_df, latest_date


df, active_session = load_unified_dataset()
reports_df = fetch_live_reports_catalog(REPORTS_SHEET_CSV_URL)

if df.empty:
    st.error("Dashboard failed to initialize.")
    st.stop()

# ------------------------------------------------------------------------------
# 4. REPORT MATCHING ENGINE
# ------------------------------------------------------------------------------
def match_research_report(row, rep_df):
    if rep_df.empty:
        return "None"
    
    sym = str(row.get("SYMBOL", "")).upper()
    comp = str(row.get("COMPANY", "")).upper()
    comp_clean = re.sub(r"[^\w\s]", "", comp).split()[0] if comp else ""

    # Match in title or symbol column
    for _, rep in rep_df.iterrows():
        title = str(rep.get("REPORT_TITLE", "")).upper()
        keyword = str(rep.get("SYMBOL_OR_NAME", "")).upper()
        link = str(rep.get("DRIVE_VIEW_LINK", ""))

        if (len(sym) >= 3 and (sym in title or sym in keyword)) or \
           (len(comp_clean) >= 4 and (comp_clean in title or comp_clean in keyword)):
            return link

    return "None"


df["REPORT_LINK"] = df.apply(lambda r: match_research_report(r, reports_df), axis=1)
df["HAS_REPORT"] = df["REPORT_LINK"] != "None"

# ------------------------------------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------------------------------------
st.sidebar.title("⚡ Unified Engine Filters")
st.sidebar.caption(f"Session: **{active_session.strftime('%d %b %Y')}**")

report_only = st.sidebar.checkbox("⭐ Show Only Stocks with Research Reports", value=False)
if report_only:
    df = df[df["HAS_REPORT"]]

search_input = st.sidebar.text_input("🔍 Search Symbol or Company:").strip().upper()
if search_input:
    s_mask = df["SYMBOL"].str.contains(search_input, na=False)
    c_mask = df["COMPANY"].astype(str).str.upper().str.contains(search_input, na=False) if "COMPANY" in df.columns else False
    df = df[s_mask | c_mask]

if "CONFLUENCE_SCORE" in df.columns:
    min_score = st.sidebar.slider("Minimum Confluence Score (0–5):", 0, 5, 0, 1)
    df = df[df["CONFLUENCE_SCORE"].fillna(0) >= min_score]

min_dsi = st.sidebar.slider("Min Delivery Surge Index (DSI):", 0.5, 5.0, 1.0, 0.1)
df = df[df["DSI"] >= min_dsi]

roce_col = [c for c in df.columns if "ROCE" in c]
if roce_col:
    min_roce = st.sidebar.slider("Min ROCE %:", 0.0, 50.0, 0.0, 1.0)
    if min_roce > 0:
        df = df[pd.to_numeric(df[roce_col[0]], errors="coerce").fillna(0) >= min_roce]

# ------------------------------------------------------------------------------
# DASHBOARD DISPLAY
# ------------------------------------------------------------------------------
st.title("⚡ Institutional Confluence & Research Terminal")
st.markdown(f"Screening **{len(df):,}** equities with linked fundamental scores & live Google Drive research coverage.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Screened Equities", f"{len(df):,} Stocks")
k2.metric("Accumulation Setups", len(df[df["Accumulation_Alert"].isin(["STRONG ACCUMULATION", "DELIVERY SURGE"])]))
k3.metric("Research Reports Matched", len(df[df["HAS_REPORT"]]))
k4.metric("Avg DSI", f"{df['DSI'].mean():.2f}x")

st.divider()

display_cols = [
    "REPORT_LINK", "SYMBOL", "COMPANY", "SECTOR", "CLOSE", "PCT_CHANGE",
    "DELIV_QTY", "DSI", "MANSFIELD_RS", "CONFLUENCE_SCORE", "FUNDAMENTAL_SIGNALS",
    "Accumulation_Alert", "P/E", "ROCE %", "QTR PROFIT VAR %", "MAR CAP (CR)"
]
visible_cols = [c for c in display_cols if c in df.columns]

# Format Google Drive Report links as clickable interactive links
st.dataframe(
    df[visible_cols].sort_values(by=["HAS_REPORT", "DSI"], ascending=[False, False]),
    column_config={
        "REPORT_LINK": st.column_config.LinkColumn(
            "Research Report",
            help="Direct link to view PDF report in Google Drive",
            validate=r"^https://.*",
            max_chars=100,
            display_text="📄 View PDF"
        ),
        "CLOSE": st.column_config.NumberColumn("Close (₹)", format="₹%.2f"),
        "PCT_CHANGE": st.column_config.NumberColumn("Change (%)", format="%+.2f%%"),
        "DSI": st.column_config.NumberColumn("DSI", format="%.2fx"),
        "MANSFIELD_RS": st.column_config.NumberColumn("Mansfield RS", format="%+.2f"),
        "DELIV_QTY": st.column_config.NumberColumn("Delivery Vol", format="%d")
    },
    use_container_width=True,
    height=650
)
