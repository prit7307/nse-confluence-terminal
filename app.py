import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import os

st.set_page_config(
    page_title="NSE Unified Confluence & Research Scanner",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# DATA ENDPOINTS & URLS
# ==============================================================================
GITHUB_PARQUET_URL = "https://raw.githubusercontent.com/prit7307/nse-working-scanner/main/data/nse_rolling_60d.parquet"
SECTOR_MASTER_URL = "https://raw.githubusercontent.com/prit7307/nse-working-scanner/main/nse_sector_master.csv"
FUNDAMENTAL_CSV_FILE = "nse_confluence_picks_20261030.csv"

# Live published Google Sheet CSV URL
REPORTS_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRipXwixKWlGix_0O58Vq5kp-l5W8TubMbWaSeVf9P6j-JtT3GhpajftV4OK9kcl7cl6WJkfzGY1JRO/pub?output=csv"


def clean_header(name: str) -> str:
    """Normalizes Screener multi-space column names."""
    c = re.sub(r"\s+", " ", str(name)).strip().upper()
    return c.replace(" RS.CR.", " (CR)").replace(" RS.", "")


# ==============================================================================
# 1. OPTIMIZED GOOGLE SHEET REPORT CATALOG LOADER
# ==============================================================================
@st.cache_data(ttl=600)  # Caches for 10 minutes to eliminate network latency
def get_report_link_map(sheet_url: str):
    """
    Downloads the Google Sheet once and builds a fast keyword-to-link catalog.
    """
    try:
        rep_df = pd.read_csv(sheet_url)
        rep_df.columns = [str(c).strip().upper() for c in rep_df.columns]
    except Exception:
        return []

    link_col = next((c for c in ["DRIVE_VIEW_LINK", "VIEW_URL", "LINK"] if c in rep_df.columns), None)
    if not link_col:
        return []

    report_items = []
    for _, row in rep_df.iterrows():
        title = str(row.get("REPORT_TITLE", "")).upper()
        sym_name = str(row.get("SYMBOL_OR_NAME", "")).upper()
        link = str(row.get(link_col, "")).strip()
        combined_text = f"{title} {sym_name}"
        if link and link.startswith("http"):
            report_items.append((combined_text, link))

    return report_items


# ==============================================================================
# 2. CORE EXCHANGE TECHNICAL & FUNDAMENTAL LOADER
# ==============================================================================
@st.cache_data(ttl=300)
def load_base_datasets():
    # A. Ingest Live Parquet from GitHub
    try:
        raw_tech = pd.read_parquet(GITHUB_PARQUET_URL)
        raw_tech["DATE"] = pd.to_datetime(raw_tech["DATE"])
        latest_date = raw_tech["DATE"].max()
        tech_df = raw_tech[raw_tech["DATE"] == latest_date].copy()

        for c in ["CLOSE", "PREV_CLOSE", "HIGH", "LOW", "DELIV_QTY", "TOTTRDQTY"]:
            if c in tech_df.columns:
                tech_df[c] = pd.to_numeric(tech_df[c], errors="coerce").fillna(0.0)

        # 1. Delivery Surge Index (DSI)
        rolling_deliv = raw_tech.groupby("SYMBOL")["DELIV_QTY"].transform(
            lambda s: s.rolling(50, min_periods=1).mean()
        ).replace(0, np.nan)
        tech_df["DSI"] = (tech_df["DELIV_QTY"] / rolling_deliv.loc[tech_df.index]).fillna(1.0).round(2)

        # 2. Mansfield Relative Strength vs Universe Median
        benchmark = raw_tech.groupby("DATE")["CLOSE"].transform("median").replace(0, np.nan)
        raw_tech["RAW_RS"] = (raw_tech["CLOSE"] / benchmark).fillna(1.0)
        raw_tech["SMA50_RS"] = raw_tech.groupby("SYMBOL")["RAW_RS"].transform(
            lambda s: s.rolling(50, min_periods=1).mean()
        ).replace(0, np.nan)
        tech_df["MANSFIELD_RS"] = (((raw_tech.loc[tech_df.index, "RAW_RS"] / raw_tech.loc[tech_df.index, "SMA50_RS"]) - 1.0) * 100.0).fillna(0.0).round(2)

        # 3. NR7 Volatility Contraction
        raw_tech["RANGE"] = (raw_tech["HIGH"] - raw_tech["LOW"]).round(2)
        min_7d = raw_tech.groupby("SYMBOL")["RANGE"].transform(lambda s: s.rolling(7, min_periods=1).min())
        tech_df["NR7"] = (raw_tech.loc[tech_df.index, "RANGE"] <= min_7d.loc[tech_df.index]) & (raw_tech.loc[tech_df.index, "RANGE"] > 0)

        # 4. Standard Session % Change
        if "PCT_CHANGE" not in tech_df.columns or tech_df["PCT_CHANGE"].abs().sum() == 0:
            tech_df["PCT_CHANGE"] = np.where(
                tech_df["PREV_CLOSE"] > 0,
                ((tech_df["CLOSE"] - tech_df["PREV_CLOSE"]) / tech_df["PREV_CLOSE"] * 100.0).round(2),
                0.0
            )

        # Accumulation Alert
        tech_df["Accumulation_Alert"] = np.where(
            (tech_df["DSI"] >= 1.5) & (tech_df["PCT_CHANGE"] > 0) & (tech_df["NR7"]),
            "STRONG ACCUMULATION",
            np.where((tech_df["DSI"] >= 1.5) & (tech_df["PCT_CHANGE"] > 0), "DELIVERY SURGE",
            np.where(tech_df["NR7"], "NR7 SQUEEZE", "NEUTRAL"))
        )
        tech_df["SYMBOL"] = tech_df["SYMBOL"].astype(str).str.strip().str.upper()

    except Exception as e:
        st.error(f"Error fetching technical exchange data: {e}")
        return pd.DataFrame(), None

    # B. Ingest Sector Master
    try:
        sec_df = pd.read_csv(SECTOR_MASTER_URL)
        sec_df["SYMBOL"] = sec_df["SYMBOL"].astype(str).str.strip().str.upper()
        tech_df = tech_df.merge(sec_df[["SYMBOL", "SECTOR"]], on="SYMBOL", how="left")
        tech_df["SECTOR"] = tech_df["SECTOR"].fillna("Broad Market / Small & Mid Cap")
    except Exception:
        tech_df["SECTOR"] = "Broad Market / Small & Mid Cap"

    # C. Ingest Confluence Fundamental Metrics
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


# ==============================================================================
# 3. HIGH-SPEED VECTORIZED REPORT MATCHER
# ==============================================================================
@st.cache_data(ttl=300)
def attach_reports_fast(base_df, catalog_items):
    """Matches stocks against Drive reports in under 0.1 seconds."""
    if not catalog_items or base_df.empty:
        base_df["REPORT_LINK"] = None
        base_df["HAS_REPORT"] = False
        return base_df

    def match_item(sym, comp):
        s = str(sym).strip().upper()
        c = re.sub(r"[^\w\s]", "", str(comp).upper()).split()[0] if pd.notna(comp) and str(comp).strip() else ""

        for text, link in catalog_items:
            if len(s) >= 3 and s in text:
                return link
            if len(c) >= 4 and c in text:
                return link
        return None

    symbols = base_df["SYMBOL"].values
    companies = base_df["COMPANY"].values if "COMPANY" in base_df.columns else [""] * len(base_df)

    links = [match_item(s, c) for s, c in zip(symbols, companies)]
    base_df["REPORT_LINK"] = links
    base_df["HAS_REPORT"] = [l is not None for l in links]
    return base_df


# --- LOAD DATA ---
raw_df, active_session = load_base_datasets()
if raw_df.empty:
    st.error("Dashboard failed to initialize. Please verify GitHub Parquet access.")
    st.stop()

report_items = get_report_link_map(REPORTS_SHEET_CSV_URL)
df = attach_reports_fast(raw_df, report_items)

# ==============================================================================
# 4. SIDEBAR CONTROLS & FILTERS
# ==============================================================================
st.sidebar.title("⚡ Confluence & Report Filters")
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

if "SECTOR" in df.columns:
    sectors = sorted(df["SECTOR"].unique().tolist())
    selected_sec = st.sidebar.multiselect("Filter by Sector:", options=sectors, default=sectors)
    df = df[df["SECTOR"].isin(selected_sec)]

# ==============================================================================
# 5. DASHBOARD LAYOUT & METRICS
# ==============================================================================
st.title("⚡ Institutional Confluence & Research Terminal")
st.markdown(f"Screening **{len(df):,}** equities with technical alerts, Screener metrics, and live Google Drive research coverage.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Screened Equities", f"{len(df):,} Stocks")
k2.metric("Accumulation Setups", len(df[df["Accumulation_Alert"].isin(["STRONG ACCUMULATION", "DELIVERY SURGE"])]))
k3.metric("Research Reports Matched", len(df[df["HAS_REPORT"]]))
k4.metric("Avg DSI", f"{df['DSI'].mean():.2f}x")

st.divider()

# Organize columns cleanly
display_cols = [
    "REPORT_LINK", "SYMBOL", "COMPANY", "SECTOR", "CLOSE", "PCT_CHANGE",
    "DELIV_QTY", "DSI", "MANSFIELD_RS", "CONFLUENCE_SCORE", "FUNDAMENTAL_SIGNALS",
    "Accumulation_Alert", "P/E", "ROCE %", "QTR PROFIT VAR %", "MAR CAP (CR)"
]
visible_cols = [c for c in display_cols if c in df.columns]

# Sort on the full dataframe first so HAS_REPORT is always accessible
sorted_df = df.sort_values(by=["HAS_REPORT", "DSI"], ascending=[False, False])

st.dataframe(
    sorted_df[visible_cols],
    column_config={
        "REPORT_LINK": st.column_config.LinkColumn(
            "Research Report",
            help="Direct link to view PDF report in Google Drive",
            validate=r"^https://.*",
            max_chars=100,
            display_text="📄 View Report"
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
