from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "submission" / "R1_predictions.csv"

st.set_page_config(page_title="PhantomPulse | Stock-out alerts", page_icon="📦", layout="wide")
st.title("PhantomPulse — Stock-out alert dashboard")
st.caption("Saved model rankings for review. An alert is a reason to inspect an item, not proof that stock is missing.")

if not PREDICTIONS.exists():
    st.error("The predictions file was not found. Expected: submission/R1_predictions.csv")
    st.stop()

@st.cache_data
def load_predictions(path: str, modified: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date", "store_id", "sku_id", "rank", "confidence", "reason"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df.dropna(subset=["date", "rank", "confidence", "store_id", "sku_id"])
    df["store_id"] = df["store_id"].astype(str)
    df["sku_id"] = df["sku_id"].astype(str)
    return df

try:
    data = load_predictions(str(PREDICTIONS), PREDICTIONS.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not load predictions: {exc}")
    st.stop()

if data.empty:
    st.warning("The predictions file has no valid rows.")
    st.stop()

min_date = data["date"].min().date()
max_date = data["date"].max().date()
left, right = st.columns(2)
with left:
    selected_date = st.date_input("Prediction date", value=max_date, min_value=min_date, max_value=max_date)
with right:
    stores = sorted(data.loc[data["date"].dt.date == selected_date, "store_id"].unique().tolist())
    if not stores:
        st.warning("No predictions are available for this date.")
        st.stop()
    selected_store = st.selectbox("Store", stores)

view = data[(data["date"].dt.date == selected_date) & (data["store_id"] == selected_store)].sort_values("rank").copy()

m1, m2, m3 = st.columns(3)
m1.metric("Ranked alerts", len(view))
m2.metric("Top-five alerts", int((view["rank"] <= 5).sum()))
m3.metric("Top-20 mean score", f"{view.head(20)['confidence'].mean():.3f}" if not view.empty else "—")

st.subheader("Ranked items to inspect")
st.dataframe(view[["rank", "sku_id", "confidence", "reason"]], use_container_width=True, hide_index=True)

if not view.empty:
    labels = [f"#{int(row.rank)} — {row.sku_id} (score {row.confidence:.3f})" for row in view.itertuples()]
    chosen = st.selectbox("View an alert explanation", labels)
    row = view.iloc[labels.index(chosen)]
    st.markdown(f"### SKU: `{row['sku_id']}`")
    st.write(row["reason"])
    st.metric("Ranking score", f"{row['confidence']:.3f}")

st.download_button(
    "Download these store-day alerts as CSV",
    data=view.to_csv(index=False).encode("utf-8"),
    file_name=f"phantompulse_{selected_store}_{selected_date}.csv",
    mime="text/csv",
)

with st.expander("About these results"):
    st.write(
        f"This dashboard displays saved predictions from `{PREDICTIONS.relative_to(ROOT).as_posix()}`. "
        "It does not train or run the model on new data. The confidence value is a ranking score, "
        "not a calibrated probability. Verify alerts in the store before taking action."
    )
    st.write(f"Rows loaded: {len(data):,} · Dates: {min_date} to {max_date} · Stores: {data['store_id'].nunique():,}")
