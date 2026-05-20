import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

st.set_page_config(
    page_title="PSX Stock Analyzer",
    page_icon="📈",
    layout="wide"
)

# ── Data & Model Loading ───────────────────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv("psx_phase3_labeled.csv")
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    return df

@st.cache_resource
def load_model():
    return joblib.load("psx_model.pkl")

try:
    df = load_data()
    model = load_model()
except Exception as e:
    st.error(f"❌ Error loading data or model: {e}")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.title("📈 PSX Stock Analyzer")
st.sidebar.markdown("---")

company = st.sidebar.selectbox("Select Company", sorted(df["Company"].unique()))

st.sidebar.markdown("---")

# Dataset info
total_companies = df["Company"].nunique()
date_min = df["Date"].min().strftime("%d %b %Y")
date_max = df["Date"].max().strftime("%d %b %Y")

st.sidebar.markdown("### 📊 Dataset Info")
st.sidebar.markdown(f"- **Companies:** {total_companies}")
st.sidebar.markdown(f"- **From:** {date_min}")
st.sidebar.markdown(f"- **To:** {date_max}")
st.sidebar.markdown(f"- **Source:** Historical PSX (Offline)")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info(
    "This app analyzes Pakistan Stock Exchange (PSX) stocks using "
    "Machine Learning and Technical Indicators including RSI, MACD, "
    "Bollinger Bands, and Moving Averages."
)
st.sidebar.markdown("---")
st.sidebar.caption("🛠️ Built with Streamlit · Python · scikit-learn")
st.sidebar.caption("👤 Developed by: shome2924-star")

# ── Filter company data ────────────────────────────────────────────────────────

cdf = df[df["Company"] == company].sort_values("Date").reset_index(drop=True)

if len(cdf) < 2:
    st.warning("Not enough data for this company.")
    st.stop()

# ── Header ─────────────────────────────────────────────────────────────────────

st.title(f"📊 {company} — Stock Analysis")
st.caption("⚠️ Data source: Historical PSX dataset (offline). Not real-time.")
st.markdown("---")

# ── KPI Metrics ────────────────────────────────────────────────────────────────

latest = cdf.iloc[-1]
prev   = cdf.iloc[-2]

price_change = latest["Close"] - prev["Close"]
pct_change   = (price_change / prev["Close"]) * 100

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Current Price (PKR)", f"{latest['Close']:.2f}",   f"{pct_change:+.2f}%")
col2.metric("RSI",                 f"{latest['RSI']:.1f}",     help="Relative Strength Index. >70 = Overbought, <30 = Oversold")
col3.metric("MACD",                f"{latest['MACD']:.3f}",    help="Moving Average Convergence Divergence")
col4.metric("Daily Return",        f"{latest['Daily_Return']:.2f}%")
col5.metric("Rule-based Signal",   latest["Signal"],           help="Signal derived from technical rules in the dataset")

st.markdown("---")

# ── Price Chart with Bollinger Bands ──────────────────────────────────────────

st.subheader("📈 Price Chart with Bollinger Bands & Moving Averages")

fig = go.Figure()
fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["Close"],    name="Close Price", line=dict(color="#00b4d8", width=2)))
fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["BB_Upper"], name="BB Upper",    line=dict(color="rgba(255,100,100,0.6)", dash="dash")))
fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["BB_Lower"], name="BB Lower",    line=dict(color="rgba(100,255,100,0.6)", dash="dash"),
                         fill="tonexty", fillcolor="rgba(173,216,230,0.08)"))
fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["MA_20"],   name="MA 20",       line=dict(color="orange", width=1.5)))
fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["MA_50"],   name="MA 50",       line=dict(color="yellow", width=1.5)))

if "MA_200" in cdf.columns and cdf["MA_200"].notna().any():
    fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["MA_200"], name="MA 200", line=dict(color="violet", width=1, dash="dot")))

fig.update_layout(
    template="plotly_dark", height=450,
    xaxis_title="Date", yaxis_title="Price (PKR)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# ── Volume Chart ──────────────────────────────────────────────────────────────

st.subheader("📦 Trading Volume")

fig_vol = go.Figure()
fig_vol.add_trace(go.Bar(
    x=cdf["Date"], y=cdf["Volume"],
    name="Volume",
    marker_color=["#ef233c" if r < 0 else "#2ec4b6"
                  for r in cdf["Daily_Return"].fillna(0)]
))
fig_vol.update_layout(
    template="plotly_dark", height=280,
    xaxis_title="Date", yaxis_title="Volume",
    showlegend=False
)
st.plotly_chart(fig_vol, use_container_width=True)

# ── RSI Chart ─────────────────────────────────────────────────────────────────

st.subheader("📉 RSI — Relative Strength Index")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=cdf["Date"], y=cdf["RSI"], name="RSI", line=dict(color="#f77f00", width=2)))
fig2.add_hline(y=70, line_dash="dash", line_color="red",   annotation_text="Overbought (70)")
fig2.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
fig2.update_layout(template="plotly_dark", height=300, yaxis=dict(range=[0, 100]))
st.plotly_chart(fig2, use_container_width=True)

# ── MACD Chart ────────────────────────────────────────────────────────────────

st.subheader("📊 MACD — Moving Average Convergence Divergence")

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=cdf["Date"], y=cdf["MACD"],        name="MACD",      line=dict(color="#4cc9f0")))
fig3.add_trace(go.Scatter(x=cdf["Date"], y=cdf["MACD_Signal"], name="Signal Line", line=dict(color="#f72585")))
fig3.add_trace(go.Bar(    x=cdf["Date"], y=cdf["MACD_Hist"],   name="Histogram",  marker_color="gray"))
fig3.update_layout(template="plotly_dark", height=300,
                   legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig3, use_container_width=True)

# ── ML Recommendation ─────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("🤖 ML Model Recommendation")
st.caption("Prediction made by a trained machine learning model (separate from the rule-based signal above).")

features = [
    "RSI", "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Upper", "BB_Middle", "BB_Lower",
    "MA_20", "MA_50", "Daily_Return", "Volume"
]

# Only include MA_200 if it exists and has data
if "MA_200" in cdf.columns and cdf["MA_200"].notna().any():
    features.append("MA_200")

try:
    latest_features = cdf[features].iloc[-1:].values

    if np.isnan(latest_features).any():
        st.warning("⚠️ Latest row has missing values. Using last valid row for prediction.")
        valid_row = cdf[features].dropna().iloc[-1:]
        latest_features = valid_row.values

    prediction   = model.predict(latest_features)[0]
    probabilities = model.predict_proba(latest_features)[0]
    classes      = model.classes_

    if prediction == "BUY":
        st.success("🟢 **Recommendation: BUY** — Model suggests this stock may rise")
    elif prediction == "SELL":
        st.error("🔴 **Recommendation: SELL** — Model suggests this stock may fall")
    else:
        st.warning("🟡 **Recommendation: HOLD** — Model suggests waiting for a clearer signal")

    prob_df = pd.DataFrame({
        "Signal":      classes,
        "Probability": [f"{p * 100:.1f}%" for p in probabilities]
    })
    st.dataframe(prob_df, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"❌ ML prediction failed: {e}")

# ── Recent Data Table ─────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("📋 Recent Data (Last 20 Trading Days)")

show_cols = ["Date", "Open", "High", "Low", "Close", "Volume",
             "RSI", "MACD", "Daily_Return", "Signal"]

display_df = cdf[show_cols].tail(20).sort_values("Date", ascending=False).copy()
display_df["Date"] = display_df["Date"].dt.strftime("%d-%b-%Y")

st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:0.85em;'>"
    "📈 PSX Stock Analyzer · Built with Python & Streamlit · "
    "Data: Historical PSX Dataset (Offline) · "
    "Model: scikit-learn Classifier"
    "</div>",
    unsafe_allow_html=True
)
