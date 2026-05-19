
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

@st.cache_data
def load_data():
    df = pd.read_csv("psx_phase3_labeled.csv")
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    return df

@st.cache_resource
def load_model():
    return joblib.load("psx_model.pkl")

try:
    df    = load_data()
    model = load_model()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

st.sidebar.title("📈 PSX Stock Analyzer")
st.sidebar.markdown("---")
company = st.sidebar.selectbox("Select Company", sorted(df["Company"].unique()))
st.sidebar.markdown("---")
st.sidebar.info("This app analyzes PSX stocks using Machine Learning and Technical Indicators.")

cdf = df[df["Company"] == company].sort_values("Date")

st.title(f"📊 {company} Stock Analysis")
st.markdown("---")

latest       = cdf.iloc[-1]
prev         = cdf.iloc[-2]
price_change = latest["Close"] - prev["Close"]
pct_change   = (price_change / prev["Close"]) * 100

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current Price",  f"PKR {latest['Close']:.2f}", f"{pct_change:.2f}%")
col2.metric("RSI",            f"{latest['RSI']:.1f}")
col3.metric("MACD",           f"{latest['MACD']:.2f}")
col4.metric("Daily Return",   f"{latest['Daily_Return']:.2f}%")
col5.metric("Signal",         latest["Signal"])

st.markdown("---")

# Price Chart
st.subheader("📈 Price Chart with Bollinger Bands")
fig = go.Figure()
fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["Close"],    name="Close",     line=dict(color="#00b4d8", width=2)))
fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["BB_Upper"], name="BB Upper",  line=dict(color="rgba(255,100,100,0.5)", dash="dash")))
fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["BB_Lower"], name="BB Lower",  line=dict(color="rgba(100,255,100,0.5)", dash="dash"), fill="tonexty", fillcolor="rgba(173,216,230,0.1)"))
fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["MA_20"],    name="MA 20",     line=dict(color="orange", width=1)))
fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["MA_50"],    name="MA 50",     line=dict(color="yellow", width=1)))
fig.update_layout(template="plotly_dark", height=450, xaxis_title="Date", yaxis_title="Price (PKR)")
st.plotly_chart(fig, use_container_width=True)

# RSI Chart
st.subheader("📉 RSI — Relative Strength Index")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=cdf["Date"], y=cdf["RSI"], name="RSI", line=dict(color="#f77f00", width=2)))
fig2.add_hline(y=70, line_dash="dash", line_color="red",   annotation_text="Overbought (70)")
fig2.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
fig2.update_layout(template="plotly_dark", height=300, yaxis=dict(range=[0,100]))
st.plotly_chart(fig2, use_container_width=True)

# MACD Chart
st.subheader("📊 MACD")
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=cdf["Date"], y=cdf["MACD"],        name="MACD",      line=dict(color="#4cc9f0")))
fig3.add_trace(go.Scatter(x=cdf["Date"], y=cdf["MACD_Signal"], name="Signal",    line=dict(color="#f72585")))
fig3.add_bar(             x=cdf["Date"], y=cdf["MACD_Hist"],   name="Histogram", marker_color="gray")
fig3.update_layout(template="plotly_dark", height=300)
st.plotly_chart(fig3, use_container_width=True)

# ML Recommendation
st.markdown("---")
st.subheader("🤖 ML Recommendation")

features = ["RSI","MACD","MACD_Signal","MACD_Hist",
            "BB_Upper","BB_Middle","BB_Lower",
            "MA_20","MA_50","MA_200","Daily_Return","Volume"]

latest_features = cdf[features].iloc[-1:].values
prediction      = model.predict(latest_features)[0]
probabilities   = model.predict_proba(latest_features)[0]
classes         = model.classes_

if prediction == "BUY":
    st.success(f"🟢 Recommendation: BUY — Model suggests this stock may rise")
elif prediction == "SELL":
    st.error(f"🔴 Recommendation: SELL — Model suggests this stock may fall")
else:
    st.warning(f"🟡 Recommendation: HOLD — Model suggests waiting")

prob_df = pd.DataFrame({"Signal": classes, "Probability": [f"{p*100:.1f}%" for p in probabilities]})
st.dataframe(prob_df, use_container_width=True)

# Recent Data Table
st.markdown("---")
st.subheader("📋 Recent Data")
show_cols = ["Date","Open","High","Low","Close","Volume","RSI","MACD","Daily_Return","Signal"]
st.dataframe(cdf[show_cols].tail(20).sort_values("Date", ascending=False), use_container_width=True)
