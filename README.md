# 📈 PSX Stock Analyzer

A web-based stock analysis dashboard for the **Pakistan Stock Exchange (PSX)**, built with Python and Streamlit. It combines technical analysis indicators with a trained machine learning model to generate BUY / SELL / HOLD recommendations for PSX-listed companies.

🔗 **Live App:** [psx-stock-analyzer on Streamlit Cloud](https://psx-stock-analyzer-s6hrjykneppplejyywwtvx.streamlit.app)

---

## 🚀 Features

- **Company Selector** — Browse and analyze any PSX company from the dataset
- **KPI Metrics Row** — Current price, daily return %, RSI, MACD, and rule-based signal at a glance
- **Interactive Charts** (powered by Plotly):
  - Price chart with Bollinger Bands and Moving Averages (MA 20, MA 50, MA 200)
  - Trading Volume bar chart (color-coded by daily return)
  - RSI chart with overbought/oversold threshold lines
  - MACD chart with signal line and histogram
- **ML Recommendation** — A trained scikit-learn model predicts BUY / SELL / HOLD with confidence probabilities
- **Recent Data Table** — Last 20 trading days with formatted dates and key indicators

---

## 🧠 Machine Learning Model

- **File:** `psx_model.pkl`
- **Type:** scikit-learn classifier (trained on labeled PSX historical data)
- **Features used:** RSI, MACD, MACD Signal, MACD Histogram, Bollinger Bands (Upper/Middle/Lower), MA 20, MA 50, MA 200 (if available), Daily Return, Volume
- **Output:** BUY / SELL / HOLD signal with class probabilities

---

## 📊 Technical Indicators Explained

| Indicator | Description |
|-----------|-------------|
| **RSI** | Relative Strength Index. >70 = Overbought, <30 = Oversold |
| **MACD** | Moving Average Convergence Divergence — trend momentum |
| **Bollinger Bands** | Price volatility bands around a 20-day moving average |
| **MA 20 / MA 50 / MA 200** | Short, medium, and long-term moving averages |
| **Daily Return** | Percentage price change from previous trading day |

---

## 🗂️ Project Structure

```
psx-stock-analyzer/
├── psx_app.py                  # Main Streamlit application
├── psx_model.pkl               # Trained ML model (scikit-learn)
├── psx_phase3_labeled.csv      # Historical PSX dataset with technical indicators
├── requirements.txt            # Python dependencies
└── README.md
```

---

## ⚙️ Installation & Local Run

```bash
# 1. Clone the repository
git clone https://github.com/shome2924-star/psx-stock-analyzer.git
cd psx-stock-analyzer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run psx_app.py
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
plotly
joblib
scikit-learn
```

---

## ⚠️ Data Note

This app uses a **historical offline dataset** (`psx_phase3_labeled.csv`). It does not fetch live or real-time PSX data. The dataset includes pre-computed technical indicators and rule-based signal labels.

---

## 👤 Author

**shome2924-star**  
GitHub: [github.com/shome2924-star](https://github.com/shome2924-star)

---

## 📄 License

This project is for educational and demonstration purposes.
