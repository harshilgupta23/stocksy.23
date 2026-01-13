# ==========================================================
# STOCKSY - Streamlit Web App (LSTM Stock Predictor)
# ==========================================================

import io
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import timedelta

plt.rcParams["figure.figsize"] = (12,5)

SEQ_LEN = 50

# ---------------------------------------------------------
# PAGE CONFIG
# Sets website title
# Makes the page wide
# Adds an icon
# ---------------------------------------------------------
st.set_page_config(
    page_title="STOCKSY - AI Stock Predictor",
    layout="wide",
    page_icon="📈"
)

st.title("📈 STOCKSY — AI Stock Predictor")
st.caption("LSTM-based global stock prediction using Yahoo Finance")

# ---------------------------------------------------------
# 1️⃣ GLOBAL TICKER DATABASE
# ---------------------------------------------------------

@st.cache_data
def download_nasdaq_list():
    url = "https://nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
    r = requests.get(url)
    txt = r.content.decode('utf-8', errors='ignore')
    df = pd.read_csv(io.StringIO(txt), sep='|')
    df = df.rename(columns={'Symbol':'symbol','Security Name':'name'})
    return df[['symbol','name']].dropna()

@st.cache_data
def try_github_csv(url):
    r = requests.get(url)
    df = pd.read_csv(io.StringIO(r.text))
    sc, nc = None, None
    for col in df.columns:
        if col.lower() in ['symbol','ticker','code']:
            sc = col
        if 'name' in col.lower():
            nc = col
    if sc is None:
        return pd.DataFrame(columns=['symbol','name'])
    if nc is None:
        df['name'] = df[sc]
        return df[[sc,'name']].rename(columns={sc:'symbol'})
    return df[[sc,nc]].rename(columns={sc:'symbol', nc:'name'})

# NASDAQ official website

# GitHub stock lists

# Combines them into one big table
@st.cache_data
def build_global_db():
    sources = [download_nasdaq_list()]
    urls = [
        "https://raw.githubusercontent.com/abbadata/stock-tickers/master/data/nasdaq.csv",
        "https://raw.githubusercontent.com/abbadata/stock-tickers/master/data/nyse.csv",
        "https://raw.githubusercontent.com/GunnarPDX/csv_tickers/master/nasdaq.csv",
    ]
    for u in urls:
        sources.append(try_github_csv(u))

    df = pd.concat(sources, ignore_index=True)
    # FIX: Remove NaN values
    df = df.dropna(subset=['symbol', 'name'])


    # df['symbol'] = df['symbol'].str.upper().str.strip()
    # df['name'] = df['name'].str.strip()

    df['symbol'] = df['symbol'].astype(str).str.upper().str.strip()
    df['name'] = df['name'].astype(str).str.strip()
    return df.drop_duplicates('symbol').sort_values('symbol')

global_db = build_global_db()

# ---------------------------------------------------------
# 2️⃣ STOCK SELECTION UI
# ---------------------------------------------------------

st.subheader("🔍 Select Stock")

# 1️⃣ User input
search_text = st.text_input(
    "Search company or enter ticker",
    placeholder="Type AMZN, AAPL, MSFT..."
)

# 2️⃣ Clean & safe database
safe_db = global_db.dropna(subset=["symbol", "name"]).copy()
safe_db["symbol"] = safe_db["symbol"].astype(str)
safe_db["name"] = safe_db["name"].astype(str)

# 3️⃣ Filter results based on input
if search_text:
    filtered = safe_db[
        safe_db["symbol"].str.contains(search_text.upper(), na=False) |
        safe_db["name"].str.lower().str.contains(search_text.lower(), na=False)
    ]
else:
    filtered = safe_db

filtered = filtered.head(50)

# 4️⃣ Dropdown options
options = [f"{row.symbol} — {row.name}" for _, row in filtered.iterrows()]

# 5️⃣ Auto-select correct ticker
default_index = 0
if search_text:
    for i, opt in enumerate(options):
        if opt.startswith(search_text.upper() + " —"):
            default_index = i
            break

ticker_option = st.selectbox(
    "Select Ticker",
    options,
    index=default_index
)

# 6️⃣ Final ticker (SAFE)
ticker = ticker_option.split(" — ")[0]


# ---------------------------------------------------------
# 3️⃣ INVESTMENT DETAILS
# ---------------------------------------------------------

st.subheader("💰 Investment Details")

col1, col2, col3 = st.columns(3)

with col1:
    action = st.selectbox("Action", ["Buy", "Sell"])

with col2:
    investment = st.number_input("Investment Amount (₹)", min_value=1000.0, step=500.0)

with col3:
    holding_days = st.slider("Holding Duration (days)", 1, 90, 10)

# ---------------------------------------------------------
# 4️⃣ DOWNLOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, period="3y", interval="1d", progress=False)
    return df[['Close']].dropna()

if st.button("🚀 Predict Stock"):
    data = load_data(ticker)

    if len(data) < SEQ_LEN + 1:
        st.error("Not enough historical data for LSTM prediction.")
        st.stop()

    # ---------------------------------------------------------
    # 5️⃣ PREPARE DATA
    # ---------------------------------------------------------

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[['Close']])

    def make_sequences(arr, window):
        X, y = [], []
        for i in range(window, len(arr)):
            X.append(arr[i-window:i, 0])
            y.append(arr[i, 0])
        return np.array(X).reshape(-1, window, 1), np.array(y)

    X, y = make_sequences(scaled, SEQ_LEN)
    split = int(len(X)*0.8)
    X_train, y_train = X[:split], y[:split]

    # ---------------------------------------------------------
    # 6️⃣ LSTM MODEL
    # ---------------------------------------------------------

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN,1)),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    with st.spinner("Training LSTM model..."):
        model.fit(X_train, y_train, epochs=8, batch_size=32, verbose=0)

    # ---------------------------------------------------------
    # 7️⃣ PREDICTION
    # ---------------------------------------------------------

    last_seq = scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
    scaled_pred = model.predict(last_seq)
    pred_price = scaler.inverse_transform(scaled_pred)[0][0]

    # last_close = data['Close'].iloc[-1]
    last_close = float(data['Close'].iloc[-1])
    # change_pct = (pred_price - last_close)/last_close*100
    change_pct = ((pred_price - last_close)/last_close) * 100
    # decision = "BUY" if change_pct > 1 else ("SELL" if change_pct < -0.5 else "HOLD")
    if change_pct > 1:
        decision = "BUY"
    elif change_pct < -0.5:
        decision = "SELL"
    else:
        decision = "HOLD"
    expected_return = investment * change_pct / 100

    # ---------------------------------------------------------
    # 8️⃣ RESULTS
    # ---------------------------------------------------------

    st.subheader("📊 Prediction Result")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Close", f"₹{last_close:.2f}")
    c2.metric("Predicted Price", f"₹{pred_price:.2f}")
    c3.metric("Change %", f"{change_pct:.2f}%")
    c4.metric("Decision", decision)

    st.success(f"Expected return on ₹{investment:.2f} → ₹{expected_return:.2f}")

    # ---------------------------------------------------------
    # 9️⃣ PLOT
    # ---------------------------------------------------------

    future_prices = []
    seq = scaled[-SEQ_LEN:].copy()

    for _ in range(holding_days):
        p = model.predict(seq.reshape(1, SEQ_LEN, 1))
        seq = np.vstack([seq[1:], p])
        future_prices.append(scaler.inverse_transform(p)[0][0])

    future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, holding_days+1)]

    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close'], label="Historical")
    ax.scatter(data.index[-1] + timedelta(days=1), pred_price, color='red', label="Next-Day")
    ax.plot(future_dates, future_prices, linestyle='--', label="Future")

    ax.set_title(f"{ticker} Price Prediction")
    ax.legend()
    ax.grid()

    st.pyplot(fig)
