# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from datetime import datetime, timedelta
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# === Page Config ===
st.set_page_config(page_title="ARIMA Trading Agent", layout="wide")
st.title("🤖 Agentic ARIMA Trading Dashboard")
st.markdown("Auto ARIMA forecasting with **buy/sell signals** based on prediction vs actual price.")

# === Sidebar Controls ===
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()
period = st.sidebar.selectbox("Data Period", ["1y", "2y", "6mo", "3mo", "1mo"], index=0)
forecast_days = st.sidebar.slider("Forecast Days Ahead", 1, 30, 7)
signal_threshold = st.sidebar.slider("Signal Threshold (%)", 0.1, 5.0, 1.0) / 100

# === Fetch Data ===
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_data(ticker, period):
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data.empty:
            st.error(f"No data found for {ticker}")
            return None
        data = data[['Close']].copy()
        data.dropna(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

data = get_data(ticker, period)
if data is None:
    st.stop()

# Split: Train on all but last N days for validation
train_end = len(data) - forecast_days
train_data = data.iloc[:train_end]
test_data = data.iloc[train_end:]

# === Auto ARIMA Model ===
@st.cache_resource
def fit_arima_model(train_series):
    with st.spinner("Fitting Auto ARIMA model..."):
        model = auto_arima(
            train_series,
            start_p=0, start_q=0,
            max_p=5, max_q=5,
            d=None, seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=False
        )
    return model

model = fit_arima_model(train_data['Close'])

# === Forecast ===
forecast = model.predict(n_periods=forecast_days)
forecast_index = test_data.index
forecast_series = pd.Series(forecast, index=forecast_index)

# === Generate Signals ===
signals = []
for i in range(len(test_data)):
    actual = test_data['Close'].iloc[i]
    pred = forecast[i]
    pct_diff = (actual - pred) / pred
    if pct_diff > signal_threshold:
        signals.append("SELL 🔴")
    elif pct_diff < -signal_threshold:
        signals.append("BUY 🟢")
    else:
        signals.append("HOLD ⚪")

test_data = test_data.copy()
test_data['Forecast'] = forecast
test_data['Signal'] = signals
test_data['% Error'] = ((test_data['Close'] - test_data['Forecast']) / test_data['Forecast']) * 100

# === Plotting ===
fig = go.Figure()

# Historical
fig.add_trace(go.Scatter(
    x=data.index, y=data['Close'],
    mode='lines', name='Historical Price',
    line=dict(color='gray')
))

# Forecast
fig.add_trace(go.Scatter(
    x=forecast_index, y=forecast_series,
    mode='lines+markers', name='ARIMA Forecast',
    line=dict(color='blue', dash='dot')
))

# Actual test
fig.add_trace(go.Scatter(
    x=test_data.index, y=test_data['Close'],
    mode='lines+markers', name='Actual Price',
    line=dict(color='black')
))

# Buy/Sell Markers
buy_signals = test_data[test_data['Signal'] == "BUY 🟢"]
sell_signals = test_data[test_data['Signal'] == "SELL 🔴"]

fig.add_trace(go.Scatter(
    x=buy_signals.index, y=buy_signals['Close'],
    mode='markers', name='BUY',
    marker=dict(symbol='triangle-up', size=12, color='green')
))

fig.add_trace(go.Scatter(
    x=sell_signals.index, y=sell_signals['Close'],
    mode='markers', name='SELL',
    marker=dict(symbol='triangle-down', size=12, color='red')
))

fig.update_layout(
    title=f"{ticker} - ARIMA Forecast & Trading Signals",
    xaxis_title="Date",
    yaxis_title="Price",
    hovermode='x unified',
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# === Metrics & Table ===
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Last Close", f"${data['Close'].iloc[-1]:.2f}")
with col2:
    st.metric("Forecast (Last)", f"${forecast[-1]:.2f}")
with col3:
    last_error = test_data['% Error'].iloc[-1]
    st.metric("Last % Error", f"{last_error:+.2f}%")

st.subheader("Recent Signals")
signal_df = test_data[['Close', 'Forecast', '% Error', 'Signal']].round(2)
st.dataframe(signal_df, use_container_width=True)

# === Model Summary ===
with st.expander("Model Details"):
    st.write("**Best ARIMA Order:**", model.order)
    st.write("**AIC:**", f"{model.aic():.2f}")
    st.code(model.summary().as_text()[:1000] + "\n...")

# === Footer ===
st.markdown("---")
st.caption("Auto ARIMA agent generates **BUY** when actual price is significantly below forecast, **SELL** when above. Deployed on Render.")