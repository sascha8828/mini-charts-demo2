import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Mini Chart Demo", layout="wide")
st.title("📊 Plotly + Streamlit Mini-Chart Beispiel")

def get_fake_data():
    dates = pd.date_range("2025-11-02 14:30", periods=30, freq="5min")
    base = np.cumsum(np.random.randn(len(dates))) + 100
    df = pd.DataFrame({
        "date": dates,
        "open": base,
        "high": base + np.random.rand(len(dates)),
        "low": base - np.random.rand(len(dates)),
        "close": base + np.random.randn(len(dates)) / 2
    })
    return df

symbols = ["AAPL", "NVDA", "PLTR", "TSLA", "SMCI", "AMD", "META", "GOOG", "CRDO", "HOOD"]
st.sidebar.header("⚙️ Optionen")
num_charts = st.sidebar.slider("Anzahl der Charts", 1, 20, 10)

cols = st.columns(5)
for i, sym in enumerate(symbols[:num_charts]):
    df = get_fake_data()
    fig = go.Figure(data=[go.Candlestick(
        x=df["date"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        increasing_line_color="#3F7E6A",
        decreasing_line_color="#BB4E36",
    )])
    fig.update_layout(
        height=180,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    with cols[i % 5]:
        st.plotly_chart(fig, use_container_width=True)
        st.caption(sym)
