# chart_quickchart.py
import json, pandas as pd, numpy as np, urllib.parse

def make_quickchart_url(symbol, df_5_last21, df_daily,
                        adr, perf_open, rs_score, fcs, group_name, group_rs):
    """
    Baut die QuickChart-URL für ein Symbol:
    - 21×5-Min Kerzen (candlestick)
    - EMA6 / EMA21 (intraday)
    - Daily-EMA10/EMA21 (gestrichelt, nur wenn im Sichtbereich)
    - Titel mit ADR%, Perf%, RS, FCS und MACD-Signalfolge (●/○/◐)
    """
    if df_5_last21 is None or df_5_last21.empty:
        return ""

    col_up, col_dn = "#3F7E6A", "#BB4E36"
    col_ema6, col_ema21 = "#E8BE40", "#000000"

    # Intraday EMAs
    close = pd.to_numeric(df_5_last21["close"], errors="coerce")
    df_5_last21 = df_5_last21.copy()
    df_5_last21["ema6"]  = close.ewm(span=6,  adjust=False).mean()
    df_5_last21["ema21"] = close.ewm(span=21, adjust=False).mean()

    x_vals = list(range(len(df_5_last21)))
    candle_data = [
        {"x":x_vals[i],
         "o":float(df_5_last21.loc[i,'open']),
         "h":float(df_5_last21.loc[i,'high']),
         "l":float(df_5_last21.loc[i,'low']),
         "c":float(df_5_last21.loc[i,'close'])}
        for i in range(len(df_5_last21))
    ]
    ema6_pts  = [{"x":x_vals[i],"y":float(df_5_last21.loc[i,'ema6'])}  for i in range(len(df_5_last21))]
    ema21_pts = [{"x":x_vals[i],"y":float(df_5_last21.loc[i,'ema21'])} for i in range(len(df_5_last21))]

    # Daily-EMAs (nur Linie an aktueller Y-Position, falls im Bereich)
    ema10_val = ema21_val = None
    if df_daily is not None and not df_daily.empty and "close" in df_daily.columns:
        dclose = pd.to_numeric(df_daily["close"], errors="coerce")
        if len(dclose) >= 10:
            ema10_val = float(dclose.ewm(span=10, adjust=False).mean().iloc[-1])
        if len(dclose) >= 21:
            ema21_val = float(dclose.ewm(span=21, adjust=False).mean().iloc[-1])

    annotations = {}
    price_min, price_max = float(df_5_last21["low"].min()), float(df_5_last21["high"].max())
    if ema10_val and price_min <= ema10_val <= price_max:
        annotations["ema10"] = {"type":"line","yMin":ema10_val,"yMax":ema10_val,
                                "borderColor":"rgba(128,128,128,0.6)",
                                "borderDash":[6,6],"borderWidth":1.2}
    if ema21_val and price_min <= ema21_val <= price_max:
        annotations["ema21"] = {"type":"line","yMin":ema21_val,"yMax":ema21_val,
                                "borderColor":"rgba(0,0,0,0.8)",
                                "borderDash":[6,6],"borderWidth":1.2}

    # MACD-Signalfolge (6,20,9) – ●/○/◐ für die letzten 5 Bars
    macd_signal_str = ""
    if len(close) >= 9:
        ema_fast = close.ewm(span=6,  adjust=False).mean()
        ema_slow = close.ewm(span=20, adjust=False).mean()
        macd  = ema_fast - ema_slow
        signal= macd.ewm(span=9, adjust=False).mean()
        eps = 1e-5
        bull, bear, flat = "●","○","◐"
        for i in range(-5,0):
            if i >= -len(macd):
                diff = float(macd.iloc[i] - signal.iloc[i])
                macd_signal_str += bull if diff>eps else bear if diff<-eps else flat

    # Titelzeilen (Header über dem Chart)
    parts = []
    if pd.notna(adr):        parts.append(f"ADR: {adr*100:.1f}%")
    if pd.notna(perf_open):  parts.append(f"{perf_open*100:+.1f}%")
    if pd.notna(rs_score):   parts.append(f"RS: {rs_score:+.2f}")
    if pd.notna(fcs):        parts.append(f"FCS: {fcs:+.2f}")
    if macd_signal_str:      parts.append(f"MACD: {macd_signal_str}")
    subline = " | ".join(parts)
    gline = f"{group_name or ''}" + (f" | {group_rs:+.2f}" if group_rs is not None and pd.notna(group_rs) else "")
    title_text = f"{symbol}\n{subline}\n{gline}".strip()

    chart_config = {
        "type": "candlestick",
        "data": {"datasets": [
            {"type":"candlestick","data": candle_data,
             "borderColor":{"up":col_up,"down":col_dn,"unchanged":"#999"},
             "color":{"up":col_up,"down":col_dn,"unchanged":"#999"},
             "barPercentage":0.8,"categoryPercentage":0.7},
            {"type":"line","data":ema6_pts,"borderColor":col_ema6,
             "borderWidth":1.0,"fill":False,"pointRadius":0},
            {"type":"line","data":ema21_pts,"borderColor":col_ema21,
             "borderWidth":1.0,"fill":False,"pointRadius":0}
        ]},
        "options": {
            "animation": False,
            "plugins": {
                "title":   {"display": True,"text": title_text,"position": "top",
                            "align": "start","font": {"size": 12},"color": "rgba(90,90,90,1)"},
                "legend":  {"display": False},
                "tooltip": {"enabled": False},
                "annotation":{"annotations": annotations}
            },
            "scales": {"x": {"display": False}, "y": {"display": True}}
        }
    }

    base = ("https://quickchart.io/chart?"
            "width=300&height=200&backgroundColor=transparent&version=3"
            "&plugins=chartjs-chart-financial,chartjs-plugin-annotation&c=")
    return base + urllib.parse.quote(json.dumps(chart_config))
