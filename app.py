# app.py — RS Dashboard (Tiingo + QuickChart + CSV-Upload + Group-Score B + Sync-Refresh :05)

import io, os, time, json, math
import numpy as np
import pandas as pd
import requests
import streamlit as st
from datetime import datetime, timedelta
from dateutil import tz
from chart_quickchart import make_quickchart_url

# ============ Grund-Setup ============
st.set_page_config(page_title="Intraday RS Dashboard", layout="wide")
ET = tz.gettz("America/New_York")
UTC = tz.gettz("UTC")
SPY = "SPY"

# ---- Sidebar: Einstellungen ----
st.sidebar.title("⚙️ Einstellungen")

api_key = st.sidebar.text_input("Tiingo API Key", type="password", value=st.secrets.get("TIINGO_API_KEY", ""))
days_intraday = st.sidebar.number_input("Intraday Tage (5-Min, Kalendertage)", 10, 120, value=40, step=5)
days_daily    = st.sidebar.number_input("Daily Tage (Kalendertage)", 30, 200, value=60, step=5)
benchmark     = st.sidebar.text_input("Benchmark", value=SPY).strip().upper()

# Mapping: Upload ODER URL/Cache
st.sidebar.subheader("📂 Mapping (symbol,group)")
uploaded_file = st.sidebar.file_uploader("CSV hochladen", type=["csv"], key="mapping_upload")
df_mapping = pd.DataFrame(columns=["symbol","group"])
if uploaded_file:
    try:
        df_mapping = pd.read_csv(uploaded_file)
        if {"symbol","group"}.issubset(df_mapping.columns):
            df_mapping["symbol"] = df_mapping["symbol"].astype(str).str.upper().str.strip()
            df_mapping["group"]  = df_mapping["group"].astype(str).str.strip()
            st.session_state["mapping_df"] = df_mapping.copy()
            with open("mapping_latest.csv","wb") as f: f.write(uploaded_file.getbuffer())
            st.sidebar.success(f"✅ {len(df_mapping)} Zuordnungen geladen.")
        else:
            st.sidebar.error("Spalten 'symbol' und 'group' fehlen.")
            df_mapping = pd.DataFrame(columns=["symbol","group"])
    except Exception as e:
        st.sidebar.error(f"Fehler beim Einlesen: {e}")
else:
    if "mapping_df" in st.session_state and not st.session_state["mapping_df"].empty:
        df_mapping = st.session_state["mapping_df"].copy()
        st.sidebar.info(f"📄 Mapping aus Session geladen ({len(df_mapping)})")
    elif os.path.exists("mapping_latest.csv"):
        try:
            df_mapping = pd.read_csv("mapping_latest.csv")
            df_mapping["symbol"] = df_mapping["symbol"].astype(str).str.upper().str.strip()
            df_mapping["group"]  = df_mapping["group"].astype(str).str.strip()
            st.session_state["mapping_df"] = df_mapping.copy()
            st.sidebar.info(f"📄 Mapping aus lokalem Cache geladen ({len(df_mapping)})")
        except Exception as e:
            st.sidebar.warning(f"Cache-Datei fehlerhaft: {e}")

# Auto-Refresh exakt zu :05
st.sidebar.subheader("🔄 Auto-Refresh (synchronisiert)")
auto_opt = st.sidebar.selectbox("Intervall", ["Manuell", "1 Minute", "5 Minuten", "10 Minuten"], index=2)

def seconds_until_next(interval_min=5):
    now = datetime.now()
    minute = ((now.minute // interval_min) + 1) * interval_min
    next_t = now.replace(minute=minute % 60, second=5, microsecond=0)
    if minute >= 60: next_t += timedelta(hours=1)
    return max((next_t - now).total_seconds(), 0)

if auto_opt != "Manuell":
    freq = {"1 Minute":1, "5 Minuten":5, "10 Minuten":10}[auto_opt]
    delay = seconds_until_next(freq)
    st.sidebar.caption(f"Nächster Refresh: {(datetime.now()+timedelta(seconds=delay)):%H:%M:%S}")
    time.sleep(delay)
    st.experimental_rerun()

# ============ Hilfsfunktionen ============
def date_str(days_back: int) -> str:
    return (datetime.now(tz=UTC) - timedelta(days=days_back)).date().isoformat()

@st.cache_data(ttl=300, show_spinner=False)
def fetch_intraday_5min(symbol: str, start_date: str, token: str) -> pd.DataFrame:
    url = (f"https://api.tiingo.com/iex/{symbol}/prices"
           f"?startDate={start_date}&resampleFreq=5min"
           f"&columns=open,high,low,close,volume,date&token={token}")
    r = requests.get(url, timeout=40); 
    if r.status_code == 429: time.sleep(1.0); r = requests.get(url, timeout=40)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data)
    if df.empty: return df
    df["datetime"] = pd.to_datetime(df["date"], utc=True)
    df["datetime_et"] = df["datetime"].dt.tz_convert(ET)
    df = df.rename(columns=str.lower)
    return df[["datetime","datetime_et","open","high","low","close","volume"]].sort_values("datetime").reset_index(drop=True)

@st.cache_data(ttl=24*3600, show_spinner=False)
def fetch_daily_adj(symbol: str, start_date: str, token: str) -> pd.DataFrame:
    url = (f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
           f"?startDate={start_date}&columns=adjOpen,adjHigh,adjLow,adjClose,adjVolume,date&token={token}")
    r = requests.get(url, timeout=40); r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data)
    if df.empty: return df
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(UTC)
    df = df.rename(columns=lambda c: c.replace("adj","").lower())
    return df[["date","open","high","low","close","volume"]].sort_values("date").reset_index(drop=True)

def ema(s: pd.Series, n: int) -> pd.Series: 
    return s.ewm(span=n, adjust=False).mean()

def candle_strength_from_row(row) -> float:
    h,l,o,c = float(row["high"]), float(row["low"]), float(row["open"]), float(row["close"])
    rng = h - l
    if rng <= 0: return 0.0
    body = abs(c - o); body_ratio = body / rng
    open_pos  = (o - l) / rng
    close_pos = (c - l) / rng
    score = 0.0
    if c > o:
        if body_ratio >= 0.55: score += 0.3
        if open_pos  <= 0.20:  score += 0.2
        if close_pos >= 0.80:  score += 0.3
        score += 0.2
    elif c < o:
        if body_ratio >= 0.55: score -= 0.3
        if open_pos  >= 0.80:  score -= 0.2
        if close_pos <= 0.20:  score -= 0.3
        score -= 0.2
    return float(max(-1.0, min(1.0, score)))

def perf_since_open(df_5: pd.DataFrame) -> float:
    if df_5.empty: return np.nan
    df = df_5.copy()
    df["date_et"] = df["datetime_et"].dt.date
    today = datetime.now(tz=ET).date()
    d = df[df["date_et"]==today]
    if d.empty: return np.nan
    t0930 = datetime(2000,1,1,9,30, tzinfo=ET).timetz()
    s = d[d["datetime_et"].dt.time >= t0930]
    if s.empty: return np.nan
    o0 = float(s.iloc[0]["open"]); cN = float(s.iloc[-1]["close"])
    return (cN - o0)/o0 if o0 else np.nan

def opening_candle_strength(df_5: pd.DataFrame) -> float:
    if df_5.empty: return np.nan
    df = df_5.copy()
    df["date_et"] = df["datetime_et"].dt.date
    t0930 = datetime(2000,1,1,9,30, tzinfo=ET).timetz()
    today = datetime.now(tz=ET).date()
    first = df[(df["date_et"]==today) & (df["datetime_et"].dt.time==t0930)]
    if first.empty: return np.nan
    return candle_strength_from_row(first.iloc[0])

def rvol_buzz_kumuliert(df_5: pd.DataFrame) -> float:
    """CumVol bis aktuelle Bar / Ø CumVol gleicher Slot über vergangene Tage (TC2000-Stil)."""
    if df_5.empty: return np.nan
    df = df_5.copy()
    df["date_et"] = df["datetime_et"].dt.date
    df["time_et"] = df["datetime_et"].dt.time
    t_start = datetime(2000,1,1,9,30, tzinfo=ET).timetz()
    t_end   = datetime(2000,1,1,16, 0, tzinfo=ET).timetz()
    df = df[(df["time_et"]>=t_start)&(df["time_et"]<=t_end)]
    if df.empty: return np.nan
    df["slot"] = df.groupby("date_et").cumcount()
    df["cum_vol"] = df.groupby("date_et")["volume"].cumsum()

    today = datetime.now(tz=ET).date()
    todays = df[df["date_et"]==today]
    if todays.empty: return np.nan
    last_slot = int(todays["slot"].max())
    cum_today = float(df[(df["date_et"]==today)&(df["slot"]==last_slot)]["cum_vol"].iloc[0])
    prev = df[(df["date_et"]<today)&(df["slot"]==last_slot)]
    if prev.empty: return np.nan
    avg_prev = float(prev["cum_vol"].mean())
    return cum_today/avg_prev if avg_prev>0 else np.nan

def ema_context_bonus(df_5: pd.DataFrame, df_daily: pd.DataFrame) -> float:
    bonus = 0.0
    if len(df_5) >= 6:
        ema6 = float(ema(df_5["close"],6).iloc[-1])
        if float(df_5["close"].iloc[-1]) > ema6: bonus += 0.10
    if len(df_daily) >= 21:
        e10 = float(ema(df_daily["close"],10).iloc[-1])
        e21 = float(ema(df_daily["close"],21).iloc[-1])
        last = float(df_daily["close"].iloc[-1])
        if last > e10: bonus += 0.10
        if last > e21: bonus += 0.10
    return bonus

def rvol_bonus_from_value(v: float) -> float:
    if pd.isna(v): return 0.0
    if v >= 2.0:  return 0.15
    if v >= 1.5:  return 0.10
    if v >= 1.0:  return 0.05
    return 0.0

def build_group_series(frames: list) -> pd.DataFrame:
    """Variante B: synthetische Gruppenserie (5-Min):
       open/close = geom. Mittel; high=max; low=min; volume=sum."""
    if len(frames) < 2: return pd.DataFrame()
    dfs = []
    for df in frames:
        if df is None or df.empty: continue
        tmp = df[["datetime","datetime_et","open","high","low","close","volume"]].copy().set_index("datetime")
        dfs.append(tmp)
    if len(dfs) < 2: return pd.DataFrame()
    idx = dfs[0].index
    for d in dfs[1:]:
        idx = idx.intersection(d.index)
    if len(idx) == 0: return pd.DataFrame()
    aligned = [d.loc[idx] for d in dfs]
    open_geo  = np.exp(np.log(np.vstack([a["open"].values  for a in aligned])).mean(axis=0))
    close_geo = np.exp(np.log(np.vstack([a["close"].values for a in aligned])).mean(axis=0))
    high_g    = np.max(np.vstack([a["high"].values for a in aligned]), axis=0)
    low_g     = np.min(np.vstack([a["low"].values  for a in aligned]), axis=0)
    vol_g     = np.sum(np.vstack([a["volume"].values for a in aligned]), axis=0)
    out = pd.DataFrame({"datetime": idx, "open": open_geo, "high": high_g, "low": low_g, "close": close_geo, "volume": vol_g})
    out["datetime_et"] = out["datetime"].dt.tz_convert(ET)
    return out.sort_values("datetime").reset_index(drop=True)

# ============ Daten laden ============
st.title("📊 Intraday Relative Strength Dashboard (QuickChart, Tiingo)")

if not api_key:
    st.info("Bitte Tiingo API Key in der Sidebar eintragen (oder in Secrets als TIINGO_API_KEY).")
    st.stop()

if df_mapping.empty:
    st.warning("Kein Mapping geladen — bitte CSV mit Spalten 'symbol,group' hochladen.")
    st.stop()

symbols = df_mapping["symbol"].unique().tolist()
start_5m   = date_str(days_intraday)
start_day  = date_str(days_daily)

# Benchmark
try:
    df5_bm = fetch_intraday_5min(benchmark, start_5m, api_key)
    dfd_bm = fetch_daily_adj(benchmark, start_day, api_key)
except Exception as e:
    st.error(f"Benchmark-Fehler ({benchmark}): {e}")
    st.stop()
if df5_bm.empty:
    st.error("Keine Benchmark-Intradaydaten verfügbar.")
    st.stop()

# Symboldaten + Caches
rows = []
df5_cache = {}
dfd_cache = {}
group_frames = {}

progress = st.progress(0.0, text="Lade Symbole …")
for i, sym in enumerate(symbols):
    progress.progress((i+1)/max(1,len(symbols)), text=f"Lade {sym} …")
    try:
        df5 = fetch_intraday_5min(sym, start_5m, api_key)
        dfd = fetch_daily_adj(sym, start_day, api_key)
    except Exception as e:
        st.warning(f"{sym}: API-Fehler – {e}")
        continue

    df5_cache[sym] = df5
    dfd_cache[sym] = dfd

    # Metriken
    cs_sym = candle_strength_from_row(df5.iloc[-1]) if not df5.empty else np.nan
    cs_bm  = candle_strength_from_row(df5_bm.iloc[-1]) if not df5_bm.empty else np.nan
    rs_rel = cs_sym - cs_bm if (pd.notna(cs_sym) and pd.notna(cs_bm)) else np.nan
    perf   = perf_since_open(df5)
    fcs    = opening_candle_strength(df5)

    # ADR(20) (adjusted)
    if dfd is not None and not dfd.empty and len(dfd) >= 20:
        adr20 = float((dfd["high"] - dfd["low"]).tail(20).mean())
    else:
        adr20 = np.nan

    # Boni nur bei starker Candle (>=0.70)
    ema_bonus = rvol_val = rvol_bonus = 0.0
    if pd.notna(cs_sym) and cs_sym >= 0.70:
        ema_bonus = ema_context_bonus(df5, dfd) if (df5 is not None and dfd is not None) else 0.0
        rvol_val  = rvol_buzz_kumuliert(df5)
        rvol_bonus= rvol_bonus_from_value(rvol_val)

    # Performancebonus (wie besprochen, leichter Boost)
    perf_bonus = 0.10 if (pd.notna(perf) and perf > 0.01) else 0.05 if (pd.notna(perf) and perf > 0.005) else 0.0

    # Gesamt-RS (–1..+1), Gewichte: RS_rel 0.5, EMA 0.2, RVOL 0.2, Perf 0.1
    rs_score = 0.0
    if pd.notna(rs_rel):
        rs_score = (0.50 * rs_rel) + (0.20 * ema_bonus) + (0.20 * rvol_bonus) + (0.10 * perf_bonus)
        rs_score = max(-1.0, min(1.0, rs_score))
    else:
        rs_score = np.nan

    grp = df_mapping.loc[df_mapping["symbol"]==sym, "group"].iloc[0]
    rows.append({
        "symbol": sym,
        "group": grp,
        "adr20": adr20,
        "perf_open": perf,
        "rs_score": rs_score,
        "opening_score": fcs
    })

    # für Gruppenserie sammeln
    if grp not in group_frames: group_frames[grp] = []
    if df5 is not None and not df5.empty: group_frames[grp].append(df5)

progress.empty()

df_final = pd.DataFrame(rows)
if df_final.empty:
    st.error("Keine Symbolwerte berechnet.")
    st.stop()

# ============ Group Score (Variante B: synthetische Serie) ============
group_scores = []
for grp, frames in group_frames.items():
    members = df_mapping[df_mapping["group"]==grp]["symbol"].tolist()
    if len(frames) < 2 or len(members) < 2:
        group_scores.append({"group": grp, "group_rs_score": np.nan})
        continue

    df_g = build_group_series(frames)
    if df_g.empty:
        group_scores.append({"group": grp, "group_rs_score": np.nan})
        continue

    cs_grp = candle_strength_from_row(df_g.iloc[-1])
    cs_bm  = candle_strength_from_row(df5_bm.iloc[-1]) if not df5_bm.empty else np.nan
    rs_grp = cs_grp - cs_bm if (pd.notna(cs_grp) and pd.notna(cs_bm)) else np.nan

    # EMA-Bonus Gruppe (nur bei starker Candle)
    ema_bonus_grp = 0.0
    if pd.notna(cs_grp) and cs_grp >= 0.70:
        # intraday EMA6 auf Gruppenserie
        if len(df_g) >= 6:
            ema6_g = float(ema(df_g["close"],6).iloc[-1])
            if float(df_g["close"].iloc[-1]) > ema6_g: ema_bonus_grp += 0.10
        # Daily-Approx: Median der Member-EMAs (robust & schnell)
        e10_vals, e21_vals, last_cls = [], [], []
        for m in members:
            dfd_m = dfd_cache.get(m)
            if dfd_m is not None and len(dfd_m) >= 21:
                e10_vals.append(float(ema(dfd_m["close"],10).iloc[-1]))
                e21_vals.append(float(ema(dfd_m["close"],21).iloc[-1]))
                last_cls.append(float(dfd_m["close"].iloc[-1]))
        if e10_vals and e21_vals and last_cls:
            last_med = float(np.median(last_cls))
            if last_med > float(np.median(e10_vals)): ema_bonus_grp += 0.10
            if last_med > float(np.median(e21_vals)): ema_bonus_grp += 0.10

    # RVOL-Buzz Gruppe (kumuliert)
    rvol_grp_val = rvol_grp_bonus = 0.0
    if pd.notna(cs_grp) and cs_grp >= 0.70:
        rvol_grp_val   = rvol_buzz_kumuliert(df_g)
        rvol_grp_bonus = rvol_bonus_from_value(rvol_grp_val)

    # kleiner Perf-Bonus Gruppe
    # (optional; hier analog Einzelsymbol über perf_since_open der Gruppe)
    perf_grp = np.nan
    try:
        perf_grp = (df_g["close"].iloc[-1] - df_g["open"].iloc[0]) / df_g["open"].iloc[0]
    except Exception:
        pass
    perf_grp_bonus = 0.10 if (pd.notna(perf_grp) and perf_grp > 0.01) else 0.05 if (pd.notna(perf_grp) and perf_grp > 0.005) else 0.0

    group_rs_score = (0.50*rs_grp) + (0.20*ema_bonus_grp) + (0.20*rvol_grp_bonus) + (0.10*perf_grp_bonus) if pd.notna(rs_grp) else np.nan
    group_rs_score = max(-1.0, min(1.0, group_rs_score)) if pd.notna(group_rs_score) else np.nan

    group_scores.append({"group": grp, "group_rs_score": group_rs_score})

df_group = pd.DataFrame(group_scores)
df_final = df_final.merge(df_group, on="group", how="left")

# ============ Charts anzeigen (3 pro Zeile, QuickChart) ============
df_sorted = df_final.sort_values("rs_score", ascending=False)
cols = st.columns(3)
for i, row in enumerate(df_sorted.itertuples()):
    sym = row.symbol
    d5  = df5_cache.get(sym)
    dd  = dfd_cache.get(sym)
    # nur letzte 21 Bars zeigen
    last21 = d5.tail(21).reset_index(drop=True) if d5 is not None and not d5.empty else pd.DataFrame()
    col = cols[i % 3]
    with col:
        url = make_quickchart_url(
            sym, last21, dd,
            row.adr20, row.perf_open, row.rs_score, row.opening_score,
            row.group, row.group_rs_score
        )
        if url:
            st.image(url, use_column_width=True)
        else:
            st.write(f"{sym}: keine Daten")
