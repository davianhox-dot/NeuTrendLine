"""
Trendline Bounce Scanner PRO – Streamlit App
============================================
Scannt Aktien & Kryptos nach langfristigen Support-Trendlinien (linear & log),
mit Bounce-Bestätigung, Volumen-Check, Konfluenz-Score, Earnings-Warnung,
Qualitäts- & Insider-Daten, Marktfilter, relativer Stärke und Backtest.

Lokal:   pip install -r requirements.txt && streamlit run app.py
Online:  GitHub-Repo -> share.streamlit.io -> Deploy -> URL zum Homescreen.
"""

import io
import math
import itertools
import datetime as dt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Trendline Scanner Pro", page_icon="📉", layout="wide")

PIVOT_WINDOW = 3
TF_CFG = {"1W": {"interval": "1wk", "period": "10y"},
          "1M": {"interval": "1mo", "period": "max"}}
SECTOR_ETF = {"Technology": "XLK", "Financial Services": "XLF", "Healthcare": "XLV",
              "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP", "Energy": "XLE",
              "Industrials": "XLI", "Basic Materials": "XLB", "Utilities": "XLU",
              "Real Estate": "XLRE", "Communication Services": "XLC"}

# ============================ Sidebar ============================
st.sidebar.title("⚙️ Einstellungen")
scan_mode = st.sidebar.radio(
    "Scan-Modus",
    ["Viele Bounces (UiPath-Stil)", "Starke lange Linien (ServiceNow/Salesforce-Stil)"])
max_dist   = st.sidebar.slider("Max. Distanz zur Linie (%)", 1.0, 15.0, 5.0, 0.5)
min_bounce = st.sidebar.slider("Min. Anzahl Bounces", 2, 6, 3)
touch_tol  = st.sidebar.slider("Touch-Toleranz (%)", 0.5, 5.0, 2.0, 0.5)
break_tol  = st.sidebar.slider("Bruch-Toleranz (%)", 0.5, 5.0, 1.5, 0.5)
min_price  = st.sidebar.number_input("Min. Kurs ($)", 0.0, 100.0, 1.0)

st.sidebar.markdown("**Filter**")
need_confirm = st.sidebar.checkbox("Nur bestätigte Bounces (grüne Kerze / Hammer an der Linie)", value=False)
need_quality = st.sidebar.checkbox("Qualitäts-Filter (Umsatzwachstum > 0, Schulden ok)", value=False)
use_log      = st.sidebar.checkbox("Auch logarithmische Trendlinien", value=True)

universe_choice = st.sidebar.selectbox(
    "Ticker-Universum",
    ["Schnell: S&P 500 + Krypto (~520)",
     "Mittel: + NASDAQ-100 + DAX/MDAX/SDAX (~650)",
     "Voll: Alle NASDAQ + NYSE + DE + Krypto (~8000)"])
timeframes = st.sidebar.multiselect("Timeframes", ["1W", "1M"], default=["1W", "1M"])
batch_size = st.sidebar.select_slider("Chargengröße", [100, 250, 500, 1000], value=500)
auto_cont  = st.sidebar.checkbox("Automatisch weiterscannen", value=True)

# ============================ Universum ============================
def read_html_ua(url):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                     timeout=20)
    r.raise_for_status()
    return pd.read_html(io.StringIO(r.text))

@st.cache_data(ttl=86400, show_spinner=False)
def get_universe(choice: str):
    tickers, errors = set(), []
    try:
        sp500 = read_html_ua("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers.update(sp500["Symbol"].str.replace(".", "-", regex=False))
    except Exception as e:
        errors.append(f"S&P 500: {e}")
    tickers.update(["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
                    "ADA-USD", "DOGE-USD", "AVAX-USD", "LINK-USD", "DOT-USD",
                    "LTC-USD", "ATOM-USD", "NEAR-USD", "ARB-USD", "INJ-USD"])
    if choice.startswith("Schnell"):
        return sorted(tickers), errors
    try:
        for t in read_html_ua("https://en.wikipedia.org/wiki/Nasdaq-100"):
            if "Ticker" in t.columns:
                tickers.update(t["Ticker"]); break
    except Exception as e:
        errors.append(f"NASDAQ-100: {e}")
    for page in ["DAX", "MDAX", "SDAX"]:
        try:
            for t in read_html_ua(f"https://en.wikipedia.org/wiki/{page}"):
                col = next((c for c in t.columns if "icker" in str(c) or "ymbol" in str(c)), None)
                if col is not None and len(t) > 20:
                    syms = t[col].dropna().astype(str)
                    tickers.update(s if s.endswith(".DE") else s + ".DE" for s in syms)
                    break
        except Exception as e:
            errors.append(f"{page}: {e}")
    if choice.startswith("Mittel"):
        return sorted(tickers), errors
    try:
        nq = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", sep="|")
        nq = nq[(nq["Test Issue"] == "N") & (nq["ETF"] == "N")]
        tickers.update(nq["Symbol"].dropna())
    except Exception as e:
        errors.append(f"NASDAQ-Liste: {e}")
    try:
        ot = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", sep="|")
        ot = ot[(ot["Test Issue"] == "N") & (ot["ETF"] == "N")]
        syms = ot["ACT Symbol"].dropna()
        syms = syms[~syms.str.contains(r"[\$\.]")]
        tickers.update(syms.str.replace("/", "-"))
    except Exception as e:
        errors.append(f"NYSE-Liste: {e}")
    return sorted(t for t in tickers if isinstance(t, str) and t.strip()), errors

# ============================ Trendlinien ============================
def find_pivot_lows(vals, window=PIVOT_WINDOW):
    return [i for i in range(window, len(vals) - window)
            if vals[i] == vals[i - window:i + window + 1].min()]

def best_trendline(df, min_bounces, touch_tol_pct, break_tol_pct,
                   strong_mode=False, log_scale=False):
    """Rueckgabe: (slope, intercept, touches, log_scale) oder None.
    log_scale: Linie wird im log-Preisraum gefittet (Dekaden-Charts)."""
    raw_lows = df["Low"].values.astype(float)
    raw_close = df["Close"].values.astype(float)
    if log_scale and (raw_lows <= 0).any():
        return None
    lows = np.log(raw_lows) if log_scale else raw_lows
    pivots = find_pivot_lows(raw_lows)
    if len(pivots) < 2:
        return None
    n = len(df)
    min_span = int(n * 0.6) if strong_mode else 10
    req_bounces = 2 if strong_mode else min_bounces
    best = None
    for i, j in itertools.combinations(pivots, 2):
        if j - i < min_span:
            continue
        slope = (lows[j] - lows[i]) / (j - i)
        if strong_mode and slope <= 0:
            continue
        intercept = lows[i] - slope * i
        line = slope * np.arange(n) + intercept
        line_px = np.exp(line) if log_scale else line
        if (line_px <= 0).any():
            continue
        if (raw_close[i:] < line_px[i:] * (1 - break_tol_pct / 100)).any():
            continue
        touches = [p for p in pivots if p >= i and
                   abs(raw_lows[p] - line_px[p]) / line_px[p] * 100 <= touch_tol_pct]
        if len(touches) < req_bounces:
            continue
        score = (j - i, len(touches)) if strong_mode else (len(touches), j - i)
        if best is None or score > best[0]:
            best = (score, slope, intercept, touches)
    return None if best is None else (*best[1:], log_scale)

def line_values(hit):
    x = np.arange(len(hit["df"]))
    line = hit["slope"] * x + hit["intercept"]
    return np.exp(line) if hit["log"] else line

# ============================ Signal-Checks ============================
def bounce_confirmed(df):
    """Letzte Kerze grün ODER Hammer (langer unterer Docht)."""
    o, h, l, c = (float(df[k].iloc[-1]) for k in ["Open", "High", "Low", "Close"])
    body = abs(c - o)
    lower_wick = min(o, c) - l
    green = c > o
    hammer = body > 0 and lower_wick > 2 * body and (h - max(o, c)) < body
    return green or hammer

def volume_signal(df):
    """>1 = letzte Kerze hat überdurchschnittliches Volumen."""
    if "Volume" not in df or df["Volume"].iloc[-20:].sum() == 0:
        return None
    avg = float(df["Volume"].iloc[-21:-1].mean())
    return float(df["Volume"].iloc[-1]) / avg if avg > 0 else None

def confluence_factors(df, line_now, tol_pct=3.0):
    """Weitere Support-Level nahe der Trendlinie -> stärkeres Setup."""
    out = []
    close = df["Close"]
    sma200 = close.rolling(min(200, len(close) - 1)).mean().iloc[-1]
    if abs(sma200 - line_now) / line_now * 100 <= tol_pct:
        out.append("SMA200 auf der Linie")
    # Fibonacci 0.618 des letzten großen Aufwärtsimpulses
    lo_i = int(close.values.argmin())
    hi_i = lo_i + int(close.values[lo_i:].argmax())
    if hi_i > lo_i:
        lo, hi = float(close.iloc[lo_i]), float(close.iloc[hi_i])
        fib618 = hi - 0.618 * (hi - lo)
        if abs(fib618 - line_now) / line_now * 100 <= tol_pct:
            out.append("Fib 0.618 Konfluenz")
    # Horizontale Support-Zone: frühere Pivot-Lows nahe dem Level
    lows = df["Low"].values
    piv = find_pivot_lows(lows)
    near = [p for p in piv[:-1] if abs(lows[p] - line_now) / line_now * 100 <= tol_pct]
    if len(near) >= 2:
        out.append(f"Horizontaler Support ({len(near)} alte Tiefs)")
    return out

# ============================ Fundamental / Kontext ============================
@st.cache_data(ttl=43200, show_spinner=False)
def get_fundamentals(ticker):
    out = {"target": None, "upside": None, "rec": None, "rec_score": None,
           "n_analysts": None, "news": [], "earnings_days": None,
           "rev_growth": None, "debt_eq": None, "sector": None,
           "insider_buys": 0, "insider_sells": 0, "quality_ok": True}
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        out["target"] = info.get("targetMeanPrice")
        out["rec"] = info.get("recommendationKey")
        out["rec_score"] = info.get("recommendationMean")
        out["n_analysts"] = info.get("numberOfAnalystOpinions")
        out["rev_growth"] = info.get("revenueGrowth")
        out["debt_eq"] = info.get("debtToEquity")
        out["sector"] = info.get("sector")
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if out["target"] and price:
            out["upside"] = (out["target"] / price - 1) * 100
        # Qualität: Umsatz wächst & Schulden nicht extrem
        rg, de = out["rev_growth"], out["debt_eq"]
        if rg is not None and rg < 0:
            out["quality_ok"] = False
        if de is not None and de > 300:
            out["quality_ok"] = False
        # Earnings-Termin
        try:
            ed = t.get_earnings_dates(limit=4)
            if ed is not None and len(ed):
                future = [d for d in ed.index.tz_localize(None)
                          if d >= pd.Timestamp.now()]
                if future:
                    out["earnings_days"] = (min(future) - pd.Timestamp.now()).days
        except Exception:
            pass
        # Insider-Transaktionen (letzte 6 Monate)
        try:
            ins = t.insider_transactions
            if ins is not None and len(ins):
                ins = ins.copy()
                if "Start Date" in ins.columns:
                    ins["Start Date"] = pd.to_datetime(ins["Start Date"], errors="coerce")
                    recent = ins[ins["Start Date"] >= pd.Timestamp.now() - pd.Timedelta(days=180)]
                else:
                    recent = ins
                txt = recent.get("Text", recent.get("Transaction", pd.Series(dtype=str))).astype(str).str.lower()
                out["insider_buys"] = int(txt.str.contains("purchase|buy").sum())
                out["insider_sells"] = int(txt.str.contains("sale|sell").sum())
        except Exception:
            pass
        for n in (t.news or [])[:3]:
            title = n.get("title") or (n.get("content") or {}).get("title")
            if title:
                out["news"].append(title)
    except Exception:
        pass
    return out

@st.cache_data(ttl=21600, show_spinner=False)
def market_status():
    """Marktfilter: S&P 500 über/unter SMA200 (Tageschart)."""
    try:
        spy = yf.download("SPY", period="2y", interval="1d", progress=False, auto_adjust=True)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        price = float(spy["Close"].iloc[-1])
        sma = float(spy["Close"].rolling(200).mean().iloc[-1])
        return price > sma, price, sma
    except Exception:
        return None, None, None

@st.cache_data(ttl=21600, show_spinner=False)
def sector_perf_3m(etf):
    try:
        df = yf.download(etf, period="4mo", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return float(df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    except Exception:
        return None

def relative_strength(df, sector):
    """Aktien-Performance 3M vs. Sektor-ETF."""
    etf = SECTOR_ETF.get(sector)
    if not etf or len(df) < 14:
        return None
    bars = 13 if len(df) > 60 else 3        # ~3 Monate in 1W bzw. 1M
    own = float(df["Close"].iloc[-1] / df["Close"].iloc[-bars] - 1) * 100
    sec = sector_perf_3m(etf)
    return None if sec is None else own - sec

# ============================ Scan ============================
def scan_ticker(ticker, tf, params):
    cfg = TF_CFG[tf]
    try:
        df = yf.download(ticker, interval=cfg["interval"], period=cfg["period"],
                         progress=False, auto_adjust=True)
    except Exception:
        return None
    if df is None or len(df) < 60:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if len(df) < 60 or float(df["Close"].iloc[-1]) < params["min_price"]:
        return None

    candidates = []
    scales = [False, True] if params["use_log"] else [False]
    for log_scale in scales:
        res = best_trendline(df, params["min_bounce"], params["touch_tol"],
                             params["break_tol"], strong_mode=params["strong"],
                             log_scale=log_scale)
        if res:
            candidates.append(res)
    if not candidates:
        return None
    # beste Variante: mehr Touches, dann längere Linie
    candidates.sort(key=lambda r: (len(r[2]),), reverse=True)
    slope, intercept, touches, log_scale = candidates[0]

    n = len(df) - 1
    line_now = math.exp(slope * n + intercept) if log_scale else slope * n + intercept
    price = float(df["Close"].iloc[-1])
    dist = (price - line_now) / line_now * 100
    if dist < -params["break_tol"] or dist > params["max_dist"]:
        return None
    if params["strong"]:
        recent_high = float(df["High"].iloc[-24:].max())
        if recent_high < price * 1.25:
            return None

    confirmed = bounce_confirmed(df)
    if params["need_confirm"] and not confirmed:
        return None

    hit = {"ticker": ticker, "tf": tf, "price": price, "line": float(line_now),
           "dist": float(dist), "bounces": len(touches), "df": df,
           "slope": slope, "intercept": intercept, "touches": touches,
           "log": log_scale, "confirmed": confirmed,
           "vol_ratio": volume_signal(df),
           "confluence": confluence_factors(df, line_now)}

    if params["need_quality"] and not ticker.endswith("-USD"):
        f = get_fundamentals(ticker)
        if not f["quality_ok"]:
            return None
    return hit

# ============================ Bewertung & Levels ============================
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])

def technical_assessment(hit, break_tol_pct, fund=None, mkt_bull=None):
    df, price = hit["df"], hit["price"]
    close = df["Close"]
    reasons, score = [], 0

    r = rsi(close)
    if not math.isnan(r):
        if r < 35:   score += 2; reasons.append(f"RSI {r:.0f} – überverkauft (bullisch)")
        elif r < 50: score += 1; reasons.append(f"RSI {r:.0f} – neutral-schwach")
        elif r > 65: score -= 1; reasons.append(f"RSI {r:.0f} – bereits gelaufen")
        else:        reasons.append(f"RSI {r:.0f} – neutral")

    sma200 = close.rolling(min(200, len(close) - 1)).mean().iloc[-1]
    if price > sma200: score += 1; reasons.append("Kurs über SMA200")
    else:              score -= 1; reasons.append("Kurs unter SMA200")

    ema12, ema26 = close.ewm(span=12).mean(), close.ewm(span=26).mean()
    macd = ema12 - ema26; sig = macd.ewm(span=9).mean()
    if macd.iloc[-1] > sig.iloc[-1]: score += 1; reasons.append("MACD dreht positiv")
    else:                            reasons.append("MACD noch negativ")

    if hit["bounces"] >= 4: score += 1; reasons.append(f"{hit['bounces']} Bounces – starke Linie")
    if abs(hit["dist"]) <= 2: score += 1; reasons.append("Sehr nah an der Linie")

    if hit["confirmed"]: score += 2; reasons.append("✅ Bounce bestätigt (grüne Kerze/Hammer)")
    else:                reasons.append("⏳ Noch keine Bounce-Bestätigung – fallendes Messer möglich")

    vr = hit["vol_ratio"]
    if vr is not None:
        if vr >= 1.5 and hit["confirmed"]:
            score += 1; reasons.append(f"Volumen {vr:.1f}× Schnitt – Käufer verteidigen das Level")
        elif vr >= 1.5:
            reasons.append(f"Volumen {vr:.1f}× Schnitt – hoher Verkaufsdruck, Vorsicht")

    for c in hit["confluence"]:
        score += 1; reasons.append(f"Konfluenz: {c}")

    if mkt_bull is True:    score += 1; reasons.append("Marktfilter: S&P 500 über SMA200 (Rückenwind)")
    elif mkt_bull is False: score -= 2; reasons.append("⚠️ Marktfilter: S&P 500 unter SMA200 – Bounces unzuverlässig")

    if fund:
        if fund.get("earnings_days") is not None and fund["earnings_days"] <= 14:
            score -= 1
            reasons.append(f"⚠️ Earnings in {fund['earnings_days']} Tagen – Gap-Risiko")
        if fund.get("insider_buys", 0) > fund.get("insider_sells", 0):
            score += 1; reasons.append(f"Insider kaufen ({fund['insider_buys']} Käufe / {fund['insider_sells']} Verkäufe, 6M)")
        if fund.get("rev_growth") is not None and fund["rev_growth"] < 0:
            score -= 1; reasons.append("Umsatz schrumpft – Abverkauf evtl. fundamental begründet")
        rs_val = fund.get("rel_strength")
        if rs_val is not None:
            if rs_val < -10: score -= 1; reasons.append(f"Schwächer als Sektor ({rs_val:+.0f}% rel. 3M)")
            elif rs_val > 0: score += 1; reasons.append(f"Stärker als Sektor ({rs_val:+.0f}% rel. 3M)")

    invalidation = hit["line"] * (1 - break_tol_pct / 100)
    highs = df["High"].values
    swing_highs = sorted({float(highs[i]) for i in range(3, len(highs) - 3)
                          if highs[i] == highs[i - 3:i + 4].max() and highs[i] > price * 1.02})
    tp1 = swing_highs[0] if swing_highs else price * 1.10
    tp2 = swing_highs[1] if len(swing_highs) > 1 else price * 1.20
    risk = price - invalidation
    rr1 = (tp1 - price) / risk if risk > 0 else float("nan")

    if score >= 5:   verdict = "🟢 Starkes Setup"
    elif score >= 2: verdict = "🟡 Okay – auf Bestätigung/Kontext achten"
    else:            verdict = "🔴 Schwach – eher meiden"

    return {"score": score, "verdict": verdict, "reasons": reasons,
            "invalidation": invalidation, "tp1": tp1, "tp2": tp2, "rr1": rr1}

# ============================ Chart ============================
def make_chart(hit, levels=None):
    df = hit["df"]
    x = np.arange(len(df))
    line = line_values(hit)
    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor="#11131a")
    ax.set_facecolor("#11131a")
    up = df["Close"] >= df["Open"]
    ax.vlines(x, df["Low"], df["High"], color=np.where(up, "#2ebd85", "#f6465d"), lw=0.7)
    ax.plot(x, line, color="#f0b90b", lw=1.8)
    for t in hit["touches"]:
        ax.plot(t, df["Low"].iloc[t], "o", color="#f0b90b", ms=5, mfc="none")
    if levels:
        ax.axhline(levels["invalidation"], color="#f6465d", ls="--", lw=1, alpha=0.8)
        ax.axhline(levels["tp1"], color="#2ebd85", ls="--", lw=1, alpha=0.8)
        ax.axhline(levels["tp2"], color="#2ebd85", ls=":", lw=1, alpha=0.6)
    if hit["log"]:
        ax.set_yscale("log")
    ax.tick_params(colors="#888", labelsize=7)
    for s in ax.spines.values():
        s.set_color("#333")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return buf.getvalue()

# ============================ Backtest ============================
def backtest_ticker(ticker, tf, params, hold_bars=12):
    """Historische Simulation: an jedem Punkt, an dem das Setup galt,
    Forward-Return & Linien-Halten messen."""
    cfg = TF_CFG[tf]
    try:
        df = yf.download(ticker, interval=cfg["interval"], period=cfg["period"],
                         progress=False, auto_adjust=True)
    except Exception:
        return []
    if df is None or len(df) < 120:
        return []
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    trades = []
    step = 2
    for end in range(80, len(df) - hold_bars, step):
        sub = df.iloc[:end]
        res = best_trendline(sub, params["min_bounce"], params["touch_tol"],
                             params["break_tol"], strong_mode=params["strong"],
                             log_scale=False)
        if res is None:
            continue
        slope, intercept, touches, _ = res
        n = end - 1
        line_now = slope * n + intercept
        price = float(sub["Close"].iloc[-1])
        dist = (price - line_now) / line_now * 100
        if dist < -params["break_tol"] or dist > params["max_dist"]:
            continue
        inval = line_now * (1 - params["break_tol"] / 100)
        fwd = df.iloc[end:end + hold_bars]
        max_gain = float(fwd["High"].max() / price - 1) * 100
        ret = float(fwd["Close"].iloc[-1] / price - 1) * 100
        broke = bool((fwd["Close"] < inval).any())
        trades.append({"Ticker": ticker, "Datum": str(df.index[end - 1].date()),
                       "Einstieg": round(price, 2), "Max. Gewinn %": round(max_gain, 1),
                       "Return %": round(ret, 1), "Linie gebrochen": broke})
        end += hold_bars  # keine überlappenden Trades
    return trades

# ============================ UI ============================
st.title("📉 Trendline Bounce Scanner Pro")

bull, spy_p, spy_sma = market_status()
if bull is True:
    st.success(f"🟢 Marktfilter: S&P 500 ({spy_p:.0f}) über SMA200 ({spy_sma:.0f}) – Bounce-freundliches Umfeld")
elif bull is False:
    st.error(f"🔴 Marktfilter: S&P 500 ({spy_p:.0f}) UNTER SMA200 ({spy_sma:.0f}) – Bounces brechen häufiger, Vorsicht!")

tab_scan, tab_bt = st.tabs(["🔍 Scanner", "📊 Backtest"])

params = {"max_dist": max_dist, "min_bounce": min_bounce,
          "touch_tol": touch_tol, "break_tol": break_tol, "min_price": min_price,
          "strong": scan_mode.startswith("Starke"),
          "need_confirm": need_confirm, "need_quality": need_quality,
          "use_log": use_log}

with tab_scan:
    for key, default in [("hits", []), ("universe", None), ("pos", 0), ("running", False)]:
        if key not in st.session_state:
            st.session_state[key] = default

    c_start, c_stop, c_reset = st.columns(3)
    if c_start.button("🚀 Scan starten / fortsetzen", type="primary", use_container_width=True):
        if st.session_state.universe is None:
            uni, errs = get_universe(universe_choice)
            st.session_state.universe = uni
            for e in errs:
                st.warning(f"Quelle fehlgeschlagen: {e}")
            if len(uni) < 50:
                st.error("Universum verdächtig klein – Listen-Abruf prüfen.")
            st.session_state.pos = 0
            st.session_state.hits = []
        st.session_state.running = True
    if c_stop.button("⏸️ Pause", use_container_width=True):
        st.session_state.running = False
    if c_reset.button("🔄 Zurücksetzen", use_container_width=True):
        st.session_state.universe = None
        st.session_state.pos = 0
        st.session_state.hits = []
        st.session_state.running = False

    uni = st.session_state.universe
    if uni is not None:
        total, done = len(uni), st.session_state.pos
        st.progress(done / total if total else 0.0,
                    text=f"Fortschritt: {done}/{total} Ticker · {len(st.session_state.hits)} Treffer bisher")

    if st.session_state.running and uni is not None and st.session_state.pos < len(uni):
        start = st.session_state.pos
        end = min(start + batch_size, len(uni))
        batch = uni[start:end]
        st.info(f"Scanne Charge {start + 1}–{end} von {len(uni)} …")
        bar = st.progress(0.0)
        status = st.empty()
        for k, ticker in enumerate(batch):
            for tf in timeframes:
                hit = scan_ticker(ticker, tf, params)
                if hit:
                    st.session_state.hits.append(hit)
                    status.success(f"Treffer: {ticker} ({tf}) {hit['dist']:+.1f}% | {hit['bounces']} Bounces")
            bar.progress((k + 1) / len(batch))
        st.session_state.pos = end
        st.session_state.hits.sort(key=lambda h: abs(h["dist"]))
        bar.empty()
        if st.session_state.pos >= len(uni):
            st.session_state.running = False
            st.success("✅ Scan komplett!")
            st.rerun()
        elif auto_cont:
            st.rerun()
        else:
            st.session_state.running = False
            st.rerun()

    hits = st.session_state.hits
    if hits:
        st.subheader(f"✅ {len(hits)} Treffer")
        table = pd.DataFrame([{
            "Ticker": h["ticker"], "TF": h["tf"],
            "Skala": "log" if h["log"] else "lin",
            "Kurs": round(h["price"], 2), "Distanz %": round(h["dist"], 2),
            "Bounces": h["bounces"],
            "Bestätigt": "✅" if h["confirmed"] else "—",
            "Konfluenz": len(h["confluence"])} for h in hits])
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button("📥 CSV", table.to_csv(index=False), "hits.csv", "text/csv",
                           use_container_width=True)
        st.divider()
        st.caption("⚠️ Keine Anlageberatung. Regelbasierte Einschätzung als Orientierung.")
        for h in hits:
            badge = "✅" if h["confirmed"] else "⏳"
            with st.expander(f"{badge} {h['ticker']} · {h['tf']} ({'log' if h['log'] else 'lin'}) — "
                             f"{h['dist']:+.1f}% · {h['bounces']} Bounces", expanded=False):
                fund = None
                if not h["ticker"].endswith("-USD"):
                    fund = get_fundamentals(h["ticker"])
                    fund["rel_strength"] = relative_strength(h["df"], fund.get("sector"))
                lv = technical_assessment(h, break_tol, fund=fund, mkt_bull=bull)
                st.image(make_chart(h, lv), use_container_width=True)
                st.markdown(f"### {lv['verdict']}  (Score {lv['score']:+d})")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Einstieg (akt.)", f"{h['price']:.2f}")
                c2.metric("❌ Invalidierung", f"{lv['invalidation']:.2f}",
                          f"{(lv['invalidation']/h['price']-1)*100:+.1f}%")
                c3.metric("🎯 TP1", f"{lv['tp1']:.2f}",
                          f"{(lv['tp1']/h['price']-1)*100:+.1f}%")
                c4.metric("🎯 TP2", f"{lv['tp2']:.2f}",
                          f"{(lv['tp2']/h['price']-1)*100:+.1f}%")
                if not math.isnan(lv["rr1"]):
                    st.caption(f"Chance/Risiko bis TP1: **{lv['rr1']:.1f} : 1** · "
                               f"Invalidiert bei {h['tf']}-Schluss unter {lv['invalidation']:.2f}")

                st.markdown("**Faktoren:**")
                for r in lv["reasons"]:
                    st.markdown(f"- {r}")

                if fund and (fund["target"] or fund["rec"]):
                    parts = []
                    if fund["target"]:
                        up = f" ({fund['upside']:+.0f}%)" if fund["upside"] is not None else ""
                        parts.append(f"Ø-Kursziel **{fund['target']:.2f}**{up}")
                    if fund["rec"]:
                        parts.append(f"Rating **{fund['rec'].replace('_',' ').title()}**"
                                     + (f" ({fund['rec_score']:.1f}/5)" if fund["rec_score"] else ""))
                    if fund["n_analysts"]:
                        parts.append(f"{fund['n_analysts']} Analysten")
                    if fund["earnings_days"] is not None:
                        parts.append(f"📅 Earnings in {fund['earnings_days']} Tagen")
                    st.markdown("**Analysten:** " + " · ".join(parts))
                    for n in fund["news"]:
                        st.caption(f"📰 {n}")
    else:
        st.write("Noch keine Ergebnisse. **Scan starten** drücken – Treffer erscheinen schon während des Scans.")

with tab_bt:
    st.markdown("Testet die Strategie historisch: Wäre an jedem früheren Signalpunkt "
                "ein Einstieg erfolgt, was wäre passiert? (Vereinfachtes Modell, lineare Linien)")
    bt_tickers = st.text_input("Ticker (kommagetrennt) – leer = aktuelle Treffer nutzen",
                               placeholder="NOW, CRM, PATH")
    bt_tf = st.selectbox("Timeframe", ["1W", "1M"], index=0)
    hold = st.slider("Haltedauer (Kerzen)", 4, 26, 12)
    if st.button("📊 Backtest starten", use_container_width=True):
        if bt_tickers.strip():
            tickers = [t.strip().upper() for t in bt_tickers.split(",") if t.strip()]
        else:
            tickers = sorted({h["ticker"] for h in st.session_state.get("hits", [])})[:20]
        if not tickers:
            st.warning("Keine Ticker angegeben und keine Scan-Treffer vorhanden.")
        else:
            all_trades = []
            bar = st.progress(0.0)
            for k, t in enumerate(tickers):
                all_trades += backtest_ticker(t, bt_tf, params, hold_bars=hold)
                bar.progress((k + 1) / len(tickers))
            bar.empty()
            if not all_trades:
                st.info("Keine historischen Signale mit diesen Parametern gefunden.")
            else:
                bt = pd.DataFrame(all_trades)
                wins = (bt["Return %"] > 0).mean() * 100
                held = (~bt["Linie gebrochen"]).mean() * 100
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Signale", len(bt))
                c2.metric("Trefferquote", f"{wins:.0f}%")
                c3.metric("Ø Return", f"{bt['Return %'].mean():+.1f}%")
                c4.metric("Linie hielt", f"{held:.0f}%")
                st.caption(f"Ø max. Gewinn innerhalb {hold} Kerzen: {bt['Max. Gewinn %'].mean():+.1f}% · "
                           "Vergangene Performance garantiert nichts für die Zukunft.")
                st.dataframe(bt, use_container_width=True, hide_index=True)
