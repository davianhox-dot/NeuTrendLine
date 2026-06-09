"""
Trendline Bounce Scanner – Streamlit App
========================================
Mobile-freundliche Web-App: scannt Aktien & Kryptos nach langfristigen
Support-Trendlinien (1W & 1M), an denen der Kurs zuvor mehrfach gebounced
ist und denen er sich aktuell wieder naehert.

Lokal starten:
    pip install -r requirements.txt
    streamlit run app.py

Deployment (kostenlos, vom Handy erreichbar):
    1. app.py + requirements.txt in ein GitHub-Repo pushen
    2. share.streamlit.io -> "New app" -> Repo auswaehlen -> Deploy
    3. URL am Handy oeffnen -> "Zum Startbildschirm hinzufuegen"
"""

import itertools
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Trendline Scanner", page_icon="📉", layout="wide")

# ----------------------- Sidebar / Einstellungen -----------------------
st.sidebar.title("⚙️ Einstellungen")
max_dist   = st.sidebar.slider("Max. Distanz zur Linie (%)", 1.0, 15.0, 5.0, 0.5)
min_bounce = st.sidebar.slider("Min. Anzahl Bounces", 2, 6, 3)
touch_tol  = st.sidebar.slider("Touch-Toleranz (%)", 0.5, 5.0, 2.0, 0.5)
break_tol  = st.sidebar.slider("Bruch-Toleranz (%)", 0.5, 5.0, 1.5, 0.5)
min_price  = st.sidebar.number_input("Min. Kurs ($)", 0.0, 100.0, 1.0)

universe_choice = st.sidebar.selectbox(
    "Ticker-Universum",
    ["Schnell: S&P 500 + Krypto (~520)",
     "Mittel: + NASDAQ-100 + DAX (~650)",
     "Voll: Alle NASDAQ + NYSE + DE + Krypto (~8000, dauert Stunden!)"])

timeframes = st.sidebar.multiselect("Timeframes", ["1W", "1M"], default=["1W", "1M"])

PIVOT_WINDOW = 3
TF_CFG = {"1W": {"interval": "1wk", "period": "10y"},
          "1M": {"interval": "1mo", "period": "max"}}

# ----------------------- Universum -----------------------
@st.cache_data(ttl=86400, show_spinner=False)
def get_universe(choice: str):
    tickers = set()
    try:
        sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers.update(sp500["Symbol"].str.replace(".", "-", regex=False))
    except Exception:
        pass
    tickers.update(["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
                    "ADA-USD", "DOGE-USD", "AVAX-USD", "LINK-USD", "DOT-USD",
                    "LTC-USD", "ATOM-USD", "NEAR-USD", "ARB-USD", "INJ-USD"])
    if choice.startswith("Schnell"):
        return sorted(tickers)

    try:
        for t in pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100"):
            if "Ticker" in t.columns:
                tickers.update(t["Ticker"]); break
    except Exception:
        pass
    for page in ["DAX", "MDAX", "SDAX"]:
        try:
            for t in pd.read_html(f"https://en.wikipedia.org/wiki/{page}"):
                col = next((c for c in t.columns if "icker" in str(c) or "ymbol" in str(c)), None)
                if col is not None and len(t) > 20:
                    syms = t[col].dropna().astype(str)
                    tickers.update(s if s.endswith(".DE") else s + ".DE" for s in syms)
                    break
        except Exception:
            pass
    if choice.startswith("Mittel"):
        return sorted(tickers)

    # Voll: alle US-Listings
    try:
        nq = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", sep="|")
        nq = nq[(nq["Test Issue"] == "N") & (nq["ETF"] == "N")]
        tickers.update(nq["Symbol"].dropna())
    except Exception:
        pass
    try:
        ot = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", sep="|")
        ot = ot[(ot["Test Issue"] == "N") & (ot["ETF"] == "N")]
        syms = ot["ACT Symbol"].dropna()
        syms = syms[~syms.str.contains(r"[\$\.]")]
        tickers.update(syms.str.replace("/", "-"))
    except Exception:
        pass
    return sorted(t for t in tickers if isinstance(t, str) and t.strip())

# ----------------------- Trendlinien-Logik -----------------------
def find_pivot_lows(low: pd.Series, window: int):
    vals = low.values
    return [i for i in range(window, len(vals) - window)
            if vals[i] == vals[i - window:i + window + 1].min()]

def best_trendline(df, min_bounces, touch_tol_pct, break_tol_pct):
    lows, closes = df["Low"].values, df["Close"].values
    pivots = find_pivot_lows(df["Low"], PIVOT_WINDOW)
    if len(pivots) < 2:
        return None
    best = None
    for i, j in itertools.combinations(pivots, 2):
        if j - i < 10:
            continue
        slope = (lows[j] - lows[i]) / (j - i)
        intercept = lows[i] - slope * i
        line = slope * np.arange(len(df)) + intercept
        if (line <= 0).any():
            continue
        if (closes[i:] < line[i:] * (1 - break_tol_pct / 100)).any():
            continue
        touches = [p for p in pivots
                   if p >= i and abs(lows[p] - line[p]) / line[p] * 100 <= touch_tol_pct]
        if len(touches) < min_bounces:
            continue
        score = (len(touches), j - i)
        if best is None or score > best[0]:
            best = (score, slope, intercept, touches)
    return None if best is None else best[1:]

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
    df = df.dropna()
    if len(df) < 60 or float(df["Close"].iloc[-1]) < params["min_price"]:
        return None
    res = best_trendline(df, params["min_bounce"], params["touch_tol"], params["break_tol"])
    if res is None:
        return None
    slope, intercept, touches = res
    n = len(df) - 1
    line_now = slope * n + intercept
    price = float(df["Close"].iloc[-1])
    dist = (price - line_now) / line_now * 100
    if dist < -params["break_tol"] or dist > params["max_dist"]:
        return None
    return {"ticker": ticker, "tf": tf, "price": price, "line": float(line_now),
            "dist": float(dist), "bounces": len(touches),
            "df": df, "slope": slope, "intercept": intercept, "touches": touches}

# ----------------------- Analyse: Analysten, Technik, Levels -----------------------
@st.cache_data(ttl=43200, show_spinner=False)
def get_analyst_info(ticker):
    """Analystendaten + Sentiment-Proxy via yfinance."""
    out = {"target": None, "upside": None, "rec": None, "rec_score": None,
           "n_analysts": None, "news": []}
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        out["target"] = info.get("targetMeanPrice")
        out["rec"] = info.get("recommendationKey")          # buy / hold / sell ...
        out["rec_score"] = info.get("recommendationMean")    # 1 = strong buy, 5 = sell
        out["n_analysts"] = info.get("numberOfAnalystOpinions")
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if out["target"] and price:
            out["upside"] = (out["target"] / price - 1) * 100
        for n in (t.news or [])[:3]:
            title = n.get("title") or n.get("content", {}).get("title")
            if title:
                out["news"].append(title)
    except Exception:
        pass
    return out

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])

def technical_assessment(hit, break_tol_pct):
    """Regelbasierter Technik-Score + Levels. Keine Anlageberatung."""
    df = hit["df"]
    close = df["Close"]
    price = hit["price"]
    reasons, score = [], 0

    r = rsi(close)
    if not math.isnan(r):
        if r < 35:
            score += 2; reasons.append(f"RSI {r:.0f} – überverkauft (bullisch)")
        elif r < 50:
            score += 1; reasons.append(f"RSI {r:.0f} – neutral-schwach")
        elif r > 65:
            score -= 1; reasons.append(f"RSI {r:.0f} – bereits gelaufen")
        else:
            reasons.append(f"RSI {r:.0f} – neutral")

    sma200 = close.rolling(min(200, len(close) - 1)).mean().iloc[-1]
    if price > sma200:
        score += 1; reasons.append("Kurs über SMA200 (übergeordneter Aufwärtstrend)")
    else:
        score -= 1; reasons.append("Kurs unter SMA200 (übergeordnet schwach)")

    # MACD-Momentum
    ema12 = close.ewm(span=12).mean(); ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26; sig = macd.ewm(span=9).mean()
    if macd.iloc[-1] > sig.iloc[-1]:
        score += 1; reasons.append("MACD über Signallinie (Momentum dreht)")
    else:
        reasons.append("MACD unter Signallinie (Momentum noch negativ)")

    # Bounce-Qualität
    if hit["bounces"] >= 4:
        score += 1; reasons.append(f"{hit['bounces']} bestätigte Bounces – starke Linie")
    if abs(hit["dist"]) <= 2:
        score += 1; reasons.append("Sehr nah an der Linie – gutes Chance/Risiko")

    # Levels
    invalidation = hit["line"] * (1 - break_tol_pct / 100)
    highs = df["High"].values
    swing_highs = sorted({float(highs[i]) for i in range(3, len(highs) - 3)
                          if highs[i] == highs[i - 3:i + 4].max() and highs[i] > price * 1.02})
    tp1 = swing_highs[0] if swing_highs else price * 1.10
    tp2 = swing_highs[1] if len(swing_highs) > 1 else price * 1.20
    risk = price - invalidation
    rr1 = (tp1 - price) / risk if risk > 0 else float("nan")

    if score >= 3:
        verdict = "🟢 Eher kaufenswert (Setup spricht dafür)"
    elif score >= 1:
        verdict = "🟡 Neutral – auf Bounce-Bestätigung warten"
    else:
        verdict = "🔴 Eher nicht – Technik spricht dagegen"

    return {"score": score, "verdict": verdict, "reasons": reasons,
            "invalidation": invalidation, "tp1": tp1, "tp2": tp2, "rr1": rr1}

def make_chart(hit, levels=None):
    df, slope, b = hit["df"], hit["slope"], hit["intercept"]
    x = np.arange(len(df))
    line = slope * x + b
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
    ax.tick_params(colors="#888", labelsize=7)
    for s in ax.spines.values():
        s.set_color("#333")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return buf.getvalue()

# ----------------------- UI -----------------------
st.title("📉 Trendline Bounce Scanner")
st.caption("Findet Aktien & Kryptos nahe langfristiger Support-Trendlinien mit mehreren Bounces.")

if "hits" not in st.session_state:
    st.session_state.hits = []

if st.button("🚀 Scan starten", type="primary", use_container_width=True):
    params = {"max_dist": max_dist, "min_bounce": min_bounce,
              "touch_tol": touch_tol, "break_tol": break_tol, "min_price": min_price}
    universe = get_universe(universe_choice)
    st.info(f"Scanne {len(universe)} Ticker × {len(timeframes)} Timeframes …")
    progress = st.progress(0.0)
    status = st.empty()
    hits = []
    for k, ticker in enumerate(universe):
        for tf in timeframes:
            hit = scan_ticker(ticker, tf, params)
            if hit:
                hits.append(hit)
                status.success(f"Treffer: {ticker} ({tf}) {hit['dist']:+.1f}% | {hit['bounces']} Bounces")
        progress.progress((k + 1) / len(universe))
    hits.sort(key=lambda h: abs(h["dist"]))
    st.session_state.hits = hits
    progress.empty()

hits = st.session_state.hits
if hits:
    st.subheader(f"✅ {len(hits)} Treffer")
    table = pd.DataFrame([{
        "Ticker": h["ticker"], "TF": h["tf"], "Kurs": round(h["price"], 2),
        "Linie": round(h["line"], 2), "Distanz %": round(h["dist"], 2),
        "Bounces": h["bounces"]} for h in hits])
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button("📥 CSV herunterladen", table.to_csv(index=False),
                       "hits.csv", "text/csv", use_container_width=True)
    st.divider()
    st.caption("⚠️ Keine Anlageberatung. Regelbasierte technische Einschätzung + Analystendaten als Orientierung.")
    for h in hits:
        with st.expander(f"{h['ticker']} · {h['tf']} — {h['dist']:+.1f}% zur Linie · {h['bounces']} Bounces",
                         expanded=False):
            lv = technical_assessment(h, break_tol)
            st.image(make_chart(h, lv), use_container_width=True)
            st.markdown(f"### {lv['verdict']}")

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
                           f"Invalidiert bei {h['tf']}-Schlusskurs unter {lv['invalidation']:.2f}")

            st.markdown("**Technische Faktoren:**")
            for r in lv["reasons"]:
                st.markdown(f"- {r}")

            ai = get_analyst_info(h["ticker"])
            if ai["target"] or ai["rec"]:
                st.markdown("**Analysten & Sentiment:**")
                parts = []
                if ai["target"]:
                    up = f" ({ai['upside']:+.0f}% zum Kurs)" if ai["upside"] is not None else ""
                    parts.append(f"Ø-Kursziel **{ai['target']:.2f}**{up}")
                if ai["rec"]:
                    parts.append(f"Rating: **{ai['rec'].replace('_',' ').title()}**"
                                 + (f" ({ai['rec_score']:.1f}/5, 1=Strong Buy)" if ai["rec_score"] else ""))
                if ai["n_analysts"]:
                    parts.append(f"{ai['n_analysts']} Analysten")
                st.markdown(" · ".join(parts))
                for n in ai["news"]:
                    st.caption(f"📰 {n}")
            else:
                st.caption("Keine Analystendaten verfügbar (z. B. bei Kryptos).")
else:
    st.write("Noch keine Ergebnisse. Einstellungen links wählen und **Scan starten** drücken.")
