#!/usr/bin/env python3
"""
📱 Stock Analyzer IDX - Telegram Bot
Full-featured stock analysis bot for Indonesian market (IDX)
Includes: Basic TA, Advanced Indicators, Market Intelligence, Divergence

Setup:
1. Install: pip install python-telegram-bot yfinance pandas numpy scipy requests
2. Get bot token from @BotFather on Telegram
3. Set TOKEN variable below
4. Run: python telegram_stock_bot.py
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf
from dotenv import load_dotenv



# ── Telegram ──────────────────────────────────────────────────────────────────
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, ContextTypes
)
from telegram.constants import ParseMode

# ── Bot Token ─────────────────────────────────────────────────────────────────
#TOKEN = "ganti ini"   # ← ganti dengan token dari @BotFather
# Mengambil data dari file .env
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')



# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ── IDX Stocks List ───────────────────────────────────────────────────────────
IDX_POPULAR = {
    "BBCA": "Bank Central Asia",
    "BBRI": "Bank Rakyat Indonesia",
    "BMRI": "Bank Mandiri",
    "TLKM": "Telkom Indonesia",
    "ASII": "Astra International",
    "GOTO": "GoTo Gojek Tokopedia",
    "BYAN": "Bayan Resources",
    "MDKA": "Merdeka Copper Gold",
    "ADRO": "Adaro Energy",
    "INDF": "Indofood Sukses Makmur",
    "UNVR": "Unilever Indonesia",
    "ICBP": "Indofood CBP",
    "KLBF": "Kalbe Farma",
    "HMSP": "HM Sampoerna",
    "EXCL":  "XL Axiata",
    "TOWR": "Sarana Menara Nusantara",
    "BBNI": "Bank Negara Indonesia",
    "BSDE": "Bumi Serpong Damai",
    "PTBA": "Bukit Asam",
    "CPIN": "Charoen Pokphand Indonesia",
}

# ─────────────────────────────────────────────────────────────────────────────
#  TECHNICAL ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def get_stock_data(ticker: str, period: str = "6mo") -> pd.DataFrame | None:
    """Fetch OHLCV from Yahoo Finance."""
    if not ticker.endswith(".JK"):
        ticker = ticker.upper() + ".JK"
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 30:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df
    except Exception as e:
        logger.error(f"Data fetch error {ticker}: {e}")
        return None


def calc_basic_indicators(df: pd.DataFrame) -> dict:
    """Calculate 8 basic technical indicators."""
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    result = {}

    # 1. RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss
    result["rsi"] = float(100 - 100 / (1 + rs.iloc[-1]))

    # 2. MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd  = ema12 - ema26
    signal_line = macd.ewm(span=9).mean()
    result["macd"]        = float(macd.iloc[-1])
    result["macd_signal"] = float(signal_line.iloc[-1])
    result["macd_hist"]   = float(macd.iloc[-1] - signal_line.iloc[-1])

    # 3. Bollinger Bands
    ma20  = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    result["bb_upper"] = float(ma20.iloc[-1] + 2 * std20.iloc[-1])
    result["bb_mid"]   = float(ma20.iloc[-1])
    result["bb_lower"] = float(ma20.iloc[-1] - 2 * std20.iloc[-1])

    # 4. Moving Averages
    result["ma5"]  = float(close.rolling(5).mean().iloc[-1])
    result["ma20"] = float(close.rolling(20).mean().iloc[-1])
    result["ma50"] = float(close.rolling(50).mean().iloc[-1])

    # 5. Stochastic
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    k      = 100 * (close - low14) / (high14 - low14 + 1e-9)
    result["stoch_k"] = float(k.iloc[-1])
    result["stoch_d"] = float(k.rolling(3).mean().iloc[-1])

    # 6. Volume analysis
    avg_vol = vol.rolling(20).mean().iloc[-1]
    result["vol_ratio"] = float(vol.iloc[-1] / avg_vol)

    # 7. ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    result["atr"] = float(tr.rolling(14).mean().iloc[-1])

    # 8. Momentum / ROC
    result["roc10"] = float(((close.iloc[-1] / close.iloc[-11]) - 1) * 100)

    result["current_price"] = float(close.iloc[-1])
    result["price_change"]  = float(close.pct_change().iloc[-1] * 100)
    return result


def calc_advanced_indicators(df: pd.DataFrame) -> dict:
    """Advanced indicators: Elliott Wave, Fibonacci, Ichimoku, VSA, SMC, VWAP, Harmonic, Market Profile."""
    try:
        from scipy import signal as scipy_signal
    except ImportError:
        return {}

    close = df["Close"].values
    high  = df["High"].values
    low   = df["Low"].values
    vol   = df["Volume"].values
    result = {}

    # ── Fibonacci Levels ──────────────────────────────────────────────────────
    recent = df.tail(100)
    swing_high = float(recent["High"].max())
    swing_low  = float(recent["Low"].min())
    diff = swing_high - swing_low
    current = float(df["Close"].iloc[-1])
    fib_618 = swing_high - 0.618 * diff
    fib_382 = swing_high - 0.382 * diff
    fib_786 = swing_high - 0.786 * diff

    fib_signal, fib_score = "HOLD", 0
    if abs(current - fib_618) < diff * 0.02:
        fib_signal, fib_score = "STRONG_BUY (0.618)", 2.5
    elif abs(current - fib_786) < diff * 0.02:
        fib_signal, fib_score = "STRONG_BUY (0.786)", 2.0
    elif abs(current - fib_382) < diff * 0.02:
        fib_signal, fib_score = "BUY (0.382)", 1.5
    elif current < swing_low * 1.02:
        fib_signal, fib_score = "EXTREME_BUY", 3.0
    elif current > swing_high * 0.98:
        fib_signal, fib_score = "TAKE_PROFIT", -1.5
    result["fib"] = {"signal": fib_signal, "score": fib_score,
                     "618": round(fib_618, 2), "382": round(fib_382, 2), "786": round(fib_786, 2)}

    # ── VWAP ──────────────────────────────────────────────────────────────────
    tp    = (df["High"] + df["Low"] + df["Close"]) / 3
    vwap  = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()
    vwap_val   = float(vwap.iloc[-1])
    pct_diff   = (current - vwap_val) / vwap_val * 100
    if pct_diff > 3:
        vwap_signal, vwap_score = "OVERBOUGHT", -1.0
    elif pct_diff > 0:
        vwap_signal, vwap_score = "ABOVE VWAP", 1.0
    elif pct_diff > -3:
        vwap_signal, vwap_score = "BELOW VWAP", -1.0
    else:
        vwap_signal, vwap_score = "OVERSOLD", 1.5
    result["vwap"] = {"value": round(vwap_val, 2), "signal": vwap_signal, "score": vwap_score}

    # ── Ichimoku ──────────────────────────────────────────────────────────────
    if len(df) >= 52:
        t9h  = df["High"].rolling(9).max();  t9l  = df["Low"].rolling(9).min()
        t26h = df["High"].rolling(26).max(); t26l = df["Low"].rolling(26).min()
        t52h = df["High"].rolling(52).max(); t52l = df["Low"].rolling(52).min()
        tenkan  = (t9h + t9l) / 2
        kijun   = (t26h + t26l) / 2
        span_a  = (tenkan + kijun) / 2
        span_b  = (t52h + t52l) / 2
        t_val   = float(tenkan.iloc[-1])
        k_val   = float(kijun.iloc[-1])
        sa_val  = float(span_a.iloc[-26]) if len(span_a) > 26 else float(span_a.iloc[-1])
        sb_val  = float(span_b.iloc[-26]) if len(span_b) > 26 else float(span_b.iloc[-1])
        cloud_top = max(sa_val, sb_val)
        cloud_bot = min(sa_val, sb_val)
        ichi_score = 0
        if current > cloud_top:
            ichi_signal = "BULLISH (Above Cloud)"
            ichi_score  = 2.5
        elif current < cloud_bot:
            ichi_signal = "BEARISH (Below Cloud)"
            ichi_score  = -2.5
        else:
            ichi_signal = "NEUTRAL (Inside Cloud)"
        if t_val > k_val:
            ichi_score += 0.5
        result["ichimoku"] = {"signal": ichi_signal, "score": ichi_score,
                              "tenkan": round(t_val, 2), "kijun": round(k_val, 2)}
    else:
        result["ichimoku"] = {"signal": "INSUFFICIENT DATA", "score": 0}

    # ── VSA (Volume Spread Analysis) ──────────────────────────────────────────
    spread   = df["High"].values - df["Low"].values
    avg_spread = np.mean(spread[-20:]) if len(spread) >= 20 else np.mean(spread)
    avg_vol20  = np.mean(vol[-20:])    if len(vol) >= 20 else np.mean(vol)
    last_spread = spread[-1]
    last_vol    = vol[-1]
    vsa_score   = 0
    if last_spread > avg_spread * 1.5 and last_vol > avg_vol20 * 1.5 and close[-1] > close[-2]:
        vsa_signal, vsa_score = "DEMAND_BAR (Bullish)", 2.0
    elif last_spread > avg_spread * 1.5 and last_vol > avg_vol20 * 1.5 and close[-1] < close[-2]:
        vsa_signal, vsa_score = "SUPPLY_BAR (Bearish)", -2.0
    elif last_spread < avg_spread * 0.5 and last_vol < avg_vol20 * 0.5:
        vsa_signal, vsa_score = "NO_DEMAND", -1.0
    else:
        vsa_signal = "NEUTRAL"
    result["vsa"] = {"signal": vsa_signal, "score": vsa_score}

    # ── SMC (Smart Money Concepts) ────────────────────────────────────────────
    smc_signals = []
    smc_score   = 0
    if len(df) >= 20:
        recent_high = float(df["High"].tail(20).iloc[:-1].max())
        recent_low  = float(df["Low"].tail(20).iloc[:-1].min())
        if current > recent_high:
            smc_signals.append("BULLISH_BOS"); smc_score += 1.5
        elif current < recent_low:
            smc_signals.append("BEARISH_BOS"); smc_score -= 1.5
        # Order Blocks (simplified)
        for i in range(-5, -1):
            if df["Close"].iloc[i] < df["Open"].iloc[i]:  # bearish candle
                if current <= df["High"].iloc[i] and current >= df["Low"].iloc[i]:
                    smc_signals.append("BEARISH_OB"); smc_score -= 1.5; break
            else:
                if current <= df["High"].iloc[i] and current >= df["Low"].iloc[i]:
                    smc_signals.append("BULLISH_OB"); smc_score += 1.5; break
    result["smc"] = {"signals": smc_signals if smc_signals else ["NEUTRAL"], "score": smc_score}

    # ── Market Profile (POC) ──────────────────────────────────────────────────
    if len(df) >= 50:
        rec  = df.tail(50)
        p_min, p_max = float(rec["Low"].min()), float(rec["High"].max())
        bins = np.linspace(p_min, p_max, 51)
        vp   = np.zeros(50)
        for _, row in rec.iterrows():
            lb = max(0, int(np.digitize(float(row["Low"]),  bins)) - 1)
            hb = min(49, int(np.digitize(float(row["High"]), bins)) - 1)
            n  = hb - lb + 1
            vp[lb:hb+1] += float(row["Volume"]) / n
        poc_idx = int(np.argmax(vp))
        poc     = float((bins[poc_idx] + bins[poc_idx+1]) / 2)
        if current < poc * 0.99:
            mp_signal, mp_score = "BELOW_POC (BUY zone)", 2.0
        elif current > poc * 1.01:
            mp_signal, mp_score = "ABOVE_POC (SELL zone)", -2.0
        else:
            mp_signal, mp_score = "AT_POC (Neutral)", 0
        result["market_profile"] = {"poc": round(poc, 2), "signal": mp_signal, "score": mp_score}

    # ── Elliott Wave (simplified) ─────────────────────────────────────────────
    try:
        prices = df["Close"].tail(100).values
        peaks,  _ = scipy_signal.find_peaks(prices, distance=5)
        troughs,_ = scipy_signal.find_peaks(-prices, distance=5)
        tps = sorted(
            [(p, "peak",   prices[p]) for p in peaks] +
            [(t, "trough", prices[t]) for t in troughs],
            key=lambda x: x[0]
        )
        ew_signal, ew_score = "NEUTRAL", 0
        if len(tps) >= 5:
            last = tps[-5:]
            if last[-1][1] == "peak" and last[-3][1] == "peak":
                w3 = last[-3][2] - last[-4][2]
                w5 = last[-1][2] - last[-2][2]
                if w3 > w5:
                    ew_signal, ew_score = "Wave 5 Top → SELL", -2.0
                else:
                    ew_signal, ew_score = "Wave 3 Strong → BUY", 3.0
            if last[-1][1] == "trough" and last[0][1] == "peak":
                ew_signal, ew_score = "Wave C Correction → BUY", 2.5
        result["elliott"] = {"signal": ew_signal, "score": ew_score}
    except Exception:
        result["elliott"] = {"signal": "ERROR", "score": 0}

    return result


def calc_divergences(df: pd.DataFrame) -> list:
    """Detect RSI & MACD divergences."""
    try:
        from scipy import signal as scipy_signal
    except ImportError:
        return []

    divergences = []
    if len(df) < 50:
        return divergences

    recent = df.tail(50).copy()
    prices = recent["Close"].values

    # RSI
    delta = pd.Series(prices).diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    rsi   = (100 - 100 / (1 + rs)).values

    for indicator, name in [(rsi, "RSI")]:
        p_peaks,  _ = scipy_signal.find_peaks(prices,    distance=5)
        p_troughs,_ = scipy_signal.find_peaks(-prices,   distance=5)
        i_peaks,  _ = scipy_signal.find_peaks(indicator, distance=5)
        i_troughs,_ = scipy_signal.find_peaks(-indicator, distance=5)

        def nearby(arr, idx, radius=5):
            return [x for x in arr if abs(x - idx) <= radius]

        # Regular Bullish
        for k in range(len(p_troughs) - 1):
            p1, p2 = p_troughs[k], p_troughs[k+1]
            i1l, i2l = nearby(i_troughs, p1), nearby(i_troughs, p2)
            if i1l and i2l and prices[p2] < prices[p1] and indicator[i2l[0]] > indicator[i1l[0]]:
                divergences.append({"type": f"🟢 Regular Bullish ({name})", "signal": "BUY", "score": 3.0})

        # Regular Bearish
        for k in range(len(p_peaks) - 1):
            p1, p2 = p_peaks[k], p_peaks[k+1]
            i1l, i2l = nearby(i_peaks, p1), nearby(i_peaks, p2)
            if i1l and i2l and prices[p2] > prices[p1] and indicator[i2l[0]] < indicator[i1l[0]]:
                divergences.append({"type": f"🔴 Regular Bearish ({name})", "signal": "SELL", "score": -3.0})

        # Hidden Bullish
        for k in range(len(p_troughs) - 1):
            p1, p2 = p_troughs[k], p_troughs[k+1]
            i1l, i2l = nearby(i_troughs, p1), nearby(i_troughs, p2)
            if i1l and i2l and prices[p2] > prices[p1] and indicator[i2l[0]] < indicator[i1l[0]]:
                divergences.append({"type": f"🟡 Hidden Bullish ({name})", "signal": "BUY", "score": 2.0})

    return divergences[:4]  # limit output


def calc_cup_and_handle(df: pd.DataFrame) -> dict | None:
    """Detect Cup and Handle pattern."""
    if len(df) < 60:
        return None
    try:
        from scipy import signal as scipy_signal
        recent = df.tail(120)
        close  = recent["Close"].values
        vol    = recent["Volume"].values
        n      = len(close)

        for cup_len in [60, 80, 100]:
            if n < cup_len + 10:
                continue
            for start in range(0, n - cup_len - 10, 10):
                cup = close[start: start + cup_len]
                rim_l, rim_r = cup[0], cup[-1]
                bot_idx = int(np.argmin(cup))
                bottom  = cup[bot_idx]
                depth   = (min(rim_l, rim_r) - bottom) / min(rim_l, rim_r) if min(rim_l, rim_r) > 0 else 0
                if not (0.10 <= depth <= 0.35):
                    continue
                rim_diff  = abs(rim_l - rim_r) / max(rim_l, rim_r)
                handle    = close[start + cup_len: start + cup_len + 10]
                if len(handle) < 5:
                    continue
                handle_depth = (max(cup[-5:]) - min(handle)) / max(cup[-5:]) if max(cup[-5:]) > 0 else 0
                vol_cup    = vol[start: start + cup_len]
                vol_handle = vol[start + cup_len: start + cup_len + 10]
                vol_contract = np.mean(vol_handle) < np.mean(vol_cup)
                score = 0
                if rim_diff < 0.03: score += 20
                elif rim_diff < 0.05: score += 10
                if 0.05 <= handle_depth <= 0.12: score += 15
                elif handle_depth <= 0.15: score += 10
                if vol_contract: score += 15
                if depth >= 0.10: score += 15
                resistance = max(rim_l, rim_r)
                dist = (resistance - close[-1]) / resistance * 100
                if dist < 2: score += 10
                elif dist < 5: score += 5
                if score >= 55:
                    return {
                        "found": True,
                        "score": score,
                        "cup_depth_pct": round(depth * 100, 1),
                        "handle_depth_pct": round(handle_depth * 100, 1),
                        "resistance": round(resistance, 2),
                        "dist_to_breakout_pct": round(dist, 2),
                        "vol_confirms": vol_contract,
                    }
    except Exception as e:
        logger.error(f"Cup&Handle error: {e}")
    return None


def generate_recommendation(basic: dict, advanced: dict, divergences: list, cup: dict | None) -> dict:
    """Aggregate all signals into final recommendation."""
    score = 0.0
    signals = []

    # ── Basic scoring ─────────────────────────────────────────────────────────
    rsi = basic.get("rsi", 50)
    if rsi < 30:   score += 2.5; signals.append(f"📉 RSI {rsi:.1f} → OVERSOLD (BUY)")
    elif rsi < 45: score += 1.0; signals.append(f"📊 RSI {rsi:.1f} → Approaching oversold")
    elif rsi > 70: score -= 2.5; signals.append(f"📈 RSI {rsi:.1f} → OVERBOUGHT (SELL)")
    elif rsi > 55: score -= 1.0; signals.append(f"📊 RSI {rsi:.1f} → Approaching overbought")

    hist = basic.get("macd_hist", 0)
    macd = basic.get("macd", 0)
    sig  = basic.get("macd_signal", 0)
    if hist > 0 and macd > sig: score += 1.5; signals.append("✅ MACD Bullish Crossover")
    elif hist < 0 and macd < sig: score -= 1.5; signals.append("❌ MACD Bearish Crossover")

    price = basic.get("current_price", 0)
    bb_l  = basic.get("bb_lower", 0)
    bb_u  = basic.get("bb_upper", 0)
    if price <= bb_l:  score += 2.0; signals.append("🟢 At Lower Bollinger Band (BUY)")
    elif price >= bb_u: score -= 2.0; signals.append("🔴 At Upper Bollinger Band (SELL)")

    ma5, ma20, ma50 = basic.get("ma5", 0), basic.get("ma20", 0), basic.get("ma50", 0)
    if ma5 > ma20 > ma50: score += 1.5; signals.append("📊 MA5 > MA20 > MA50 (Bullish trend)")
    elif ma5 < ma20 < ma50: score -= 1.5; signals.append("📊 MA5 < MA20 < MA50 (Bearish trend)")

    stoch_k = basic.get("stoch_k", 50)
    if stoch_k < 20: score += 1.5; signals.append(f"📉 Stochastic {stoch_k:.1f} → Oversold")
    elif stoch_k > 80: score -= 1.5; signals.append(f"📈 Stochastic {stoch_k:.1f} → Overbought")

    vol_r = basic.get("vol_ratio", 1)
    if vol_r > 2: signals.append(f"🔥 Volume surge {vol_r:.1f}x avg")

    # ── Advanced scoring ──────────────────────────────────────────────────────
    for key, label in [("fib","📐 Fibonacci"), ("ichimoku","☁️ Ichimoku"),
                       ("vsa","📊 VSA"), ("smc","💰 SMC"),
                       ("vwap","📍 VWAP"), ("elliott","🌊 Elliott Wave"),
                       ("market_profile","📦 Market Profile")]:
        if key in advanced:
            s = advanced[key].get("score", 0)
            sig_str = advanced[key].get("signal", advanced[key].get("signals", [""])[0])
            score += s
            if s != 0:
                signals.append(f"{label}: {sig_str}")

    # ── Divergences ───────────────────────────────────────────────────────────
    for d in divergences:
        score += d["score"]
        signals.append(f"Divergence: {d['type']}")

    # ── Cup & Handle ──────────────────────────────────────────────────────────
    if cup and cup.get("found"):
        score += 3.0
        signals.append(f"🏆 Cup & Handle (Breakout target: {cup['resistance']})")

    # ── Final recommendation ──────────────────────────────────────────────────
    max_score = 30
    pct = (score / max_score) * 100
    buy_c  = sum(1 for s in signals if "BUY" in s.upper() or "BULLISH" in s.upper() or "OVERSOLD" in s.upper())
    sell_c = sum(1 for s in signals if "SELL" in s.upper() or "BEARISH" in s.upper() or "OVERBOUGHT" in s.upper())
    total_s = buy_c + sell_c if buy_c + sell_c > 0 else 1
    confidence = max(buy_c, sell_c) / total_s * 100

    if pct >= 50 and confidence >= 70:  rec = "🚀 VERY STRONG BUY"
    elif pct >= 35:                      rec = "🚀 STRONG BUY"
    elif pct >= 20:                      rec = "📈 BUY"
    elif pct >= -15:                     rec = "⏸️ HOLD"
    elif pct >= -35:                     rec = "📉 SELL"
    elif pct <= -50 and confidence >= 70: rec = "🔻 VERY STRONG SELL"
    else:                                rec = "🔻 STRONG SELL"

    return {
        "recommendation": rec,
        "score_pct": round(pct, 1),
        "confidence": round(confidence, 1),
        "signals": signals[:12],
        "buy_count": buy_c,
        "sell_count": sell_c,
    }


def format_currency(n: float) -> str:
    if n >= 1_000:
        return f"Rp{n:,.0f}"
    return f"Rp{n:.4f}"


def format_volume(v: float) -> str:
    if v >= 1e9:  return f"{v/1e9:.2f}B"
    if v >= 1e6:  return f"{v/1e6:.2f}M"
    if v >= 1e3:  return f"{v/1e3:.1f}K"
    return str(int(v))


# ─────────────────────────────────────────────────────────────────────────────
#  BOT HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [
        [InlineKeyboardButton("📊 Analisa Saham",    callback_data="menu_analyze"),
         InlineKeyboardButton("🔥 Saham Populer IDX", callback_data="menu_popular")],
        [InlineKeyboardButton("📈 Top Gainers IDX",   callback_data="menu_gainers"),
         InlineKeyboardButton("❓ Cara Pakai",         callback_data="menu_help")],
    ]
    text = (
        "👋 *Selamat datang di Stock Analyzer IDX Bot!*\n\n"
        "Bot ini membantu analisa saham IDX langsung di HP kamu.\n\n"
        "🔬 *Fitur Tersedia:*\n"
        "• 8 Indikator Teknikal Dasar (RSI, MACD, BB, dll)\n"
        "• Advanced: Elliott Wave, Fibonacci, Ichimoku\n"
        "• Advanced: VWAP, VSA, SMC, Market Profile\n"
        "• Divergence Detection (RSI/MACD)\n"
        "• Cup & Handle Pattern\n"
        "• Rekomendasi BUY/HOLD/SELL otomatis\n\n"
        "💬 *Cara cepat:* Ketik kode saham, contoh: `BBCA`"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN,
                                    reply_markup=InlineKeyboardMarkup(kb))


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "📖 *Cara Menggunakan Bot*\n\n"
        "1️⃣ Ketik kode saham IDX langsung, misal:\n"
        "   `BBCA` atau `TLKM` atau `GOTO`\n\n"
        "2️⃣ Gunakan perintah:\n"
        "   /analisa BBCA — Analisa lengkap\n"
        "   /quick BBCA  — Ringkasan cepat\n"
        "   /populer     — Daftar saham populer\n\n"
        "3️⃣ Bot akan memberikan:\n"
        "   • Harga & perubahan hari ini\n"
        "   • Rekomendasi BUY/HOLD/SELL\n"
        "   • Sinyal dari 19+ indikator\n"
        "   • Confidence level analisa\n\n"
        "⚠️ *Disclaimer:* Analisa ini hanya untuk referensi. "
        "Selalu lakukan riset mandiri sebelum investasi."
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "menu_analyze":
        await query.message.reply_text(
            "💬 Ketik kode saham yang ingin dianalisa.\nContoh: `BBCA`, `TLKM`, `GOTO`",
            parse_mode=ParseMode.MARKDOWN)

    elif data == "menu_popular":
        kb = []
        stocks = list(IDX_POPULAR.items())
        for i in range(0, len(stocks), 3):
            row = [InlineKeyboardButton(f"{c}", callback_data=f"analyze_{c}")
                   for c, _ in stocks[i:i+3]]
            kb.append(row)
        kb.append([InlineKeyboardButton("🔙 Kembali", callback_data="menu_back")])
        await query.message.reply_text(
            "🔥 *Saham Populer IDX* — Pilih untuk analisa:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup(kb))

    elif data == "menu_gainers":
        await query.message.reply_text("⏳ Mengambil data top gainers IDX...", parse_mode=ParseMode.MARKDOWN)
        text = await get_top_movers()
        await query.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

    elif data == "menu_help":
        await help_cmd(query, context)

    elif data.startswith("analyze_"):
        ticker = data.replace("analyze_", "")
        await query.message.reply_text(f"⏳ Menganalisa *{ticker}*...", parse_mode=ParseMode.MARKDOWN)
        result_text = await run_analysis(ticker, mode="full")
        await query.message.reply_text(result_text, parse_mode=ParseMode.MARKDOWN)

    elif data == "menu_back":
        await start(query, context)


async def analyze_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("📝 Gunakan: `/analisa KODE` — contoh `/analisa BBCA`",
                                        parse_mode=ParseMode.MARKDOWN)
        return
    ticker = context.args[0].upper()
    msg    = await update.message.reply_text(f"⏳ Menganalisa *{ticker}*...", parse_mode=ParseMode.MARKDOWN)
    result = await run_analysis(ticker, mode="full")
    await msg.edit_text(result, parse_mode=ParseMode.MARKDOWN)


async def quick_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("📝 Gunakan: `/quick KODE` — contoh `/quick BBCA`",
                                        parse_mode=ParseMode.MARKDOWN)
        return
    ticker = context.args[0].upper()
    msg    = await update.message.reply_text(f"⏳ Quick scan *{ticker}*...", parse_mode=ParseMode.MARKDOWN)
    result = await run_analysis(ticker, mode="quick")
    await msg.edit_text(result, parse_mode=ParseMode.MARKDOWN)


async def populer_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = []
    stocks = list(IDX_POPULAR.items())
    for i in range(0, len(stocks), 3):
        row = [InlineKeyboardButton(f"{c}", callback_data=f"analyze_{c}")
               for c, _ in stocks[i:i+3]]
        kb.append(row)
    await update.message.reply_text(
        "🔥 *Saham Populer IDX* — Tap untuk analisa:",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=InlineKeyboardMarkup(kb))


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle plain text — if looks like a ticker, auto-analyze."""
    text = update.message.text.strip().upper()
    if 2 <= len(text) <= 6 and text.isalpha():
        msg = await update.message.reply_text(f"⏳ Menganalisa *{text}*...", parse_mode=ParseMode.MARKDOWN)
        result = await run_analysis(text, mode="full")
        await msg.edit_text(result, parse_mode=ParseMode.MARKDOWN)
    else:
        await update.message.reply_text(
            "💬 Ketik kode saham IDX (misal `BBCA`) atau gunakan /help",
            parse_mode=ParseMode.MARKDOWN)


# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS RUNNER
# ─────────────────────────────────────────────────────────────────────────────

async def run_analysis(ticker: str, mode: str = "full") -> str:
    """Run full analysis and return formatted Telegram message."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _analyze_sync, ticker, mode)
    return result


def _analyze_sync(ticker: str, mode: str) -> str:
    """Synchronous analysis (runs in thread pool)."""
    ticker_jk = ticker.upper() + ".JK" if not ticker.endswith(".JK") else ticker.upper()
    display   = ticker_jk.replace(".JK", "")

    df = get_stock_data(ticker_jk)
    if df is None:
        return (f"❌ *{display}* tidak ditemukan atau data tidak tersedia.\n\n"
                f"Pastikan kode saham IDX benar (contoh: BBCA, TLKM, GOTO).")

    basic    = calc_basic_indicators(df)
    price    = basic["current_price"]
    chg      = basic["price_change"]
    chg_icon = "📈" if chg >= 0 else "📉"

    # Company name lookup
    company  = IDX_POPULAR.get(display, "")
    name_str = f" — {company}" if company else ""

    header = (
        f"📊 *{display}{name_str}*\n"
        f"{'─'*32}\n"
        f"💰 Harga: *{format_currency(price)}*  {chg_icon} {chg:+.2f}%\n"
        f"📅 {datetime.now().strftime('%d %b %Y  %H:%M WIB')}\n"
    )

    if mode == "quick":
        rsi     = basic["rsi"]
        macd_h  = basic["macd_hist"]
        stoch_k = basic["stoch_k"]
        vol_r   = basic["vol_ratio"]
        # Quick signal
        quick_score = 0
        if rsi < 30:    quick_score += 3
        elif rsi > 70:  quick_score -= 3
        if macd_h > 0:  quick_score += 2
        else:           quick_score -= 2
        if stoch_k < 20: quick_score += 2
        elif stoch_k > 80: quick_score -= 2
        if quick_score >= 4:   qrec = "🚀 STRONG BUY"
        elif quick_score >= 2: qrec = "📈 BUY"
        elif quick_score <= -4: qrec = "🔻 STRONG SELL"
        elif quick_score <= -2: qrec = "📉 SELL"
        else:                   qrec = "⏸️ HOLD"

        return (
            header +
            f"\n🎯 *Quick Scan: {qrec}*\n\n"
            f"📊 RSI: `{rsi:.1f}` {'🔴 Overbought' if rsi>70 else '🟢 Oversold' if rsi<30 else '⚪ Normal'}\n"
            f"📈 MACD: `{'▲' if macd_h > 0 else '▼'} {macd_h:.4f}`\n"
            f"🎲 Stochastic: `{stoch_k:.1f}` {'🔴' if stoch_k>80 else '🟢' if stoch_k<20 else '⚪'}\n"
            f"📦 Volume: `{vol_r:.1f}x` rata-rata\n\n"
            f"💡 Gunakan /analisa {display} untuk analisa lengkap (19 indikator)"
        )

    # Full analysis
    advanced    = calc_advanced_indicators(df)
    divergences = calc_divergences(df)
    cup         = calc_cup_and_handle(df)
    rec_data    = generate_recommendation(basic, advanced, divergences, cup)

    rec   = rec_data["recommendation"]
    conf  = rec_data["confidence"]
    spct  = rec_data["score_pct"]
    sigs  = rec_data["signals"]
    buyc  = rec_data["buy_count"]
    selc  = rec_data["sell_count"]

    # Confidence bar
    bar_len   = 10
    filled    = int(conf / 100 * bar_len)
    conf_bar  = "█" * filled + "░" * (bar_len - filled)

    # Indicator summary
    rsi    = basic["rsi"]
    macd_h = basic["macd_hist"]
    stk    = basic["stoch_k"]
    bb_pos = "Upper" if price >= basic["bb_upper"] else "Lower" if price <= basic["bb_lower"] else "Middle"
    vol_r  = basic["vol_ratio"]
    vwap_v = advanced.get("vwap", {}).get("value", 0)
    poc    = advanced.get("market_profile", {}).get("poc", 0)
    ichi   = advanced.get("ichimoku", {}).get("signal", "N/A")

    # Signals block
    sigs_text = "\n".join(f"  • {s}" for s in sigs) if sigs else "  • Tidak ada sinyal dominan"

    # Divergences
    div_text = ""
    if divergences:
        div_lines = "\n".join(f"  • {d['type']}" for d in divergences)
        div_text  = f"\n🔀 *Divergence:*\n{div_lines}\n"

    # Cup & Handle
    cup_text = ""
    if cup and cup.get("found"):
        cup_text = (
            f"\n🏆 *Cup & Handle Terdeteksi!*\n"
            f"  Score: {cup['score']} | Cup depth: {cup['cup_depth_pct']}%\n"
            f"  Breakout level: {format_currency(cup['resistance'])} "
            f"({cup['dist_to_breakout_pct']:.1f}% lagi)\n"
            f"  Volume konfirmasi: {'✅ Ya' if cup['vol_confirms'] else '❌ Belum'}\n"
        )

    result = (
        header +
        f"\n{'═'*32}\n"
        f"🎯 *REKOMENDASI: {rec}*\n"
        f"📊 Confidence: `{conf_bar}` {conf:.0f}%\n"
        f"📐 Score: `{spct:+.1f}` | Buy: {buyc} | Sell: {selc}\n"
        f"{'═'*32}\n\n"
        f"📋 *Indikator Utama:*\n"
        f"  RSI(14): `{rsi:.1f}` {'🔴' if rsi>70 else '🟢' if rsi<30 else '⚪'}\n"
        f"  MACD: `{'▲' if macd_h>=0 else '▼'} {macd_h:.4f}`\n"
        f"  Bollinger: `{bb_pos} Band`\n"
        f"  Stochastic: `{stk:.1f}` {'🔴' if stk>80 else '🟢' if stk<20 else '⚪'}\n"
        f"  VWAP: `{format_currency(vwap_v)}`\n"
        f"  POC: `{format_currency(poc)}`\n"
        f"  Ichimoku: `{ichi}`\n"
        f"  Volume: `{vol_r:.1f}x` rata-rata\n"
        f"  ATR(14): `{format_currency(basic['atr'])}`\n\n"
        f"📡 *Sinyal Aktif ({len(sigs)}):*\n{sigs_text}\n"
        f"{div_text}"
        f"{cup_text}"
        f"\n{'─'*32}\n"
        f"⚠️ _Hanya untuk referensi. Bukan saran investasi._"
    )
    return result


async def get_top_movers() -> str:
    """Get quick price summary of popular IDX stocks."""
    lines  = ["📈 *Ringkasan Saham Populer IDX*\n"]
    loop   = asyncio.get_event_loop()

    async def fetch_one(code):
        df = await loop.run_in_executor(None, get_stock_data, code + ".JK", "5d")
        if df is not None and len(df) >= 2:
            p   = float(df["Close"].iloc[-1])
            chg = float(df["Close"].pct_change().iloc[-1] * 100)
            return code, p, chg
        return code, None, None

    tasks   = [fetch_one(c) for c in list(IDX_POPULAR.keys())[:12]]
    results = await asyncio.gather(*tasks)

    results = [(c, p, chg) for c, p, chg in results if p is not None]
    results.sort(key=lambda x: x[2], reverse=True)

    for code, price, chg in results:
        icon = "🟢" if chg >= 0 else "🔴"
        lines.append(f"{icon} *{code}*  {format_currency(price)}  `{chg:+.2f}%`")

    lines.append("\n_Tap kode saham di menu Populer untuk analisa lengkap_")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌ Set TOKEN dulu di baris TOKEN = '...'")
        print("   Dapatkan token dari @BotFather di Telegram")
        return

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start",   start))
    app.add_handler(CommandHandler("help",    help_cmd))
    app.add_handler(CommandHandler("analisa", analyze_cmd))
    app.add_handler(CommandHandler("quick",   quick_cmd))
    app.add_handler(CommandHandler("populer", populer_cmd))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    print("🤖 Stock Analyzer IDX Bot aktif...")
    print("📱 Buka Telegram dan cari bot kamu")
    print("⌨️  Ctrl+C untuk berhenti\n")
    app.run_polling(drop_pending_updates=True)


# Gunakan variabel TOKEN untuk menjalankan bot (Cara yang Benar)
if __name__ == '__main__':
    # Inisialisasi aplikasi
    application = Application.builder().token(TOKEN).build()
    
    # Tambahkan handler kamu di sini (contoh: CommandHandler)
    # application.add_handler(CommandHandler("start", start_function))
    
    # Jalankan bot
    application.run_polling()