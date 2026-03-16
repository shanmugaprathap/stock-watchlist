#!/usr/bin/env python3
"""Stock Watchlist API — FastAPI backend for Indian equity watchlist."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

app = FastAPI(title="Stock Watchlist")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

IST = pytz.timezone("Asia/Kolkata")

# ── Stock definitions ──────────────────────────────────────────────────────────

STOCKS = {
    "active": [
        {
            "symbol": "SILVERBEES", "name": "Silver ETF (SilverBees)",
            "entry": 248.99, "sl": 232, "t1": 262, "t2": 278,
            "qty": None, "avg_cost": 248.99,
        },
        {
            "symbol": "KARURVYSYA", "name": "Karur Vysya Bank",
            "entry": None, "sl": None, "t1": None, "t2": None,
            "qty": 122, "avg_cost": 90.07, "add_alert": 265,
        },
    ],
    "swing": [
        {"symbol": "JINDALSAW", "name": "Jindal SAW", "entry_low": 190, "entry_high": 195, "sl": 181, "t1": 215},
        {"symbol": "BPCL", "name": "BPCL", "entry_low": 320, "entry_high": 325, "sl": 305, "t1": 345},
        {"symbol": "VBL", "name": "Varun Beverages", "entry_low": None, "entry_high": None, "sl": None, "t1": None},
        {"symbol": "BHARTIARTL", "name": "Bharti Airtel", "entry_low": None, "entry_high": None, "sl": None, "t1": None},
        {"symbol": "MARUTI", "name": "Maruti Suzuki", "entry_low": None, "entry_high": None, "sl": None, "t1": None},
    ],
    "peg": [
        {"symbol": "HCLTECH", "name": "HCL Tech", "peg": 0.81},
        {"symbol": "COFORGE", "name": "Coforge", "peg": 0.60},
        {"symbol": "BPCL", "name": "BPCL", "peg": 0.40},
        {"symbol": "COALINDIA", "name": "Coal India", "peg": 0.58},
        {"symbol": "BSE", "name": "BSE Ltd", "peg": 0.86},
        {"symbol": "KARURVYSYA", "name": "Karur Vysya Bank", "peg": 0.50},
        {"symbol": "UJJIVANSFB", "name": "Ujjivan SFB", "peg": 0.40},
        {"symbol": "MTARTECH", "name": "MTAR Technologies", "peg": 0.76},
        {"symbol": "LUMAXIND", "name": "Lumax Industries", "peg": 1.13},
        {"symbol": "SCHNEIDER", "name": "Schneider Electric", "peg": 1.22},
    ],
}

# ── Calculation helpers ────────────────────────────────────────────────────────

def calc_rsi(closes: pd.Series, period: int = 14) -> float:
    """Compute RSI from closing prices using Wilder's smoothing."""
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean().iloc[period]
    avg_loss = loss.rolling(window=period, min_periods=period).mean().iloc[period]
    for i in range(period + 1, len(closes)):
        avg_gain = (avg_gain * (period - 1) + gain.iloc[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss.iloc[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def fetch_stock_data(symbol: str) -> dict | None:
    """Fetch price, history, and compute technicals for a single ticker."""
    ticker_str = f"{symbol}.NS"
    try:
        tk = yf.Ticker(ticker_str)
        hist = tk.history(period="1y")
        if hist.empty:
            return None

        closes = hist["Close"]
        current = float(closes.iloc[-1])
        prev_close = float(closes.iloc[-2]) if len(closes) > 1 else current
        change_pct = round(((current - prev_close) / prev_close) * 100, 2)

        high_52w = float(hist["High"].max())
        low_52w = float(hist["Low"].min())
        pct_from_high = round(((current - high_52w) / high_52w) * 100, 2)

        sma_50 = float(closes.tail(50).mean()) if len(closes) >= 50 else None
        sma_200 = float(closes.tail(200).mean()) if len(closes) >= 200 else None

        rsi = calc_rsi(closes) if len(closes) >= 20 else None

        return {
            "symbol": symbol,
            "ticker": ticker_str,
            "price": round(current, 2),
            "prev_close": round(prev_close, 2),
            "change_pct": change_pct,
            "high_52w": round(high_52w, 2),
            "low_52w": round(low_52w, 2),
            "pct_from_high": pct_from_high,
            "sma_50": round(sma_50, 2) if sma_50 else None,
            "sma_200": round(sma_200, 2) if sma_200 else None,
            "rsi": rsi,
        }
    except Exception as e:
        print(f"Error fetching {ticker_str}: {e}")
        return None


def entry_signal_swing(data: dict, entry_low: float | None, entry_high: float | None, sl: float | None) -> str:
    """Compute entry signal for swing/active stocks with defined entry zones."""
    price = data["price"]
    rsi = data["rsi"]
    sma_200 = data["sma_200"]
    sma_50 = data["sma_50"]
    pct_from_high = abs(data["pct_from_high"])

    if sl and price < sl:
        return "AVOID"

    if entry_low is None or entry_high is None:
        # No entry zone defined — use PEG-style logic
        return entry_signal_peg(data)

    entry_mid = (entry_low + entry_high) / 2
    at_or_below_entry = price <= entry_high * 1.02
    within_3pct = price <= entry_high * 1.03

    # STRONG BUY
    if rsi and sma_200:
        if at_or_below_entry and rsi < 45 and price < sma_200 and pct_from_high > 10:
            return "STRONG BUY"

    # GOOD TO ADD — any 2 of 4 conditions
    conditions = 0
    if rsi and 45 <= rsi <= 55:
        conditions += 1
    if within_3pct:
        conditions += 1
    if sma_200 and sma_50 and sma_200 <= price <= sma_50:
        conditions += 1
    if 5 <= pct_from_high <= 10:
        conditions += 1
    if conditions >= 2:
        return "GOOD TO ADD"

    if rsi and rsi > 60:
        return "WAIT"
    if price > entry_high:
        return "WAIT"

    return "WAIT"


def entry_signal_peg(data: dict) -> str:
    """Compute entry signal for PEG watchlist stocks (no entry zone)."""
    rsi = data["rsi"]
    sma_200 = data["sma_200"]
    price = data["price"]

    if rsi is None:
        return "WAIT"

    if rsi < 40 and sma_200 and price < sma_200:
        return "STRONG BUY"
    if 40 <= rsi <= 55:
        return "GOOD TO ADD"
    return "WAIT"


def fetch_nifty() -> dict | None:
    """Fetch Nifty 50 index data."""
    try:
        tk = yf.Ticker("^NSEI")
        hist = tk.history(period="1y")
        if hist.empty:
            return None
        closes = hist["Close"]
        current = float(closes.iloc[-1])
        prev = float(closes.iloc[-2]) if len(closes) > 1 else current
        sma_200 = float(closes.tail(200).mean()) if len(closes) >= 200 else None
        mood = "Neutral"
        if sma_200:
            if current > sma_200 * 1.02:
                mood = "Bullish"
            elif current < sma_200 * 0.98:
                mood = "Bearish"
        return {
            "level": round(current, 2),
            "change_pct": round(((current - prev) / prev) * 100, 2),
            "sma_200": round(sma_200, 2) if sma_200 else None,
            "mood": mood,
        }
    except Exception as e:
        print(f"Error fetching Nifty: {e}")
        return None


def is_market_open() -> bool:
    """Check if Indian stock market is currently open."""
    now = datetime.now(IST)
    if now.weekday() >= 5:  # Saturday/Sunday
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


# ── API endpoint ───────────────────────────────────────────────────────────────

@app.get("/api/stocks")
def get_stocks():
    now = datetime.now(IST)

    # Collect unique symbols to avoid duplicate fetches
    all_symbols = set()
    for group in STOCKS.values():
        for s in group:
            all_symbols.add(s["symbol"])
    all_symbols.add("^NSEI")

    # Fetch all data
    cache = {}
    for sym in all_symbols:
        if sym == "^NSEI":
            continue
        data = fetch_stock_data(sym)
        if data:
            cache[sym] = data

    # Build active positions
    active = []
    for s in STOCKS["active"]:
        data = cache.get(s["symbol"])
        if not data:
            continue
        entry_low = s.get("entry") or s.get("entry_low")
        entry_high = s.get("entry") or s.get("entry_high")
        signal = entry_signal_swing(data, entry_low, entry_high, s.get("sl"))

        item = {
            **data,
            "name": s["name"],
            "entry": s.get("entry"),
            "avg_cost": s.get("avg_cost"),
            "qty": s.get("qty"),
            "sl": s.get("sl"),
            "t1": s.get("t1"),
            "t2": s.get("t2"),
            "add_alert": s.get("add_alert"),
            "signal": signal,
        }
        # P&L calc
        if s.get("avg_cost") and s.get("qty"):
            invested = s["avg_cost"] * s["qty"]
            current_val = data["price"] * s["qty"]
            item["pnl_abs"] = round(current_val - invested, 2)
            item["pnl_pct"] = round(((current_val - invested) / invested) * 100, 2)
        elif s.get("avg_cost"):
            item["pnl_abs"] = round(data["price"] - s["avg_cost"], 2)
            item["pnl_pct"] = round(((data["price"] - s["avg_cost"]) / s["avg_cost"]) * 100, 2)

        active.append(item)

    # Build swing watchlist
    swing = []
    for s in STOCKS["swing"]:
        data = cache.get(s["symbol"])
        if not data:
            continue
        signal = entry_signal_swing(data, s.get("entry_low"), s.get("entry_high"), s.get("sl"))
        swing.append({
            **data,
            "name": s["name"],
            "entry_low": s.get("entry_low"),
            "entry_high": s.get("entry_high"),
            "sl": s.get("sl"),
            "t1": s.get("t1"),
            "signal": signal,
        })

    # Build PEG watchlist
    peg = []
    for s in STOCKS["peg"]:
        data = cache.get(s["symbol"])
        if not data:
            continue
        signal = entry_signal_peg(data)
        peg.append({
            **data,
            "name": s["name"],
            "peg": s["peg"],
            "signal": signal,
        })

    nifty = fetch_nifty()

    return {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S IST"),
        "market_open": is_market_open(),
        "nifty": nifty,
        "active": active,
        "swing": swing,
        "peg": peg,
    }


# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")
