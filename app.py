#!/usr/bin/env python3
"""Stock Watchlist API — FastAPI backend for Indian equity watchlist."""

import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
import yfinance as yf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stock Watchlist")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

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

# ── In-memory cache ────────────────────────────────────────────────────────────

_cache = {"data": None, "timestamp": 0}
CACHE_TTL = 120  # seconds


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


def process_ticker(symbol: str, hist_df: pd.DataFrame) -> dict | None:
    """Process downloaded history for a single ticker into technicals."""
    try:
        if hist_df.empty:
            logger.warning(f"Empty history for {symbol}")
            return None

        closes = hist_df["Close"]
        highs = hist_df["High"]
        lows = hist_df["Low"]

        current = float(closes.iloc[-1])
        prev_close = float(closes.iloc[-2]) if len(closes) > 1 else current
        change_pct = round(((current - prev_close) / prev_close) * 100, 2)

        high_52w = float(highs.max())
        low_52w = float(lows.min())
        pct_from_high = round(((current - high_52w) / high_52w) * 100, 2)

        sma_50 = float(closes.tail(50).mean()) if len(closes) >= 50 else None
        sma_200 = float(closes.tail(200).mean()) if len(closes) >= 200 else None

        rsi = calc_rsi(closes) if len(closes) >= 20 else None

        return {
            "symbol": symbol,
            "ticker": f"{symbol}.NS",
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
        logger.error(f"Error processing {symbol}: {e}")
        return None


def fetch_all_data() -> dict:
    """Batch-download all tickers in one yf.download() call and compute technicals."""
    # Collect unique symbols
    all_symbols = set()
    for group in STOCKS.values():
        for s in group:
            all_symbols.add(s["symbol"])

    ns_tickers = [f"{sym}.NS" for sym in sorted(all_symbols)]
    nifty_ticker = "^NSEI"
    all_tickers = ns_tickers + [nifty_ticker]

    logger.info(f"Downloading {len(all_tickers)} tickers: {all_tickers}")

    try:
        raw = yf.download(
            tickers=all_tickers,
            period="1y",
            group_by="ticker",
            threads=True,
            progress=False,
        )
        logger.info(f"Download complete. Shape: {raw.shape}, Columns type: {type(raw.columns)}")
    except Exception as e:
        logger.error(f"yf.download failed: {e}")
        return {"cache": {}, "nifty": None}

    cache = {}
    nifty = None

    # Handle both single-ticker (flat columns) and multi-ticker (MultiIndex) results
    if isinstance(raw.columns, pd.MultiIndex):
        # Multi-ticker: columns are (TICKER, OHLCV)
        for ticker_str in all_tickers:
            try:
                ticker_data = raw[ticker_str].dropna(how="all")
                if ticker_data.empty:
                    logger.warning(f"No data for {ticker_str} in batch download")
                    continue

                if ticker_str == nifty_ticker:
                    nifty = _process_nifty(ticker_data)
                else:
                    symbol = ticker_str.replace(".NS", "")
                    result = process_ticker(symbol, ticker_data)
                    if result:
                        cache[symbol] = result
            except KeyError:
                logger.warning(f"Ticker {ticker_str} not found in download result")
            except Exception as e:
                logger.error(f"Error extracting {ticker_str}: {e}")
    else:
        # Single ticker edge case — shouldn't happen with 16+ tickers
        logger.warning("Got flat columns instead of MultiIndex — single ticker mode")

    logger.info(f"Processed {len(cache)} stock tickers, nifty={'OK' if nifty else 'FAILED'}")
    return {"cache": cache, "nifty": nifty}


def _process_nifty(hist: pd.DataFrame) -> dict | None:
    """Process Nifty 50 data from batch download."""
    try:
        closes = hist["Close"].dropna()
        if closes.empty:
            return None
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
        logger.error(f"Error processing Nifty: {e}")
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
        return entry_signal_peg(data)

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


def is_market_open() -> bool:
    """Check if Indian stock market is currently open."""
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


# ── API endpoint ───────────────────────────────────────────────────────────────

@app.get("/api/stocks")
def get_stocks():
    global _cache
    now = datetime.now(IST)

    # Return cached data if fresh enough
    if _cache["data"] and (time.time() - _cache["timestamp"]) < CACHE_TTL:
        logger.info("Serving from cache")
        result = _cache["data"]
        result["timestamp"] = now.strftime("%Y-%m-%d %H:%M:%S IST")
        result["market_open"] = is_market_open()
        return result

    logger.info("Cache miss — fetching fresh data")
    fetched = fetch_all_data()
    stock_cache = fetched["cache"]
    nifty = fetched["nifty"]

    # Build active positions
    active = []
    for s in STOCKS["active"]:
        data = stock_cache.get(s["symbol"])
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
        data = stock_cache.get(s["symbol"])
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
        data = stock_cache.get(s["symbol"])
        if not data:
            continue
        signal = entry_signal_peg(data)
        peg.append({
            **data,
            "name": s["name"],
            "peg": s["peg"],
            "signal": signal,
        })

    result = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S IST"),
        "market_open": is_market_open(),
        "nifty": nifty,
        "active": active,
        "swing": swing,
        "peg": peg,
    }

    # Update cache
    _cache = {"data": result, "timestamp": time.time()}
    logger.info(f"Response: active={len(active)}, swing={len(swing)}, peg={len(peg)}")
    return result


# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")
