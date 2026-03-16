#!/usr/bin/env python3
"""Stock Watchlist API — FastAPI backend for Indian equity watchlist."""

import logging
import time
from datetime import datetime

import httpx
import pandas as pd
import pytz
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

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

# ── Stock definitions ──────────────────────────────────────────────────────────

STOCKS = {
    "active": [
        {
            "symbol": "GOLDBEES", "name": "Gold ETF (GoldBees)",
            "entry": None, "sl": None, "t1": None, "t2": None,
            "qty": 307, "avg_cost": 73.27, "peg": None,  # ETF — no PEG
            "note": "Zerodha 205@71.11 + Dhan 102@77.61",
        },
        {
            "symbol": "SILVERBEES", "name": "Silver ETF (SilverBees)",
            "entry": 248.99, "sl": 232, "t1": 262, "t2": 278,
            "qty": None, "avg_cost": 248.99, "peg": None,  # ETF — no PEG
        },
        {
            "symbol": "SILVERCASE", "name": "Silver Case ETF (IndMoney)",
            "entry": None, "sl": None, "t1": None, "t2": None,
            "qty": 1000, "avg_cost": 15.30, "peg": None,  # ETF — no PEG
        },
        {
            "symbol": "KARURVYSYA", "name": "Karur Vysya Bank",
            "entry": None, "sl": None, "t1": None, "t2": None,
            "qty": 122, "avg_cost": 90.07, "add_alert": 265, "peg": 0.50,
        },
    ],
    "swing": [
        {"symbol": "JINDALSAW", "name": "Jindal SAW", "entry_low": 190, "entry_high": 195, "sl": 181, "t1": 215, "peg": 0.55},
        {"symbol": "VBL", "name": "Varun Beverages", "entry_low": None, "entry_high": None, "sl": None, "t1": None, "peg": 1.96},
        {"symbol": "BHARTIARTL", "name": "Bharti Airtel", "entry_low": None, "entry_high": None, "sl": None, "t1": None, "peg": 0.60},
        {"symbol": "MARUTI", "name": "Maruti Suzuki", "entry_low": None, "entry_high": None, "sl": None, "t1": None, "peg": 1.30},
    ],
    "peg": [
        {"symbol": "HCLTECH", "name": "HCL Tech", "peg": 0.81},
        {"symbol": "COFORGE", "name": "Coforge", "peg": 0.60},
        {"symbol": "BPCL", "name": "BPCL", "peg": 0.40},
        {"symbol": "COALINDIA", "name": "Coal India", "peg": 0.58},
        {"symbol": "BSE", "name": "BSE Ltd", "peg": 0.76},
        {"symbol": "KARURVYSYA", "name": "Karur Vysya Bank", "peg": 0.50},
        {"symbol": "MTARTECH", "name": "MTAR Technologies", "peg": 0.76},
        {"symbol": "LUMAXIND", "name": "Lumax Industries", "peg": 1.13},
        {"symbol": "SCHNEIDER", "name": "Schneider Electric", "peg": 1.22},
        {"symbol": "HAL", "name": "Hindustan Aeronautics", "peg": 2.35},
        {"symbol": "BEL", "name": "Bharat Electronics", "peg": 1.73},
    ],
}

# ── In-memory cache ────────────────────────────────────────────────────────────

_cache: dict = {"data": None, "timestamp": 0}
CACHE_TTL = 300  # 5 minutes — reduces Yahoo API calls


# ── Yahoo Finance direct HTTP fetcher with retry ──────────────────────────────

def fetch_chart(ticker: str, client: httpx.Client) -> dict | None:
    """Fetch 1Y daily OHLCV with retry across query1/query2 endpoints."""
    params = {"range": "1y", "interval": "1d", "includePrePost": "false"}

    for base in [
        "https://query1.finance.yahoo.com/v8/finance/chart",
        "https://query2.finance.yahoo.com/v8/finance/chart",
    ]:
        url = f"{base}/{ticker}"
        for attempt in range(3):
            try:
                r = client.get(url, params=params)
                if r.status_code == 429:
                    wait = (attempt + 1) * 2
                    logger.warning(f"429 for {ticker} on {base}, waiting {wait}s (attempt {attempt+1})")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                data = r.json()
                result = data.get("chart", {}).get("result")
                if result:
                    return result[0]
                logger.warning(f"No chart result for {ticker}")
                return None
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    time.sleep((attempt + 1) * 2)
                    continue
                logger.error(f"HTTP error for {ticker}: {e}")
                break
            except Exception as e:
                logger.error(f"Error for {ticker}: {e}")
                break
    return None


def chart_to_technicals(symbol: str, chart_data: dict) -> dict | None:
    """Convert Yahoo chart API response to technicals dict."""
    try:
        quote = chart_data.get("indicators", {}).get("quote", [{}])[0]
        closes_raw = quote.get("close", [])
        highs_raw = quote.get("high", [])
        lows_raw = quote.get("low", [])

        valid = [(c, h, l) for c, h, l in zip(closes_raw, highs_raw, lows_raw)
                 if c is not None and h is not None and l is not None]
        if len(valid) < 20:
            return None

        closes = [v[0] for v in valid]
        highs = [v[1] for v in valid]
        lows = [v[2] for v in valid]

        closes_s = pd.Series(closes)
        current = closes[-1]
        prev_close = closes[-2] if len(closes) > 1 else current

        return {
            "symbol": symbol,
            "ticker": f"{symbol}.NS",
            "price": round(current, 2),
            "prev_close": round(prev_close, 2),
            "change_pct": round(((current - prev_close) / prev_close) * 100, 2),
            "high_52w": round(max(highs), 2),
            "low_52w": round(min(lows), 2),
            "pct_from_high": round(((current - max(highs)) / max(highs)) * 100, 2),
            "sma_50": round(float(closes_s.tail(50).mean()), 2) if len(closes_s) >= 50 else None,
            "sma_200": round(float(closes_s.tail(200).mean()), 2) if len(closes_s) >= 200 else None,
            "rsi": calc_rsi(closes_s) if len(closes_s) >= 20 else None,
        }
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        return None


def calc_rsi(closes: pd.Series, period: int = 14) -> float:
    """Compute RSI using Wilder's smoothing."""
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
    return round(100 - (100 / (1 + avg_gain / avg_loss)), 2)


def fetch_all_data() -> dict:
    """Fetch all tickers with delays between requests."""
    all_symbols = set()
    for group in STOCKS.values():
        for s in group:
            all_symbols.add(s["symbol"])

    tickers = [(f"{sym}.NS", sym) for sym in sorted(all_symbols)]
    tickers.append(("^NSEI", None))

    logger.info(f"Fetching {len(tickers)} tickers")

    stock_cache = {}
    nifty = None

    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=20) as client:
        for i, (ticker, symbol) in enumerate(tickers):
            # Small delay between requests to avoid rate limiting
            if i > 0:
                time.sleep(0.5)

            chart_data = fetch_chart(ticker, client)
            if not chart_data:
                continue

            if ticker == "^NSEI":
                nifty = _process_nifty_chart(chart_data)
            elif symbol:
                result = chart_to_technicals(symbol, chart_data)
                if result:
                    stock_cache[symbol] = result

    logger.info(f"Fetched {len(stock_cache)}/{len(tickers)-1} stocks, nifty={'OK' if nifty else 'FAILED'}")
    return {"cache": stock_cache, "nifty": nifty}


def _process_nifty_chart(chart_data: dict) -> dict | None:
    try:
        quote = chart_data.get("indicators", {}).get("quote", [{}])[0]
        closes = [c for c in quote.get("close", []) if c is not None]
        if len(closes) < 2:
            return None
        current, prev = closes[-1], closes[-2]
        closes_s = pd.Series(closes)
        sma_200 = float(closes_s.tail(200).mean()) if len(closes_s) >= 200 else None
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
        logger.error(f"Nifty error: {e}")
        return None


# ── Signal logic ───────────────────────────────────────────────────────────────

def entry_signal_swing(data, entry_low, entry_high, sl):
    price, rsi, sma_200, sma_50 = data["price"], data["rsi"], data["sma_200"], data["sma_50"]
    pct_from_high = abs(data["pct_from_high"])

    if sl and price < sl:
        return "AVOID"
    if entry_low is None or entry_high is None:
        return entry_signal_peg(data)

    if rsi and sma_200 and price <= entry_high * 1.02 and rsi < 45 and price < sma_200 and pct_from_high > 10:
        return "STRONG BUY"

    conditions = sum([
        bool(rsi and 45 <= rsi <= 55),
        price <= entry_high * 1.03,
        bool(sma_200 and sma_50 and sma_200 <= price <= sma_50),
        5 <= pct_from_high <= 10,
    ])
    if conditions >= 2:
        return "GOOD TO ADD"
    return "WAIT"


def entry_signal_peg(data):
    rsi, sma_200, price = data["rsi"], data["sma_200"], data["price"]
    if rsi is None:
        return "WAIT"
    if rsi < 40 and sma_200 and price < sma_200:
        return "STRONG BUY"
    if 40 <= rsi <= 55:
        return "GOOD TO ADD"
    return "WAIT"


def is_market_open():
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    return now.replace(hour=9, minute=15, second=0) <= now <= now.replace(hour=15, minute=30, second=0)


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/stocks")
def get_stocks():
    global _cache
    now = datetime.now(IST)

    if _cache["data"] and (time.time() - _cache["timestamp"]) < CACHE_TTL:
        logger.info("Serving from cache")
        result = _cache["data"].copy()
        result["timestamp"] = now.strftime("%Y-%m-%d %H:%M:%S IST")
        result["market_open"] = is_market_open()
        return result

    logger.info("Cache miss — fetching fresh data")
    fetched = fetch_all_data()
    sc = fetched["cache"]

    active = []
    for s in STOCKS["active"]:
        data = sc.get(s["symbol"])
        if not data:
            continue
        signal = entry_signal_swing(data, s.get("entry") or s.get("entry_low"),
                                     s.get("entry") or s.get("entry_high"), s.get("sl"))
        item = {**data, "name": s["name"], "entry": s.get("entry"),
                "avg_cost": s.get("avg_cost"), "qty": s.get("qty"),
                "sl": s.get("sl"), "t1": s.get("t1"), "t2": s.get("t2"),
                "add_alert": s.get("add_alert"), "peg": s.get("peg"),
                "note": s.get("note"), "signal": signal}
        if s.get("avg_cost") and s.get("qty"):
            inv = s["avg_cost"] * s["qty"]
            cur = data["price"] * s["qty"]
            item["pnl_abs"] = round(cur - inv, 2)
            item["pnl_pct"] = round(((cur - inv) / inv) * 100, 2)
        elif s.get("avg_cost"):
            item["pnl_abs"] = round(data["price"] - s["avg_cost"], 2)
            item["pnl_pct"] = round(((data["price"] - s["avg_cost"]) / s["avg_cost"]) * 100, 2)
        active.append(item)

    swing = []
    for s in STOCKS["swing"]:
        data = sc.get(s["symbol"])
        if not data:
            continue
        signal = entry_signal_swing(data, s.get("entry_low"), s.get("entry_high"), s.get("sl"))
        swing.append({**data, "name": s["name"], "entry_low": s.get("entry_low"),
                       "entry_high": s.get("entry_high"), "sl": s.get("sl"),
                       "t1": s.get("t1"), "peg": s.get("peg"), "signal": signal})

    peg = []
    for s in STOCKS["peg"]:
        data = sc.get(s["symbol"])
        if not data:
            continue
        peg.append({**data, "name": s["name"], "peg": s["peg"],
                     "signal": entry_signal_peg(data)})

    result = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S IST"),
        "market_open": is_market_open(),
        "nifty": fetched["nifty"],
        "active": active, "swing": swing, "peg": peg,
    }

    # Only cache if we got actual data
    if active or swing or peg:
        _cache = {"data": result, "timestamp": time.time()}
    else:
        logger.warning("Empty result — not caching")
    logger.info(f"Response: active={len(active)}, swing={len(swing)}, peg={len(peg)}")
    return result


@app.get("/api/debug")
def debug():
    """Connectivity test with single ticker."""
    results = {"method": "query1 chart API", "time": datetime.now(IST).isoformat()}
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=20) as client:
        for ticker in ["BPCL.NS", "^NSEI"]:
            try:
                chart = fetch_chart(ticker, client)
                if chart:
                    quote = chart.get("indicators", {}).get("quote", [{}])[0]
                    closes = [c for c in quote.get("close", []) if c is not None]
                    results[ticker] = {"status": "OK", "points": len(closes),
                                       "last": round(closes[-1], 2) if closes else None}
                else:
                    results[ticker] = {"status": "NO_DATA"}
            except Exception as e:
                results[ticker] = {"status": "ERROR", "msg": str(e)}
    return results


# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")
