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

# ── Mutual Fund definitions ────────────────────────────────────────────────────

MUTUAL_FUNDS = [
    {"code": 145724, "name": "Tata Arbitrage Fund", "category": "Arbitrage",
     "invested": 1000000, "current_val": 1002000, "platform": "—"},
    {"code": 122639, "name": "Parag Parikh Flexi Cap", "category": "Flexicap",
     "invested": 157000, "current_val": 158000, "platform": "—"},
    {"code": 120828, "name": "Quant Small Cap", "category": "Small Cap",
     "invested": 109000, "current_val": 97650, "platform": "—"},
    {"code": 127042, "name": "Motilal Oswal Midcap", "category": "Midcap",
     "invested": 75000, "current_val": 61870, "platform": "—"},
    {"code": 118778, "name": "Nippon India Small Cap", "category": "Small Cap",
     "invested": 50000, "current_val": 46100, "platform": "—"},
    {"code": 119723, "name": "SBI ELSS Tax Saver", "category": "ELSS",
     "invested": 30000, "current_val": 28960, "platform": "—"},
    {"code": 147481, "name": "Parag Parikh ELSS Tax Saver", "category": "ELSS",
     "invested": 30000, "current_val": 28510, "platform": "—"},
]

# ── In-memory cache ────────────────────────────────────────────────────────────

_cache: dict = {"data": None, "timestamp": 0}
_mf_cache: dict = {"data": None, "timestamp": 0}
CACHE_TTL = 300  # 5 minutes — reduces Yahoo API calls
MF_CACHE_TTL = 600  # 10 minutes — MF NAVs update once a day


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


# ── Mutual Fund helpers ────────────────────────────────────────────────────────

AMFI_URL = "https://www.amfiindia.com/spages/NAVAll.txt"

def fetch_amfi_navs(codes: list[int]) -> dict[int, float]:
    """Fetch latest NAVs from AMFI for given scheme codes."""
    try:
        r = httpx.get(AMFI_URL, headers={"User-Agent": "Mozilla/5.0"},
                      timeout=30, follow_redirects=True)
        r.raise_for_status()
        nav_map = {}
        for line in r.text.split("\n"):
            parts = line.split(";")
            if len(parts) >= 5:
                try:
                    code = int(parts[0].strip())
                    if code in codes:
                        nav_map[code] = float(parts[4].strip())
                except (ValueError, IndexError):
                    continue
        logger.info(f"AMFI: fetched {len(nav_map)}/{len(codes)} NAVs")
        return nav_map
    except Exception as e:
        logger.error(f"AMFI fetch error: {e}")
        return {}


def get_nifty_technicals() -> dict | None:
    """Get Nifty RSI, SMA, and % from high for lumpsum signals."""
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=20) as client:
        chart = fetch_chart("^NSEI", client)
        if not chart:
            return None
        quote = chart.get("indicators", {}).get("quote", [{}])[0]
        closes = [c for c in quote.get("close", []) if c is not None]
        highs = [h for h in quote.get("high", []) if h is not None]
        if len(closes) < 20:
            return None
        closes_s = pd.Series(closes)
        current = closes[-1]
        high_52w = max(highs) if highs else current
        return {
            "level": round(current, 2),
            "rsi": calc_rsi(closes_s),
            "sma_50": round(float(closes_s.tail(50).mean()), 2) if len(closes_s) >= 50 else None,
            "sma_200": round(float(closes_s.tail(200).mean()), 2) if len(closes_s) >= 200 else None,
            "pct_from_high": round(((current - high_52w) / high_52w) * 100, 2),
        }


def lumpsum_signal(nifty: dict | None, category: str) -> tuple[str, str]:
    """Return (signal, reason) for lumpsum decision.

    Signals: DEPLOY NOW / GOOD ENTRY / SIP ONLY / WAIT
    """
    if not nifty or nifty.get("rsi") is None:
        return "SIP ONLY", "Market data unavailable"

    rsi = nifty["rsi"]
    sma_200 = nifty.get("sma_200")
    level = nifty["level"]
    pct_from_high = abs(nifty["pct_from_high"])

    # Arbitrage funds — always fine to deploy, they're market-neutral
    if category == "Arbitrage":
        return "DEPLOY NOW", "Arbitrage funds are market-neutral — safe anytime"

    # For equity funds, use Nifty technicals
    below_200sma = sma_200 and level < sma_200

    if rsi < 35 and below_200sma and pct_from_high > 5:
        return "DEPLOY NOW", f"Nifty oversold (RSI {rsi:.0f}), below 200 SMA, {pct_from_high:.0f}% off high"

    if rsi < 45 and (below_200sma or pct_from_high > 3):
        return "GOOD ENTRY", f"Nifty weak (RSI {rsi:.0f}), good dip-buy opportunity"

    if rsi <= 60:
        return "SIP ONLY", f"Nifty neutral (RSI {rsi:.0f}) — stick to SIP"

    return "WAIT", f"Nifty heated (RSI {rsi:.0f}) — avoid lumpsum, continue SIP"


def build_mf_response() -> dict:
    """Build mutual fund holdings with live NAVs and lumpsum signals."""
    codes = [f["code"] for f in MUTUAL_FUNDS]
    nav_map = fetch_amfi_navs(codes)
    nifty = get_nifty_technicals()

    funds = []
    total_invested = 0
    total_current = 0

    for f in MUTUAL_FUNDS:
        nav = nav_map.get(f["code"])
        invested = f["invested"]
        # Estimate units from initial current_val and initial NAV
        # On subsequent loads, recalculate current_val from live NAV
        if nav:
            # Units = initial_current_val / initial_nav (approximate)
            # We use invested/avg_nav where avg_nav = invested / (current_val/nav_at_snapshot)
            units = f["current_val"] / (f["current_val"] / (f["invested"] / f["invested"])) if nav else 0
            # Better: units = current_val_at_time / nav_at_time
            # Since we don't have exact units, derive from invested & approx avg_nav
            avg_nav = nav * (f["invested"] / f["current_val"])
            units = round(f["invested"] / avg_nav, 3)
            current_val = round(units * nav, 2)
        else:
            nav = 0
            units = 0
            current_val = f["current_val"]
            avg_nav = 0

        pnl_abs = round(current_val - invested, 2)
        pnl_pct = round(((current_val - invested) / invested) * 100, 2) if invested else 0

        signal, reason = lumpsum_signal(nifty, f["category"])

        total_invested += invested
        total_current += current_val

        funds.append({
            "name": f["name"],
            "category": f["category"],
            "platform": f["platform"],
            "code": f["code"],
            "nav": round(nav, 4) if nav else None,
            "units": units,
            "invested": invested,
            "current_val": round(current_val, 2),
            "pnl_abs": pnl_abs,
            "pnl_pct": pnl_pct,
            "weight_pct": 0,  # filled below
            "signal": signal,
            "signal_reason": reason,
        })

    # Calculate weights
    for f in funds:
        f["weight_pct"] = round((f["current_val"] / total_current) * 100, 1) if total_current else 0

    overall_pnl = round(((total_current - total_invested) / total_invested) * 100, 2) if total_invested else 0

    return {
        "funds": funds,
        "total_invested": total_invested,
        "total_current": round(total_current, 2),
        "overall_pnl_pct": overall_pnl,
        "nifty": nifty,
    }


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


@app.get("/api/mutualfunds")
def get_mutualfunds():
    global _mf_cache
    now = datetime.now(IST)

    if _mf_cache["data"] and (time.time() - _mf_cache["timestamp"]) < MF_CACHE_TTL:
        logger.info("MF: serving from cache")
        result = _mf_cache["data"].copy()
        result["timestamp"] = now.strftime("%Y-%m-%d %H:%M:%S IST")
        return result

    logger.info("MF: cache miss — fetching")
    result = build_mf_response()
    result["timestamp"] = now.strftime("%Y-%m-%d %H:%M:%S IST")

    if result["funds"]:
        _mf_cache = {"data": result, "timestamp": time.time()}
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
