#!/usr/bin/env python3
"""Validate all tickers against yfinance before building the app."""

import yfinance as yf
import pandas as pd

TICKERS = [
    "SILVERBEES",
    "KARURVYSYA",
    "JINDALSAW",
    "BPCL",
    "VBL",
    "BHARTIARTL",
    "MARUTI",
    "HCLTECH",
    "COFORGE",
    "COALINDIA",
    "BSE",
    "UJJIVANSFB",
    "MTARTECH",
    "LUMAXIND",
    "SCHNEIDER",
    # Nifty 50 index
    "^NSEI",
]

def validate_ticker(symbol: str):
    """Test a ticker with .NS suffix, fallback to .BO if empty."""
    for suffix in [".NS", ".BO"]:
        if symbol.startswith("^"):
            ticker_str = symbol
        else:
            ticker_str = f"{symbol}{suffix}"

        print(f"\n{'='*60}")
        print(f"Testing: {ticker_str}")
        print(f"{'='*60}")

        try:
            tk = yf.Ticker(ticker_str)
            info = tk.info
            hist = tk.history(period="1y")

            if hist.empty:
                print(f"  ❌ No historical data for {ticker_str}")
                if symbol.startswith("^"):
                    return None, symbol
                continue

            current = hist["Close"].iloc[-1]
            high_52w = hist["High"].max()
            low_52w = hist["Low"].min()
            last_5 = hist["Close"].tail(5)

            print(f"  ✅ Current Price: ₹{current:.2f}")
            print(f"  52W High: ₹{high_52w:.2f}")
            print(f"  52W Low:  ₹{low_52w:.2f}")
            print(f"  % from 52W High: {((current - high_52w) / high_52w) * 100:.1f}%")
            print(f"  Last 5 closes:")
            for date, price in last_5.items():
                print(f"    {date.strftime('%Y-%m-%d')}: ₹{price:.2f}")
            print(f"  Data points (1Y): {len(hist)}")

            return ticker_str, suffix.replace(".", "") if not symbol.startswith("^") else symbol
        except Exception as e:
            print(f"  ❌ Error: {e}")
            if symbol.startswith("^"):
                return None, symbol
            continue

    print(f"  ❌❌ FAILED for both .NS and .BO — needs manual fix")
    return None, None


if __name__ == "__main__":
    results = {}
    failed = []

    for sym in TICKERS:
        ticker_str, exchange = validate_ticker(sym)
        if ticker_str:
            results[sym] = ticker_str
        else:
            failed.append(sym)

    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\n✅ Working tickers ({len(results)}):")
    for sym, full in results.items():
        print(f"  {sym:20s} → {full}")

    if failed:
        print(f"\n❌ Failed tickers ({len(failed)}):")
        for sym in failed:
            print(f"  {sym}")
    else:
        print("\n🎉 All tickers validated successfully!")
