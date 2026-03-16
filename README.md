# Stock Watchlist

Personal Indian equity watchlist dashboard with live NSE prices via Yahoo Finance.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Open http://localhost:8000

## Run with Docker

```bash
docker build -t stock-watchlist .
docker run -p 8000:8000 stock-watchlist
```

## Features

- Live prices from Yahoo Finance (yfinance) for NSE stocks
- RSI (14), 50/200 SMA calculations done server-side
- Entry signal badges: Strong Buy / Good to Add / Wait / Avoid
- Nifty 50 market mood indicator
- Auto-refresh every 60 seconds during market hours
- Dark theme, mobile responsive
- Unrealised P&L for active positions
