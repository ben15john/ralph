# Top‑N Momentum Paper Trader (Streamlit)

This is a ready-to-deploy web app that runs a simple momentum rotation strategy (12-1 momentum, monthly rebalance) on a large-cap US stock universe and compares performance to SPY.

## How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## One‑click deploy (recommended)
### Streamlit Community Cloud
1. Push these files to a new **public GitHub repo**.
2. Go to https://share.streamlit.io/ and click **New app**.
3. Select your repo and `app.py`, then **Deploy**.

### Hugging Face Spaces
1. Create a new **Space** → type **Streamlit**.
2. Upload all files (or connect to your Git repo).
3. Click **Deploy**.

## What it does
- Downloads adjusted prices via `yfinance`.
- Builds a monthly rebalanced Top‑N portfolio by 12‑1 momentum.
- Includes simple fee/slippage model (bps).
- Shows performance summary, equity curve vs SPY, drawdowns, positions by month, and a trades log.
- Lets you download CSV outputs.

## Notes
- Educational use only. Not investment advice.
- You can extend this with Alpaca paper trading if you want to simulate live orders.