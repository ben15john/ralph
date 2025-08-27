
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="Top-N Momentum Paper Trader", layout="wide")

st.title("📈 Top‑N Momentum Paper Trader (Mock)")
st.caption("Monthly rotation among large‑cap US stocks using 12‑1 momentum. Backtest vs SPY.")

# ----- Sidebar controls -----
with st.sidebar:
    st.header("Settings")
    start_date = st.date_input("Backtest start date", value=datetime(2015,1,1))
    end_date = st.date_input("End date", value=datetime.today())
    initial_capital = st.number_input("Initial capital ($)", value=100_000, step=1_000, min_value=1_000)
    top_n = st.slider("Number of holdings (Top‑N)", min_value=3, max_value=20, value=10, step=1)
    rebalance_choice = st.selectbox("Rebalance", ["Monthly"], index=0)
    fee_bps = st.number_input("Fee per trade (bps)", value=5, min_value=0, max_value=100, step=1)
    slip_bps = st.number_input("Slippage per trade (bps)", value=5, min_value=0, max_value=100, step=1)
    run_btn = st.button("Run Backtest")

UNIVERSE = ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","BRK-B","AVGO","TSLA","LLY","JPM","V","COST","HD","MRK","XOM",
            "WMT","MA","ORCL","NFLX","PEP","KO","ABBV","ADBE","BAC","CSCO","PFE","CRM","NKE","TMO","ACN","WFC",
            "LIN","DHR","MCD","AMD","TXN","UNH","UPS","PM","NEE","AMAT","CVX","INTC","QCOM","AMGN","IBM","GE","CAT"]

def safe_download(tickers, start, end):
    # yfinance: use adjusted close for returns
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        close = data["Close"].to_frame()
    close = close.dropna(how="all")
    return close

@st.cache_data(show_spinner=True)
def get_data(start, end):
    prices = safe_download(UNIVERSE + ["SPY"], start, end)
    # Ensure SPY exists
    if "SPY" not in prices.columns:
        # Attempt to fetch individually
        spy = safe_download(["SPY"], start, end)
        prices = prices.join(spy["SPY"], how="outer")
    prices = prices.dropna(how="all")
    return prices

def month_ends(prices):
    # Business month end indices
    me = prices.index.to_period("M").to_timestamp("M")
    me = sorted(set(me).intersection(set(prices.index)))
    return pd.DatetimeIndex(me)

def compute_momentum(prices, lookback_days=252, gap_days=21):
    # 12-month (252d) minus last 1 month (21d) lookback
    # momentum = price(t-gap_days) / price(t-lookback_days) - 1
    mom = prices.shift(gap_days) / prices.shift(lookback_days) - 1.0
    return mom

def backtest(prices, init_capital, topn, fee_bps, slip_bps):
    prices = prices.dropna(how="all").copy()
    prices = prices.ffill()
    rebal_dates = month_ends(prices)
    rebal_dates = [d for d in rebal_dates if d in prices.index]

    mom = compute_momentum(prices)

    # Portfolio state
    equity_curve = pd.Series(index=prices.index, dtype=float)
    weights_hist = pd.DataFrame(0.0, index=rebal_dates, columns=prices.columns)
    pos_shares = pd.Series(0.0, index=prices.columns)  # current shares
    cash = init_capital
    portfolio_value = init_capital
    last_weights = pd.Series(0.0, index=prices.columns)

    trades = []  # dicts: date, symbol, action, shares, price, cost

    fee = fee_bps / 10_000.0
    slip = slip_bps / 10_000.0

    prev_date = prices.index[0]
    for i, current_date in enumerate(prices.index):
        px_today = prices.loc[current_date]

        # Mark-to-market
        m2m_value = (pos_shares * px_today).sum()
        portfolio_value = cash + m2m_value
        equity_curve.loc[current_date] = portfolio_value

        # Rebalance at month end, execute trades at next day's open (we'll approximate using today's close for UI simplicity)
        if current_date in rebal_dates:
            # Form signals using data up to current_date (month end)
            mom_today = mom.loc[current_date].dropna()
            mom_today = mom_today[mom_today.index != "SPY"]
            # Select top N
            top = mom_today.sort_values(ascending=False).head(topn).index.tolist()

            # Target equal weights for selected tickers
            target_weights = pd.Series(0.0, index=prices.columns)
            if len(top) > 0:
                w = 1.0 / len(top)
                target_weights.loc[top] = w

            # Compute dollar target per symbol
            target_value = target_weights * portfolio_value

            # Compute current value per symbol
            current_value = pos_shares * px_today

            # Desired change in value (dollar notional)
            dv = target_value - current_value

            # Apply slippage on executed price approximation
            exec_price = px_today * (1 + np.sign(dv) * slip)

            # Convert dv to shares delta
            d_shares = dv / exec_price
            d_shares = d_shares.fillna(0.0)

            # Trade cost (fees on traded notional)
            traded_notional = (abs(d_shares) * exec_price).sum()
            costs = traded_notional * fee

            # Update positions and cash
            pos_shares += d_shares
            cash -= (d_shares * exec_price).sum() + costs

            # Book trades
            for sym in prices.columns:
                if d_shares.get(sym, 0.0) != 0.0:
                    action = "BUY" if d_shares[sym] > 0 else "SELL"
                    trades.append({
                        "date": current_date,
                        "symbol": sym,
                        "action": action,
                        "shares": float(d_shares[sym]),
                        "price": float(exec_price[sym]) if not np.isnan(exec_price.get(sym, np.nan)) else np.nan,
                        "cost": float(costs * (abs(d_shares[sym]) * exec_price[sym]) / traded_notional) if traded_notional > 0 else 0.0
                    })

            # Store weights snapshot
            with np.errstate(divide='ignore', invalid='ignore'):
                new_value = pos_shares * px_today
                total_val = new_value.sum() + cash
                if total_val > 0:
                    weights_hist.loc[current_date] = new_value / (total_val - cash if (total_val - cash) != 0 else 1)

    # Compute benchmark equity
    spy = prices["SPY"].dropna()
    bench = (spy / spy.iloc[0]) * init_capital

    # Metrics
    eq = equity_curve.dropna()
    daily_ret = eq.pct_change().dropna()
    ann_factor = 252.0
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (252/len(eq)) - 1
    vol = daily_ret.std() * np.sqrt(ann_factor)
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(ann_factor) if daily_ret.std() > 0 else np.nan

    # Max drawdown
    roll_max = eq.cummax()
    drawdown = (eq - roll_max) / roll_max
    max_dd = drawdown.min()

    # Monthly stats
    m_eq = eq.resample("M").last()
    m_ret = m_eq.pct_change().dropna()
    win_rate = (m_ret > 0).mean()

    # Turn trades into DataFrame
    trades_df = pd.DataFrame(trades)
    positions_monthly = weights_hist.loc[weights_hist.index.intersection(m_eq.index)].copy()
    positions_monthly = positions_monthly.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return {
        "equity": eq,
        "benchmark": bench.reindex(eq.index).ffill(),
        "drawdown": drawdown.reindex(eq.index).fillna(0.0),
        "metrics": {
            "CAGR": cagr,
            "Volatility": vol,
            "Sharpe": sharpe,
            "Max Drawdown": max_dd,
            "Monthly Win Rate": win_rate
        },
        "trades": trades_df,
        "positions_monthly": positions_monthly
    }

if run_btn:
    st.info("Downloading data… this can take ~10–30 seconds the first time.")
    prices = get_data(start_date, end_date)

    # Run backtest
    results = backtest(prices, initial_capital, top_n, fee_bps, slip_bps)

    # --- Metrics ---
    st.subheader("Performance Summary")
    m = results["metrics"]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("CAGR", f"{m['CAGR']*100:.2f}%")
    col2.metric("Volatility", f"{m['Volatility']*100:.2f}%")
    col3.metric("Sharpe (rf=0)", f"{m['Sharpe']:.2f}" if pd.notna(m['Sharpe']) else "n/a")
    col4.metric("Max Drawdown", f"{m['Max Drawdown']*100:.2f}%")
    col5.metric("Monthly Win Rate", f"{m['Monthly Win Rate']*100:.1f}%")

    # --- Equity Curve ---
    st.subheader("Equity Curve vs SPY")
    fig1, ax1 = plt.subplots()
    ax1.plot(results["equity"].index, results["equity"].values, label="Strategy")
    ax1.plot(results["benchmark"].index, results["benchmark"].values, label="SPY")
    ax1.set_title("Equity Curve")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend()
    st.pyplot(fig1)

    # --- Drawdown ---
    st.subheader("Drawdown")
    fig2, ax2 = plt.subplots()
    ax2.plot(results["drawdown"].index, results["drawdown"].values)
    ax2.set_title("Strategy Drawdown")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Drawdown")
    st.pyplot(fig2)

    # --- Positions (Monthly) ---
    st.subheader("Positions by Month (weights)")
    st.dataframe(results["positions_monthly"].style.format("{:.2%}"))

    # --- Trades Log ---
    st.subheader("Trades Log")
    st.dataframe(results["trades"])

    # Downloads
    st.subheader("Download Outputs")
    eq_csv = results["equity"].rename("equity").to_csv().encode()
    bench_csv = results["benchmark"].rename("benchmark").to_csv().encode()
    dd_csv = results["drawdown"].rename("drawdown").to_csv().encode()
    trades_csv = results["trades"].to_csv(index=False).encode()
    pos_csv = results["positions_monthly"].to_csv().encode()

    st.download_button("Download equity.csv", eq_csv, file_name="equity.csv")
    st.download_button("Download benchmark.csv", bench_csv, file_name="benchmark.csv")
    st.download_button("Download drawdown.csv", dd_csv, file_name="drawdown.csv")
    st.download_button("Download trades.csv", trades_csv, file_name="trades.csv")
    st.download_button("Download positions_monthly.csv", pos_csv, file_name="positions_monthly.csv")

    st.success("Done. You can re-run with different settings from the sidebar.")
else:
    st.info("Set your options in the left sidebar and click **Run Backtest**.")
