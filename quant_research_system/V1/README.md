# V1 — Signal-Based Backtester

The first version of the quant research system. A simple daily signal backtester built around moving average crossovers.

---

## How It Works

1. **data_loader.py** — downloads OHLCV data via yFinance
2. **indicators.py** — computes moving averages, momentum, volatility
3. **strategies.py** — generates a binary signal (0 = flat, 1 = long) on MA crossover
4. **backtester.py** — converts signal to position (1-bar lag), computes equity curve vs buy-and-hold
5. **metrics.py** — calculates total return, annualised return, Sharpe ratio, max drawdown, win rate
6. **main.py** — runs the full pipeline and prints results

---

## Limitations

- No transaction costs or slippage
- Long-only (no short positions)
- Fixed 1-unit position sizing
- Sharpe ratio uses a configurable risk-free rate (default 0%)
- Hard-coded parameters in main.py

V1 exists as a learning baseline. V2 replaces it with a trade-level framework that handles realistic execution, multiple strategies, and Monte Carlo simulation.
