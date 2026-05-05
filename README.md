# year1-quant

A modular quantitative trading research system built in Python, progressing from a simple signal-based backtester (V1) through a trade-level research engine (V2) to a walk-forward validation framework (V3), with a roadmap centred on ML signal generation using gradient boosting.

---

## Project Structure

```
year1-quant/
├── quant_research_system/
│   ├── V1/                        # Signal-based backtester (moving average crossover)
│   │   ├── data_loader.py
│   │   ├── indicators.py
│   │   ├── strategies.py
│   │   ├── backtester.py
│   │   ├── metrics.py
│   │   ├── config.py
│   │   └── main.py
│   │
│   ├── V2/                        # Trade-level research engine (complete)
│   │   ├── core/
│   │   │   ├── trade_log.py       # Trade schema, validation, export
│   │   │   ├── performance.py     # R-based metrics (Sharpe, drawdown, recovery factor)
│   │   │   ├── plots.py           # Equity curve, drawdown, MC bands, streak chart
│   │   │   ├── monte_carlo.py     # Bootstrapped simulation, probability of ruin
│   │   │   ├── data_loader.py     # DataProvider ABC (yFinance + Databento)
│   │   │   └── config.py          # Central configuration
│   │   ├── strategies/
│   │   │   └── test_strategy.py   # Simple bullish candle strategy (pipeline test)
│   │   └── main.py
│   │
│   └── V3/                        # Walk-forward validation framework (current)
│       ├── core/                  # Inherited from V2 + walk_forward.py
│       ├── strategies/
│       └── main.py
│
└── ml_trading_project/
    └── v1_baseline_model.py       # Logistic regression baseline on AAPL
```

---

## Build Roadmap

| Version | Goal | Status |
|---------|------|--------|
| V1 | Signal-based backtester (baseline) | ✅ Done |
| V2 | Trade-level engine — trade log, R-metrics, Monte Carlo, data providers | ✅ Done |
| V3 | Walk-forward validation — rolling windows, no lookahead, OOS tracking | 🔨 Current |
| V4 | Gradient boosting signal generator — XGBoost/LightGBM, feature engineering | Planned |
| V5 | Interactive dashboard — equity curve, MC bands, rolling Sharpe, feature importance | Planned |
| V6 | Second ML layer — NLP / regime detection / RL execution (summer 2026) | Planned |

**Deferred:** Prop firm challenge simulator (parked — replaced by generic risk constraints in Phase 4). Live trading integration (after Phase 3 ships).

---

## Author

Ryan Raj — Oxford Engineering (AI focus)

---

## Disclaimer

Research and educational purposes only. Not financial advice.
