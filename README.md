# year1-quant

A modular quantitative trading research system built in Python, progressing from a simple signal-based backtester (V1) to a trade-level research engine (V2) with Monte Carlo simulation, multiple strategies, and a roadmap centred on ML signal generation using gradient boosting and walk-forward validation.

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
│   └── V2/                        # Trade-level research engine (current)
│       ├── core/
│       │   ├── trade_log.py       # Trade schema, validation, export
│       │   ├── performance.py     # R-based metrics (expectancy, profit factor, drawdown)
│       │   ├── plots.py           # Equity curve, drawdown, R-distribution
│       │   ├── monte_carlo.py     # Bootstrapped trade sequence simulation
│       │   ├── data_loader.py     # Data fetching (yFinance, Alpaca)
│       │   └── config.py          # Central configuration
│       │
│       ├── strategies/
│       │   ├── opening_range_breakout.py   # Stock ORB (yFinance/Alpaca/TradingView)
│       │   ├── futures_orb.py              # Futures ORB (ES, MES, NQ, MNQ)
│       │   └── key_level_orb.py            # 8AM key level strategy (overnight sessions)
│       │
│       └── main.py
│
└── ml_trading_project/
    └── v1_baseline_model.py       # Logistic regression baseline on AAPL
```

---

## Build Roadmap

| Version | Phase | Goal | Status |
|---------|-------|------|--------|
| V1 | — | Signal-based backtester (baseline) | Done |
| V2 | Phase 1 | Polish & solidify core — trade log, metrics, Monte Carlo, data provider | In progress |
| V3 | Phase 2 | Walk-forward validation framework — rolling windows, no lookahead, OOS tracking | Planned |
| V4 | Phase 3 | Gradient boosting signal generator — XGBoost/LightGBM, feature engineering, honest benchmarking | Planned |
| V5 | Phase 4 | Interactive dashboard — equity curve, MC bands, rolling Sharpe, feature importance | Planned |
| V6 | Phase 5 | Second ML layer — NLP on financial text / regime detection / RL execution (summer 2026) | Planned |

**Deferred:** Prop firm challenge simulator (parked — replaced by generic risk constraints in Phase 4). Live trading integration (after Phase 3 ships).

---

## Author

Ryan Raj — Oxford Engineering (AI focus)

---

## Disclaimer

Research and educational purposes only. Not financial advice.
