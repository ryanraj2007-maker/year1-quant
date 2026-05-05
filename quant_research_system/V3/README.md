Quant Research System — V3 (Walk-Forward Validation Framework)

Built on top of V2. Adds rigorous walk-forward validation so that any strategy
or ML model can be tested without lookahead bias.

---

Why walk-forward validation?

A normal backtest tests a strategy on the same data it was designed on.
That makes results look better than they really are.

Walk-forward fixes this:
  - Train on window 1 → test on window 2 (data the strategy never saw)
  - Slide forward, repeat
  - Out-of-sample results are honest

This is a prerequisite for the gradient boosting work in V4.
Without it, any ML model trained on the full dataset would be cheating.

---

What's new in V3

  core/walk_forward.py   — generate train/test window splits (rolling + expanding)
  Strict temporal ordering enforced — no shuffling, no data leakage
  Out-of-sample performance tracked per window
  Rolling Sharpe and drawdown across windows

---

Project Structure

V3/
├── core/
│   ├── walk_forward.py        — window generation (new in V3)
│   ├── trade_log.py
│   ├── performance.py
│   ├── plots.py
│   ├── monte_carlo.py
│   ├── data_loader.py
│   └── config.py
│
├── strategies/
│   └── test_strategy.py
│
└── main.py

---

Author

Ryan Raj — Oxford Engineering (AI focus)

---

Disclaimer

Research and educational purposes only. Not financial advice.
