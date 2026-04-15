# year1-quant — Project Context for Claude

## What this is
A systematic ML-driven trading research engine built by a Year 1 quant student.
End goal: gradient boosting signal generation with walk-forward validation.
Recruiting target: quant/AI internships 2026–2027.

## Versioning
| Version | Status | Description |
|---------|--------|-------------|
| V1 | Shipped | Baseline signal backtester |
| V2 | Current | Polished core + DatabentoProvider (finishing now) |
| V3 | Next | Walk-forward validation framework |
| V4 | Future | Gradient boosting signal generator (centrepiece) |
| V5 | Future | Interactive dashboard |
| V6 | Summer 2026 | Second ML layer (NLP sentiment default) |

## Repository layout
```
quant_research_system/
  V2/
    core/
      config.py        — universal params (TICKER, SYMBOL, dates, risk, MC settings)
      data_loader.py   — DataProvider ABC, YFinanceProvider, DatabentoProvider
      trade_log.py     — Trade dataclass, create_trade(), trades_to_dataframe()
      performance.py   — R-multiple analytics: Sharpe, max DD, expectancy, profit factor
      monte_carlo.py   — probability of ruin, drawdown distributions
      plots.py         — MC band overlay, win/loss streak charts, savefig export
    strategies/
      futures_orb.py   — Futures ORB (ES/MES/NQ/MNQ), includes its own fetch_data()
      opening_range_breakout.py — Equity ORB (SPY etc.)
      key_level_orb.py — ORB with key level filtering
    main.py            — entry point
```

## Core conventions

### R-multiples
All performance metrics use R-multiples (return relative to risk-per-trade), not dollar P&L.
This is standard prop trading practice. Dollar P&L is computed separately via `dollar_pnl` column.

### Timezone
All DataFrames use tz-naive Eastern time (America/New_York, then `tz_localize(None)`).
Session filtering uses `between_time("09:30", "16:15")` — no tz math needed.

### No lookahead bias
This is non-negotiable. Rules:
- Never use `shift(-n)` for feature computation (negative shift = future data)
- Walk forward splits must be strictly temporal — no shuffling ever
- Fit scalers/models only on training data, transform test data separately
- OR detection uses bar closes, not opens (next bar open used for entry)

### Strategy interface
`run_<strategy>(df, **params) -> pd.DataFrame` of trades.
Output must include: `entry_time`, `exit_time`, `direction`, `entry_price`, `exit_price`, `r_multiple`, `win`.
Compatible with `performance.summary_stats()`, `monte_carlo.run_monte_carlo()`, `plots.plot_all()`.

### Data sources
- **yfinance**: free, no setup, ~60 days of intraday history. Symbols: `ES=F`, `SPY`, etc.
- **Databento**: free tier (10 GB), 5+ years, futures via GLBX.MDP3 dataset.
  Prices in nanodollars (int) — always divide by 1e9. Continuous symbols: `ES.c.0`.

### Config
`Config` class in `core/config.py` is the single source of truth for shared params.
Strategy-specific params stay in the strategy file. Don't add strategy-specific params to Config.

## Phase 3 red flags
Target Sharpe after costs: 0.5–1.5. Above 1.5 = assume lookahead bias until proven otherwise.
