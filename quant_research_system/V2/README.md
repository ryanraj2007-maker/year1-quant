Quant Research System — V2 (Phase 1: Polish & Solidify Core)

Overview

This project is a modular quantitative research and backtesting framework designed to develop, test, and analyse systematic trading strategies.

Version 2 focuses on building a strategy-agnostic engine, where:
	•	strategies generate trades
	•	a unified framework evaluates performance
	•	risk and robustness are analysed through simulation

The goal is to replicate a professional quant workflow:
Strategy → Trade Log → Performance → Simulation → Iteration

⸻

Key Features

1. Modular Architecture

The system is structured so that:
	•	core components (data, metrics, simulation) are reusable
	•	strategies are plug-and-play

This allows rapid testing of new ideas without rewriting infrastructure.

⸻

2. Trade-Level Backtesting

Unlike simple signal-based models, this framework operates at the trade level, supporting:
	•	entry and exit prices
	•	stop loss and take profit
	•	position direction (long/short)
	•	trade duration
	•	R-multiples

All strategies output a standardized trade log, enabling consistent evaluation.

⸻

3. Performance Analytics

The system computes:
	•	Total return
	•	Win rate
	•	Average win / loss
	•	Expectancy
	•	Profit factor
	•	Maximum drawdown
	•	Equity curve

This allows objective comparison between strategies.

⸻

4. Visualisation

Built-in plotting tools provide:
	•	Equity curve
	•	Drawdown curve
	•	Trade return distribution

These help identify:
	•	consistency
	•	volatility
	•	tail risk

⸻

5. Monte Carlo Simulation

To evaluate robustness, the framework includes Monte Carlo simulation:
	•	resamples historical trades
	•	generates multiple equity paths
	•	estimates distribution of outcomes

This enables analysis of:
	•	drawdown risk
	•	variance of returns
	•	probability of different outcomes

⸻

6. Strategy Interface

Each strategy is implemented as an independent module and must return a standardized trade log.

This allows seamless comparison between:
	•	opening range strategies
	•	momentum strategies
	•	mean reversion strategies
	•	structure-based models (e.g. BOS, liquidity sweeps)

⸻

Project Structure

V2/
├── core/
│   ├── trade_log.py           — trade schema, validation, CSV/JSON export
│   ├── performance.py         — R-based metrics (expectancy, profit factor, drawdown)
│   ├── plots.py               — equity curve, drawdown curve, R-distribution
│   ├── monte_carlo.py         — bootstrapped trade sequence simulation
│   ├── data_loader.py         — data fetching (yFinance, Alpaca)
│   └── config.py              — central configuration
│
├── strategies/
│   ├── opening_range_breakout.py   — stock ORB (4 data sources, slippage, commission)
│   ├── futures_orb.py              — futures ORB (ES, MES, NQ, MNQ with multipliers)
│   └── key_level_orb.py            — 8AM key level strategy (Asia/London sessions)
│
└── main.py

⸻

First Strategy: Opening Range Breakout

The initial strategy implemented in this framework is a simple opening range breakout model.

Logic:
	1.	Define an opening range (e.g. 8:00–8:15)
	2.	If price breaks above → long bias
	3.	If price breaks below → short bias
	4.	Enter on breakout
	5.	Apply fixed stop loss and risk-reward target

This serves as a baseline model to:
	•	validate the framework
	•	generate trade data
	•	test performance and simulation modules

⸻

Design Philosophy

This system is built around the principle:

Separate strategy logic from analysis infrastructure
	•	Strategies generate trades
	•	The framework evaluates them

This ensures:
	•	scalability
	•	flexibility
	•	faster iteration

⸻

Roadmap

V2 / Phase 1 (current) — Polish & solidify core
	•	trade ID, duration, input validation, CSV/JSON export
	•	Sharpe ratio, losing streak, recovery factor in performance.py
	•	Probability of ruin, drawdown distribution in monte_carlo.py
	•	Monte Carlo band overlay on equity curve, streak plot, savefig export
	•	Centralise all parameters in config.py
	•	Abstract data_provider.py (YFinance + Alpaca + future sources)

Phase 2 — Prop firm pass simulator
	•	Rule engine for FTMO, Apex, TopStep, MyFundedFutures
	•	Monte Carlo pass/fail simulation
	•	Output: pass rate %, avg days, failure reason breakdown

Phase 3 — Interactive dashboard (Plotly Dash)
	•	Strategy overview, prop firm simulator, strategy comparison, trade log viewer

Phase 4 — Strategy discovery & optimisation
	•	Grid search, walk-forward validation, multi-strategy framework

Phase 5 — Live trading integration
	•	Alpaca (stocks), Tradovate/Rithmic (futures), real-time risk management

⸻

Purpose

This project is designed to:
	•	develop practical quant research skills
	•	build systematic trading strategies
	•	understand risk and robustness
	•	create a scalable research environment

⸻

Author

Ryan Raj
Oxford Engineering (AI focus)

⸻

Disclaimer

This project is for research and educational purposes only.
It does not constitute financial advice.

⸻
