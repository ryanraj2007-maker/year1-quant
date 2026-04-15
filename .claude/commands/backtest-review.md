Review the current backtest results from the most recently run strategy in this project.

Steps:
1. Find the trades DataFrame (check the most recent run in context, or ask which strategy to review if unclear).
2. Compute and report the following metrics using `core/performance.py`:
   - **Sharpe ratio** (annualised, trade-level R-multiples)
   - **Max drawdown** (in R and %)
   - **Calmar ratio** = annualised return / |max drawdown %| — if Calmar > 3, flag as suspicious
   - **Expectancy** (average R per trade)
   - **Profit factor**
   - **Win rate** and average win R / average loss R
   - **Longest losing streak**
   - **Recovery factor**
3. **Transaction cost sensitivity**: re-run the strategy (or compute analytically) at 0 ticks, 1 tick, 2 ticks, and 3 ticks of slippage per side. Show how Sharpe and total R degrade. Flag if the strategy breaks even below 2 ticks — that's a fragile edge.
4. **Turnover**: estimate trades per year. Flag if > 500 trades/year (transaction costs will dominate).
5. **Benchmark comparison**: compare compounded equity curve vs buy-and-hold on the same instrument and period.
6. **Red flags checklist**:
   - [ ] Sharpe > 1.5 without a clear structural reason
   - [ ] Win rate > 70% (possible exit bias or lookahead)
   - [ ] Average loss R worse than -1.5 (stops not being respected)
   - [ ] Calmar > 3
   - [ ] Edge disappears at 2 ticks slippage
   - [ ] Less than 30 trades (statistically meaningless)

Format the output as a concise report with a verdict at the end: PASS / INVESTIGATE / FAIL.
