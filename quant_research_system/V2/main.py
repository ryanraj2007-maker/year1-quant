from core.data_loader import YFinanceProvider
from strategies.test_strategy import run
from core.trade_log import trades_to_dataframe
from core.performance import summary_stats
from core.monte_carlo import run_monte_carlo, monte_carlo_summary
from core.plots import plot_all
import matplotlib.pyplot as plt

provider = YFinanceProvider()
df = provider.fetch("SPY", "2025-12-01", "2026-05-01", "1d")
trades = trades_to_dataframe(run(df))
simulations = run_monte_carlo(trades)
print(summary_stats(trades))
print(monte_carlo_summary(trades, simulations=simulations))
plot_all(trades, simulations=simulations, style="paths")
plt.show()