# =============================================================================
# V1 — Main Entry Point
# =============================================================================
# Orchestrates the full V1 backtesting pipeline in one place:
#   Load data → Add indicators → Generate signal → Backtest → Report metrics
#
# To change the strategy parameters, edit the settings block in main() below.
# =============================================================================

from data_loader import load_price_data
from indicators import add_returns, add_moving_average, add_volatility, add_momentum
from strategies import moving_average_crossover_strategy
from backtester import run_backtest
from metrics import (
    calculate_total_return,
    calculate_annualised_return,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_win_rate,
)


def main():
    # ── 1. Research settings ──────────────────────────────────────────────
    # Change these to test different tickers, time periods, or MA windows.
    ticker = "SPY"
    start_date = "2015-01-01"
    end_date   = "2025-01-01"

    short_window    = 10   # fast MA — reacts quickly to price changes
    long_window     = 50   # slow MA — captures longer-term trend
    vol_window      = 20   # window for rolling volatility
    momentum_window = 20   # window for momentum calculation

    # ── 2. Load data ──────────────────────────────────────────────────────
    df = load_price_data(ticker, start_date, end_date)

    # ── 3. Add indicators ─────────────────────────────────────────────────
    # Each call returns a new DataFrame with one extra column added.
    df = add_returns(df)
    df = add_moving_average(df, window=short_window,    column_name=f"ma_{short_window}")
    df = add_moving_average(df, window=long_window,     column_name=f"ma_{long_window}")
    df = add_volatility(df,     window=vol_window,      column_name=f"vol_{vol_window}")
    df = add_momentum(df,       window=momentum_window, column_name=f"mom_{momentum_window}")

    # Drop rows that have NaN from rolling windows — these are the warm-up
    # bars at the start where there aren't enough prior values to compute
    df = df.dropna()

    # ── 4. Generate strategy signal ───────────────────────────────────────
    # Signal = 1 (long) when the short MA is above the long MA
    # Signal = 0 (flat) when the short MA is below or equal to the long MA
    df = moving_average_crossover_strategy(
        df,
        short_ma_col=f"ma_{short_window}",
        long_ma_col=f"ma_{long_window}",
        signal_col="signal"
    )

    # ── 5. Run backtest ───────────────────────────────────────────────────
    # Converts the signal into position, returns, and equity curves
    df = run_backtest(df, signal_col="signal", return_col="returns")

    # ── 6. Calculate metrics ──────────────────────────────────────────────
    # Compute the same metrics for both the strategy and buy-and-hold
    # so we can see whether the strategy actually adds value
    strategy_total_return      = calculate_total_return(df["strategy_equity"])
    benchmark_total_return     = calculate_total_return(df["buy_and_hold_equity"])

    strategy_annualised_return  = calculate_annualised_return(df["strategy_equity"])
    benchmark_annualised_return = calculate_annualised_return(df["buy_and_hold_equity"])

    strategy_sharpe   = calculate_sharpe_ratio(df["strategy_return"])
    benchmark_sharpe  = calculate_sharpe_ratio(df["buy_and_hold_return"])

    strategy_max_drawdown  = calculate_max_drawdown(df["strategy_equity"])
    benchmark_max_drawdown = calculate_max_drawdown(df["buy_and_hold_equity"])

    strategy_win_rate  = calculate_win_rate(df["strategy_return"])
    benchmark_win_rate = calculate_win_rate(df["buy_and_hold_return"])

    # ── 7. Print results ──────────────────────────────────────────────────
    print(f"\nBacktest results for {ticker}")
    print("-" * 50)

    print("Strategy: Moving Average Crossover")
    print(f"Short window: {short_window}")
    print(f"Long window:  {long_window}")

    print("\nStrategy Performance")
    print(f"Total Return:       {strategy_total_return:.2%}")
    print(f"Annualised Return:  {strategy_annualised_return:.2%}")
    print(f"Sharpe Ratio:       {strategy_sharpe:.2f}")
    print(f"Max Drawdown:       {strategy_max_drawdown:.2%}")
    print(f"Win Rate:           {strategy_win_rate:.2%}")

    print("\nBuy and Hold Performance")
    print(f"Total Return:       {benchmark_total_return:.2%}")
    print(f"Annualised Return:  {benchmark_annualised_return:.2%}")
    print(f"Sharpe Ratio:       {benchmark_sharpe:.2f}")
    print(f"Max Drawdown:       {benchmark_max_drawdown:.2%}")
    print(f"Win Rate:           {benchmark_win_rate:.2%}")

    print("\nLast 5 rows of backtest data:")
    print(df[[
        "Close",
        "signal",
        "position",
        "returns",
        "strategy_return",
        "strategy_equity",
        "buy_and_hold_equity"
    ]].tail())


if __name__ == "__main__":
    main()
