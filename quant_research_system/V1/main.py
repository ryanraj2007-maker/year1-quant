# V1 — Main
# Full pipeline: load data → indicators → signal → backtest → metrics

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
    # Settings
    ticker = "SPY"
    start_date = "2015-01-01"
    end_date   = "2025-01-01"

    short_window    = 10
    long_window     = 50
    vol_window      = 20
    momentum_window = 20

    # Load
    df = load_price_data(ticker, start_date, end_date)

    # Indicators
    df = add_returns(df)
    df = add_moving_average(df, window=short_window,    column_name=f"ma_{short_window}")
    df = add_moving_average(df, window=long_window,     column_name=f"ma_{long_window}")
    df = add_volatility(df,     window=vol_window,      column_name=f"vol_{vol_window}")
    df = add_momentum(df,       window=momentum_window, column_name=f"mom_{momentum_window}")
    df = df.dropna()

    # Signal
    df = moving_average_crossover_strategy(
        df,
        short_ma_col=f"ma_{short_window}",
        long_ma_col=f"ma_{long_window}",
        signal_col="signal"
    )

    # Backtest
    df = run_backtest(df, signal_col="signal", return_col="returns")

    # Metrics
    strategy_total_return      = calculate_total_return(df["strategy_equity"])
    benchmark_total_return     = calculate_total_return(df["buy_and_hold_equity"])
    strategy_annualised_return  = calculate_annualised_return(df["strategy_equity"])
    benchmark_annualised_return = calculate_annualised_return(df["buy_and_hold_equity"])
    strategy_sharpe            = calculate_sharpe_ratio(df["strategy_return"])
    benchmark_sharpe           = calculate_sharpe_ratio(df["buy_and_hold_return"])
    strategy_max_drawdown      = calculate_max_drawdown(df["strategy_equity"])
    benchmark_max_drawdown     = calculate_max_drawdown(df["buy_and_hold_equity"])
    strategy_win_rate          = calculate_win_rate(df["strategy_return"])
    benchmark_win_rate         = calculate_win_rate(df["buy_and_hold_return"])

    # Results
    print(f"\nBacktest results for {ticker}")
    print("-" * 50)
    print("Strategy: Moving Average Crossover")
    print(f"Short window: {short_window} | Long window: {long_window}")

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

    print("\nLast 5 rows:")
    print(df[["Close", "signal", "position", "returns",
              "strategy_return", "strategy_equity", "buy_and_hold_equity"]].tail())


if __name__ == "__main__":
    main()
