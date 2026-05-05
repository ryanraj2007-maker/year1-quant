# V2 Core — Central Configuration
# Universal trading parameters shared across all strategies.
# Strategy-specific params stay in the strategy file itself.

class Config:

    # Data
    TICKER      = "SPY"
    SYMBOL      = "MES"
    START_DATE  = "2024-01-01"
    END_DATE    = "2024-12-31"
    INTERVAL    = "5m"          # "1m", "5m", "15m", "1h", "1d"
    DATA_SOURCE = "yfinance"    # "yfinance" | "databento"

    # Session times (tz-naive Eastern)
    SESSION_START       = "09:30"
    SESSION_END         = "16:00"
    FUTURES_SESSION_END = "16:15"

    # Risk
    RR_RATIO           = 2.0
    OR_MINUTES         = 15
    SLIPPAGE_PER_SHARE = 0.02
    ENTRY_SLIP_TICKS   = 1
    EXIT_SLIP_TICKS    = 1
    ACCOUNT_SIZE       = 50_000
    RISK_PER_TRADE     = 0.01   # 1% per trade

    # Monte Carlo
    N_SIMULATIONS   = 1_000
    STARTING_EQUITY = 1.0
    RUIN_THRESHOLD  = 0.9       # 10% drawdown limit

    # Plotting
    N_PATHS_TO_PLOT = 100
    RESULTS_DIR     = "results"
