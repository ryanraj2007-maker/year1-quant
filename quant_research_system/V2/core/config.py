# =============================================================================
# V2 Core — Central Configuration
# =============================================================================
# Defines universal trading primitives shared across ALL strategies.
# Strategy-specific parameters that no other strategy would need live as
# defaults inside the strategy file itself — not here.
#
# Design rule: if you add a new strategy tomorrow and it can read these
# values without modification, the param belongs here. If it only makes
# sense for one strategy, it stays in that strategy's file.
#
# Sections:
#   Data         — symbols, dates, bar interval, data source
#   Session      — market hours (any strategy trading a session needs these)
#   Risk         — R:R ratio, slippage, position sizing
#   Monte Carlo  — simulation parameters
#   Plotting     — chart output settings
# =============================================================================


class Config:

    # ── Data ───────────────────────────────────────────────────────────────────
    # Default ticker/symbol and date range used by data fetchers.
    # Strategies can override per-run — these are the fallback defaults.
    TICKER      = "SPY"           # Equity ticker (stocks strategies)
    SYMBOL      = "MES"           # Futures symbol (futures strategies)
    START_DATE  = "2024-01-01"
    END_DATE    = "2024-12-31"
    INTERVAL    = "5m"            # Bar size: "1m", "5m", "15m", "1h", "1d"
    DATA_SOURCE = "yfinance"      # "yfinance" | "databento" | "alpaca"

    # ── Session ────────────────────────────────────────────────────────────────
    # Regular trading hours in tz-naive Eastern time.
    # Any strategy that filters to RTH uses these — no strategy should
    # hardcode "09:30" / "16:00" when this is the single source of truth.
    SESSION_START = "09:30"
    SESSION_END   = "16:00"       # Equities (NYSE/NASDAQ)

    # Futures RTH closes 15 min later than equities
    FUTURES_SESSION_END = "16:15"

    # ── Risk ───────────────────────────────────────────────────────────────────
    # Universal risk parameters applicable to any directional strategy.

    RR_RATIO    = 2.0    # Default reward-to-risk ratio for target placement.
                          # Each strategy can override, but this is the baseline.

    # Opening range window — used by any breakout strategy that defines a range
    # at the start of the session. "15 minutes" is the most backtested default.
    OR_MINUTES  = 15

    # Slippage defaults.
    # Stocks: dollar amount per share (market order fill, ~1 tick on SPY 5m)
    # Futures: number of ticks (strategy converts to points via contract spec)
    SLIPPAGE_PER_SHARE = 0.02     # $ per share for equity strategies
    ENTRY_SLIP_TICKS   = 1        # ticks for futures entry (market order)
    EXIT_SLIP_TICKS    = 1        # ticks for futures stop fills
                                   # (target fills = limit orders, no slippage)

    # Position sizing — affects dollar P&L and compounded equity reports.
    # Does NOT change which trades are taken, only how large each position is.
    ACCOUNT_SIZE   = 50_000       # Starting capital in USD
    RISK_PER_TRADE = 0.01         # Fraction of account risked per trade (1%)

    # ── Monte Carlo ────────────────────────────────────────────────────────────
    # Shared by any strategy that runs MC analysis after backtesting.
    N_SIMULATIONS   = 1_000       # 1000 = fast; 10_000 = smoother distribution
    STARTING_EQUITY = 1.0         # Normalised starting level for R equity curve

    # Ruin threshold for probability_of_ruin().
    # 0.9 = account has lost 10% from starting equity (common prop firm limit).
    # 0.0 = catastrophic ruin (account goes negative).
    RUIN_THRESHOLD  = 0.9

    # ── Plotting ───────────────────────────────────────────────────────────────
    N_PATHS_TO_PLOT = 100         # Max MC paths drawn on the fan chart
    RESULTS_DIR     = "results"   # Output directory for save_all_figures()
