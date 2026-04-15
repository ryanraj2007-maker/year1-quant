#!/usr/bin/env python3
"""
PostToolUse hook — lookahead bias detector.
Reads a Claude tool-use JSON payload from stdin and warns if the edited
Python file contains patterns that commonly indicate lookahead bias.
"""
import json
import re
import sys

PATTERNS = {
    r"shift\(-\s*\d+":          "negative shift (reads future bars)",
    r"\.pct_change\(-\d+":      "negative pct_change (future return)",
    r"rolling\([^)]*\)\.mean\(\)(?!.*min_periods)": "rolling without min_periods (potential partial window at edges)",
    r"future_return|fwd_return|forward_return": "forward-looking variable name",
    r"train_test_split(?!.*shuffle\s*=\s*False)": "train_test_split — confirm shuffle=False for time series",
}

def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # Can't parse — don't block

    tool = payload.get("tool_name", "")
    inp  = payload.get("tool_input", {})

    file_path = inp.get("file_path", "")
    if not file_path.endswith(".py"):
        sys.exit(0)

    # Only check files inside this project's core/strategies dirs
    if not any(d in file_path for d in ["/core/", "/strategies/", "/V2/", "/V3/", "/V4/"]):
        sys.exit(0)

    # Get the new content written/edited
    content = inp.get("new_string") or inp.get("content") or ""
    if not content:
        sys.exit(0)

    warnings = []
    for pattern, explanation in PATTERNS.items():
        if re.search(pattern, content):
            warnings.append(f"  - `{pattern.split('(')[0]}...`  →  {explanation}")

    if warnings:
        print(f"\n[LOOKAHEAD BIAS CHECK] Potential issues in {file_path.split('/')[-1]}:")
        for w in warnings:
            print(w)
        print("  Review before committing — lookahead = invalid backtest results.\n")
        # Exit 0: warn but don't block Claude
        sys.exit(0)

if __name__ == "__main__":
    main()
