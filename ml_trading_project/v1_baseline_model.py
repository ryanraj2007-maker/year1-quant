# =============================================================================
# ML Trading — V1 Baseline Model
# =============================================================================
# A simple logistic regression model that tries to predict whether AAPL's
# next-day return will be positive or negative.
#
# This is a proof-of-concept baseline. It intentionally keeps the ML pipeline
# simple so the focus stays on understanding the structure:
#   Feature engineering → Train/test split → Model fit → Evaluate
# =============================================================================

import pandas as pd
import numpy as np
import sklearn as sk
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# ── Data download ─────────────────────────────────────────────────────────────
ticker     = 'AAPL'
start_date = '2020-01-01'
end_date   = '2025-01-01'

df = yf.download(ticker, start=start_date, end=end_date)

# yfinance returns a MultiIndex when downloading a single ticker — drop the
# redundant ticker level so columns are just Open, High, Low, Close, Volume
df.columns = df.columns.droplevel(1)

# ── Feature engineering ───────────────────────────────────────────────────────
# These are simple technical features derived from price and returns.
# Each one tries to capture a different aspect of recent market behaviour.

df["returns"]      = df["Close"].pct_change()              # today's return
df["ma_5"]         = df["Close"].rolling(window=5).mean()  # short-term trend
df["volatility_5"] = df["returns"].rolling(window=5).std() # recent vol
df["ma_10"]        = df["Close"].rolling(window=10).mean() # medium-term trend
df["momentum_5"]   = df["Close"] / df["Close"].shift(5) - 1  # 5-day momentum

# ── Target variable ───────────────────────────────────────────────────────────
# Predict whether tomorrow's return is positive (1) or not (0).
# shift(-1) looks one row ahead — this is valid because we're labelling today's
# data with tomorrow's outcome, which we'd only know at the next close.
df["target"] = (df["returns"].shift(-1) > 0).astype(int)

# Drop any rows that have NaN from rolling calculations or the shift
df = df.dropna()

# ── Train / test split ────────────────────────────────────────────────────────
# Use a chronological 80/20 split — NOT random shuffling.
# Shuffling would leak future data into training, inflating results.
features = ["returns", "ma_5", "ma_10", "volatility_5", "momentum_5"]
X = df[features]
y = df["target"]

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

# ── Model ─────────────────────────────────────────────────────────────────────
# Logistic regression outputs a probability between 0 and 1.
# class_weight="balanced" adjusts for unequal class frequencies
# (markets go up more often than they go down, so without this the model
# would learn to just always predict 1).
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

# ── Prediction ────────────────────────────────────────────────────────────────
# predict_proba returns [prob_class_0, prob_class_1] for each row.
# We take the probability of class 1 (up day) and apply a custom threshold.
# 0.45 instead of 0.50 means we predict "up" more aggressively,
# trading precision for recall.
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob > 0.45).astype(int)

# ── Evaluation ────────────────────────────────────────────────────────────────
# Confusion matrix: rows = actual, columns = predicted
# [[TN, FP],
#  [FN, TP]]
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Baseline: if we just predict "up" every single day, what accuracy do we get?
# This is the minimum bar the model needs to beat to be useful.
baseline = y_test.mean()
print("Baseline (always predict 1):", baseline)

# Diagnostic checks on the predicted probabilities
print("Predicted 1s:", np.mean(y_pred))
print("Min prob:",  y_prob.min())
print("Max prob:",  y_prob.max())
print("Mean prob:", y_prob.mean())

# Full precision/recall/F1 breakdown per class
print(classification_report(y_test, y_pred))
