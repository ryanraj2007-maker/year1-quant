# ML Trading — V1 Baseline Model
# Logistic regression to predict whether AAPL's next-day return will be positive or negative.
# Simple proof-of-concept — the focus is on getting the ML pipeline structure right.

import pandas as pd
import numpy as np
import sklearn as sk
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Data
ticker     = 'AAPL'
start_date = '2020-01-01'
end_date   = '2025-01-01'

df = yf.download(ticker, start=start_date, end=end_date)
df.columns = df.columns.droplevel(1)  # yfinance MultiIndex — drop redundant ticker level

# Features
df["returns"]      = df["Close"].pct_change()
df["ma_5"]         = df["Close"].rolling(window=5).mean()
df["volatility_5"] = df["returns"].rolling(window=5).std()
df["ma_10"]        = df["Close"].rolling(window=10).mean()
df["momentum_5"]   = df["Close"] / df["Close"].shift(5) - 1

# Target — 1 if tomorrow's return is positive, 0 otherwise
df["target"] = (df["returns"].shift(-1) > 0).astype(int)
df = df.dropna()

# Train/test split — chronological 80/20, NOT random (random would leak future data)
features = ["returns", "ma_5", "ma_10", "volatility_5", "momentum_5"]
X = df[features]
y = df["target"]

split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Model — class_weight="balanced" stops it from just always predicting "up"
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

# Predictions — using 0.45 threshold instead of 0.5 to catch more up days
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob > 0.45).astype(int)

# Evaluation
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Baseline: what accuracy would we get by always predicting "up"?
baseline = y_test.mean()
print("Baseline (always predict 1):", baseline)

print("Predicted 1s:", np.mean(y_pred))
print("Min prob:",  y_prob.min())
print("Max prob:",  y_prob.max())
print("Mean prob:", y_prob.mean())

print(classification_report(y_test, y_pred))
