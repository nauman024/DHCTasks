# =====================================================================
# Task 2: Stock Price Prediction (Regression)
# =====================================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

print('All libraries loaded!')

# Download 2 years of Apple stock data
ticker = 'AAPL'
df = yf.download(ticker, start='2022-01-01', end='2024-01-01')
print(f"Data shape for {ticker}: {df.shape}")
print(df.head())

# Feature Engineering: Create Target Column (Shift up by 1 day)
df['Next_Close'] = df['Close'].shift(-1)
df = df.dropna()

# Define features and target matrix
features = ['Open', 'High', 'Low', 'Volume', 'Close']
X = df[features]
y = df['Next_Close']

print('Features shape:', X.shape)
print('Target shape:', y.shape)

# Split Data into Train and Test Sets (No Shuffling for Time-Series!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
print('Training samples:', len(X_train))
print('Testing samples:', len(X_test))

# Initialize and Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
print('Model trained successfully!')

# Evaluate Model Performance
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'\n--- Evaluation Metrics ---')
print(f'Mean Absolute Error: ${mae:.2f}')
print(f'R2 Score: {r2:.4f}')

# Plot Actual vs Predicted Prices
plt.figure(figsize=(12, 5))
plt.plot(y_test.values, label='Actual Price', color='blue', linewidth=1.5)
plt.plot(y_pred, label='Predicted Price', color='red', linewidth=1.5, linestyle='--')
plt.title(f'{ticker} Actual vs Predicted Closing Price', fontsize=14)
plt.xlabel('Trading Days')
plt.ylabel('Price (USD)')
plt.legend()
plt.tight_layout()
plt.show()