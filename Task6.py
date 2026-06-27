# =====================================================================
# Task 6: House Price Prediction (Advanced Regression Modeling)
# =====================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load source files configuration matrix tracking dataset
df = pd.read_csv('train.csv')
print("House Dataset Shape:", df.shape)

# Feature Subset Selection Isolation Array
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'YearBuilt', 'OverallQual', 'GarageCars', 'TotalBsmtSF']
target = 'SalePrice'

df_model = df[features + [target]].copy()

# Fill Missing Values with Median Column Values Natively
df_model.fillna(df_model.median(), inplace=True)
print('Remaining missing values count:', df_model.isnull().sum().sum())

# EDA 1: Target Sale Price Density Metric Dist
plt.figure(figsize=(8, 4))
sns.histplot(df_model['SalePrice'], bins=40, kde=True, color='coral')
plt.title('Distribution of House Sale Prices')
plt.xlabel('Sale Price (USD)')
plt.tight_layout()
plt.show()

# EDA 2: Living Area Regression Distribution Correlation Mapping
plt.figure(figsize=(8, 5))
plt.scatter(df_model['GrLivArea'], df_model['SalePrice'], alpha=0.4, color='steelblue')
plt.xlabel('Above Ground Living Area (sq ft)')
plt.ylabel('Sale Price (USD)')
plt.title('Living Area vs Sale Price')
plt.tight_layout()
plt.show()

# Split datasets inputs transformations target spaces parameters
X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize Powerful Ensemble Gradient Boosting Regressor Architecture Model
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Compute Numerical Mathematical Metric Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'\n--- Model Evaluation Summary Metrics ---')
print(f'Mean Absolute Error (MAE): ${mae:,.2f}')
print(f'Root Mean Squared Error (RMSE): ${rmse:,.2f}')
print(f'R2 Predictive Variance Score: {r2:.4f}')

# Plot Predictions Variance Layout Map Visualizer
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', lw=2, label='Perfect Prediction Reference Line')
plt.xlabel('Actual Price (USD)')
plt.ylabel('Predicted Price (USD)')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.tight_layout()
plt.show()

# Feature Importance Computation Pipeline
importances = model.feature_importances_
feat_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=True)

plt.figure(figsize=(7, 5))
plt.barh(feat_df['Feature'], feat_df['Importance'], color='steelblue')
plt.title('Feature Importance Matrix - House Price Prediction')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()