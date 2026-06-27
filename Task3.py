# =====================================================================
# Task 3: Heart Disease Prediction (Binary Classification)
# =====================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('heart.csv')
print("Dataset Shape:", df.shape)
print(df.head())

# Data Cleaning check
print('\nMissing values per column:')
print(df.isnull().sum())

# Handle missing data if any exist dynamically
if df.isnull().sum().sum() > 0:
    df.fillna(df.mean(), inplace=True)

print('\nTarget column distribution (0 = No Disease, 1 = Disease):')
print(df['target'].value_counts())

# EDA 1: Age Distribution
plt.figure(figsize=(8, 4))
sns.histplot(df['age'], bins=20, kde=True, color='steelblue')
plt.title('Age Distribution of Patients')
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# EDA 2: Correlation Heatmap
plt.figure(figsize=(12, 9))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# Data Splitting and Feature Scaling
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Classification Model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Inferences
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
print(f'\nModel Test Accuracy: {acc * 100:.2f}%')

# Confusion Matrix Evaluation
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# ROC-AUC Curve Evaluation
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Heart Disease Prediction')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()