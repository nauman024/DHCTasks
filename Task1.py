# =====================================================================
# Task 1: Exploring & Visualizing a Dataset (Iris Flowers)
# =====================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for professional charts
sns.set_theme(style="whitegrid")
print('Libraries imported successfully!')

# Load the Iris dataset directly from seaborn
df = sns.load_dataset('iris')
print('\nFirst 5 rows of the dataset:')
print(df.head())

# Explore Dataset Structure
print('\n--- Dataset Information ---')
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
df.info()

print('\n--- Statistical Summary ---')
print(df.describe())

print('\n--- Class Distribution ---')
print(df['species'].value_counts())

# Visualization 1: Scatter Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x='sepal_length',
    y='sepal_width',
    hue='species',
    palette='Set1'
)
plt.title('Sepal Length vs Sepal Width by Species', fontsize=14, pad=10)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend(title='Species')
plt.tight_layout()
plt.show()

# Visualization 2: Histograms
df.drop('species', axis=1).hist(
    bins=20,
    figsize=(10, 6),
    color='steelblue',
    edgecolor='black'
)
plt.suptitle('Distribution of All Features', y=1.02, fontsize=14)
plt.tight_layout()
plt.show()

# Visualization 3: Box Plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
for i, feature in enumerate(features):
    ax = axes[i // 2, i % 2]
    sns.boxplot(data=df, x='species', y=feature, palette='pastel', ax=ax)
    ax.set_title(f'Box Plot of {feature.replace("_", " ").title()}')
plt.tight_layout()
plt.show()