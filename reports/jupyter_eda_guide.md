# Jupyter Notebook Guide: Breast Cancer Exploratory Data Analysis (EDA)

This document contains a structured set of cells for a Jupyter Notebook (`.ipynb`). Copy and paste the code from each section below into consecutive notebook cells to perform a comprehensive EDA with visual charts.

---

### Cell 1: Imports & Notebook Configuration
Loads the visualization packages and configures inline plotting dimensions.
```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

# Configure plotting aesthetics
%matplotlib inline
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12
```

---

### Cell 2: Dataset Loading & Metadata
Loads the dataset from scikit-learn, parses the column names to standard snake_case, and displays basic dimensions.
```python
# Load raw dataset
raw_data = load_breast_cancer()
df = pd.DataFrame(data=raw_data.data, columns=raw_data.feature_names)

# Convert column names to clean snake_case
df.columns = [col.replace(" ", "_") for col in df.columns]

# Add target variable (0 = Malignant, 1 = Benign)
df["target"] = raw_data.target

print(f"Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Target Labels Map: 0 -> Malignant, 1 -> Benign")
df.head()
```

---

### Cell 3: Check for Missing Values & Data Types
Analyzes the datatypes and checks if there are any missing or null values in the columns.
```python
print("--- Dataset Information ---")
print(df.info())

print("\n--- Missing Values Count ---")
null_counts = df.isnull().sum()
if null_counts.sum() == 0:
    print("Excellent! The dataset contains 0 missing/null values.")
else:
    print(null_counts[null_counts > 0])
```

---

### Cell 4: Class Distribution Analysis
Visualizes the ratio of Benign vs Malignant cases using a bar chart with counts.
```python
# Count class values
target_counts = df["target"].value_counts()
labels = ["Benign (1)", "Malignant (0)"]

# Create countplot
ax = sns.countplot(x="target", data=df, hue="target", palette=["#e63946", "#2a9d8f"], legend=False)
plt.title("Distribution of Tumors (Benign vs Malignant)", fontsize=14, pad=15)
plt.xlabel("Diagnosis Label")
plt.ylabel("Number of Samples")
plt.xticks(ticks=[0, 1], labels=["Malignant (0)", "Benign (1)"])

# Annotate counts on top of bars
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height() + 5),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')

plt.show()

print(f"Malignant Cases (0): {target_counts[0]} ({target_counts[0] / len(df) * 100:.2f}%)")
print(f"Benign Cases (1): {target_counts[1]} ({target_counts[1] / len(df) * 100:.2f}%)")
```

---

### Cell 5: Descriptive Statistics of Core Parameters
Summarizes mathematical distributions (mean, standard deviation, min, max, percentiles) for the core size features.
```python
# Inspect statistics of mean (core) parameters
mean_cols = [col for col in df.columns if col.startswith("mean_")]
df[mean_cols].describe().T
```

---

### Cell 6: Visualizing Feature Distributions (Histograms)
Compares the density distributions of key tumor parameters (Radius, Area, Texture, Compactness) grouped by Benign vs Malignant.
```python
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

features_to_plot = [
    ("mean_radius", "Mean Radius (pixels)"),
    ("mean_texture", "Mean Texture (gray-scale)"),
    ("mean_area", "Mean Area (pixels^2)"),
    ("mean_compactness", "Mean Compactness")
]

for idx, (col, label) in enumerate(features_to_plot):
    ax = axes[idx // 2, idx % 2]
    sns.histplot(data=df, x=col, hue="target", kde=True, bins=30,
                 palette={0: "#e63946", 1: "#2a9d8f"}, ax=ax, alpha=0.6)
    ax.set_title(f"Distribution of {label}", fontsize=12)
    ax.set_xlabel(label)
    ax.legend(labels=["Benign", "Malignant"])

plt.tight_layout()
plt.show()
```

---

### Cell 7: Correlation Analysis (Heatmap)
Visualizes relationships between the core features. High correlation indicates multicollinearity, which is important for Logistic Regression.
```python
# Compute correlation matrix of core features
corr_matrix = df[mean_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, square=True)
plt.title("Correlation Heatmap of Core Features", fontsize=14, pad=15)
plt.show()
```

---

### Cell 8: Outlier Detection (Box Plots)
Generates box plots for key worst case features to easily spot statistical outliers and evaluate diagnostic separation.
```python
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

worst_features = [
    ("worst_radius", "Worst Radius"),
    ("worst_area", "Worst Area"),
    ("worst_concavity", "Worst Concavity")
]

for idx, (col, label) in enumerate(worst_features):
    ax = axes[idx]
    sns.boxplot(x="target", y=col, data=df, hue="target", palette={0: "#e63946", 1: "#2a9d8f"}, ax=ax, legend=False)
    ax.set_title(f"Box Plot: {label}", fontsize=12)
    ax.set_xlabel("Diagnosis")
    ax.set_ylabel(label)
    ax.set_xticklabels(["Malignant (0)", "Benign (1)"])

plt.tight_layout()
plt.show()
```

---

### Cell 9: Pairplot of Strong Correlated Features
Generates a scatter matrix comparing core size parameters.
```python
# Pairplot of selected size features
selected_cols = ["mean_radius", "mean_texture", "mean_area", "mean_smoothness", "target"]
sns.pairplot(df[selected_cols], hue="target", palette={0: "#e63946", 1: "#2a9d8f"}, diag_kind="kde")
plt.suptitle("Pair Plot of Key Diagnostic Features", y=1.02, fontsize=14)
plt.show()
```
