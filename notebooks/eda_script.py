import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/command line usage
import matplotlib.pyplot as plt
import seaborn as sns
from src.logger import logger
from src.exception import CustomException

def generate_eda_report(raw_data_path: str, report_dir: str):
    """
    Reads the raw dataset and generates descriptive figures and a Markdown summary report.
    """
    try:
        logger.info("Starting automated EDA report generation...")
        os.makedirs(report_dir, exist_ok=True)
        figures_dir = os.path.join(report_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        # Load dataset
        df = pd.read_csv(raw_data_path)
        logger.info(f"Loaded dataset with shape {df.shape} for EDA.")
        
        # 1. Class Distribution
        plt.figure(figsize=(6, 4))
        sns.countplot(x="target", data=df, palette="coolwarm")
        plt.title("Tumor Class Distribution (0 = Malignant, 1 = Benign)")
        plt.xlabel("Class")
        plt.ylabel("Count")
        dist_path = os.path.join(figures_dir, "class_distribution.png")
        plt.savefig(dist_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved class distribution plot to {dist_path}")
        
        # 2. Feature Correlation (Top features correlating with target)
        correlations = df.corr()["target"].sort_values()
        top_correlated_features = list(correlations.index[:5]) + list(correlations.index[-5:])
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[top_correlated_features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap (Top Correlated Features with Target)")
        heatmap_path = os.path.join(figures_dir, "correlation_heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved correlation heatmap to {heatmap_path}")
        
        # 3. Write reports/eda_report.md
        report_md_path = os.path.join(report_dir, "eda_report.md")
        logger.info(f"Writing EDA markdown report to {report_md_path}...")
        
        malignant_count = (df["target"] == 0).sum()
        benign_count = (df["target"] == 1).sum()
        total_count = len(df)
        
        with open(report_md_path, "w") as f:
            f.write("# Exploratory Data Analysis (EDA) Report\n\n")
            f.write("This report presents the exploratory analysis of the Breast Cancer Wisconsin Diagnostic Dataset.\n\n")
            
            f.write("## 1. Dataset Overview\n")
            f.write(f"- **Total Samples**: {total_count}\n")
            f.write(f"- **Total Features**: {df.shape[1] - 1}\n")
            f.write(f"- **Malignant Cases (Class 0)**: {malignant_count} ({malignant_count/total_count*100:.2f}%)\n")
            f.write(f"- **Benign Cases (Class 1)**: {benign_count} ({benign_count/total_count*100:.2f}%)\n\n")
            
            f.write("## 2. Target Class Distribution\n")
            f.write("The distribution shows a slight class imbalance but is representative and suitable for training a logistic regression model.\n\n")
            f.write("![Class Distribution](figures/class_distribution.png)\n\n")
            
            f.write("## 3. High Correlation Features\n")
            f.write("We selected the top features that exhibit strong linear correlation with the target label. Highly negative correlated features mean that higher values indicate a malignant classification (class 0).\n\n")
            f.write("![Correlation Heatmap](figures/correlation_heatmap.png)\n\n")
            
            f.write("## 4. Key Observations\n")
            f.write("1. **Strong Predictors**: Features like `worst_concave_points`, `worst_perimeter`, `worst_radius`, and `mean_concave_points` display strong negative correlation with the target variable, meaning larger values correlate highly with Malignant (0) status.\n")
            f.write("2. **Multicollinearity**: Standard features like `radius`, `perimeter`, and `area` are highly multicollinear (correlation near 1.0), which makes regularization (L2/L1) necessary for Logistic Regression.\n")
            
        logger.info("EDA Report generation completed successfully!")
    except Exception as e:
        raise CustomException(e, sys) from e

if __name__ == "__main__":
    raw_path = "artifacts/data_ingestion/raw.csv"
    report_dir = "reports"
    generate_eda_report(raw_path, report_dir)
