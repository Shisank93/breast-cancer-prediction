import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
from src.logger import logger
from src.exception import CustomException
from src.entity import ModelEvaluationConfig, ModelEvaluationArtifact, ModelTrainerArtifact, DataTransformationArtifact
from src.utils import load_object, load_numpy_array_data

class ModelEvaluation:
    """
    ModelEvaluation class performs inference on the test dataset, computes binary metrics,
    plots evaluation charts, validates performance thresholds, and saves a JSON report.
    """
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            logger.info("Initialized ModelEvaluation component.")
        except Exception as e:
            raise CustomException(e, sys) from e

    def evaluate_model(self) -> ModelEvaluationArtifact:
        """
        Calculates classification metrics, plots confusion matrix & ROC,
        compares accuracy against acceptance criteria and returns a ModelEvaluationArtifact.
        """
        logger.info("Entered evaluate_model method of ModelEvaluation class.")
        try:
            # Load transformed test array
            test_npy_path = self.data_transformation_artifact.transformed_test_file_path
            logger.info(f"Loading transformed test array from {test_npy_path}...")
            test_arr = load_numpy_array_data(str(test_npy_path))
            
            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]
            
            # Load model object
            model_path = self.model_trainer_artifact.model_file_path
            logger.info(f"Loading model object from {model_path}...")
            model = load_object(str(model_path))
            
            # Predict labels and probabilities
            logger.info("Generating predictions on test set...")
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Compute classification metrics
            logger.info("Computing classification metrics...")
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="binary")
            rec = recall_score(y_test, y_pred, average="binary")
            f1 = f1_score(y_test, y_pred, average="binary")
            auc = roc_auc_score(y_test, y_prob)
            
            cm = confusion_matrix(y_test, y_pred)
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_str = classification_report(y_test, y_pred)
            
            logger.info(f"Metrics - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
            
            # Generate plots
            figures_dir = os.path.join(os.getcwd(), "reports", "figures")
            os.makedirs(figures_dir, exist_ok=True)
            
            # A. Confusion Matrix plot
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                        xticklabels=["Malignant", "Benign"], 
                        yticklabels=["Malignant", "Benign"])
            plt.title("Confusion Matrix")
            plt.ylabel("True Class")
            plt.xlabel("Predicted Class")
            cm_plot_path = os.path.join(figures_dir, "confusion_matrix.png")
            plt.savefig(cm_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved confusion matrix plot to {cm_plot_path}")
            
            # B. ROC Curve plot
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {auc:.4f})")
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic (ROC) Curve")
            plt.legend(loc="lower right")
            roc_plot_path = os.path.join(figures_dir, "roc_curve.png")
            plt.savefig(roc_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved ROC curve plot to {roc_plot_path}")
            
            # Model acceptance logic
            acceptance_threshold = 0.92
            is_model_accepted = bool(acc >= acceptance_threshold)
            
            metrics_report = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": auc,
                "confusion_matrix": cm.tolist(),
                "classification_report": report_dict,
                "is_model_accepted": is_model_accepted,
                "acceptance_threshold": acceptance_threshold
            }
            
            report_path = self.model_evaluation_config.report_file_path
            logger.info(f"Writing evaluation report JSON to {report_path}...")
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, "w") as f:
                json.dump(metrics_report, f, indent=4)
                
            # Log classification report details
            logger.info(f"Classification Report details:\n{report_str}")
            
            artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                report_file_path=report_path
            )
            logger.info(f"Model evaluation completed. Artifact created: {artifact}")
            return artifact
            
        except Exception as e:
            raise CustomException(e, sys) from e
