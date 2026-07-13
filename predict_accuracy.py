import os
import sys
import pandas as pd

# Append workspace directory to system path
sys.path.append(os.getcwd())

from src.pipeline.prediction_pipeline import PredictionPipeline

def test_model_predictions():
    """
    Loads samples from the validation split, runs real-time inference,
    and displays model classifications alongside confidence percentages.
    """
    print("Initializing PredictionPipeline...")
    pipeline = PredictionPipeline()
    
    test_csv_path = "artifacts/data_ingestion/test_data.csv"
    if not os.path.exists(test_csv_path):
        print(f"Error: {test_csv_path} does not exist. Please run training pipeline first.")
        return
        
    df = pd.read_csv(test_csv_path)
    print(f"Loaded test dataset from {test_csv_path} (shape: {df.shape})\n")
    
    # Filter 5 malignant (0) and 5 benign (1) samples
    malignant_samples = df[df["target"] == 0].head(5)
    benign_samples = df[df["target"] == 1].head(5)
    samples = pd.concat([malignant_samples, benign_samples]).reset_index(drop=True)
    
    print("Evaluating Test Samples:")
    print("=" * 110)
    print(f"{'No.':<4} | {'Mean Radius':<12} | {'Mean Area':<12} | {'True Class':<12} | {'Pred Class':<12} | {'Confidence (%)':<16} | {'Status':<10}")
    print("=" * 110)
    
    correct_predictions = 0
    for idx, row in samples.iterrows():
        # Format features as a single row dataframe
        feature_row = pd.DataFrame([row.drop("target").to_dict()])
        
        # Inference
        pred_class, pred_label, confidence = pipeline.predict(feature_row)
        
        true_label = "Benign" if int(row["target"]) == 1 else "Malignant"
        is_correct = (pred_class == int(row["target"]))
        status = "CORRECT" if is_correct else "MISMATCH"
        
        if is_correct:
            correct_predictions += 1
            
        print(f"{idx+1:<4} | {row['mean_radius']:<12.3f} | {row['mean_area']:<12.1f} | {true_label:<12} | {pred_label:<12} | {confidence*100:<15.2f}% | {status:<10}")
        
    print("=" * 110)
    print(f"Sample Accuracy: {correct_predictions/len(samples)*100:.2f}% ({correct_predictions}/{len(samples)})")

if __name__ == "__main__":
    test_model_predictions()
