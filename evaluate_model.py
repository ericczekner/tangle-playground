import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from joblib import load # Used to load the model artifact
import sys

# Usage: python evaluate_model.py <input_model_path> <input_test_features_path> <input_test_labels_path> <output_report_path>

input_model_path = sys.argv[1]
input_features_path = sys.argv[2] # Will be 'X_test_for_eval.csv'
input_labels_path = sys.argv[3]   # Will be 'y_test_for_eval.csv'
output_report_path = sys.argv[4]

try:
    # --- Load Artifacts ---
    print("Loading model and test data...")
    # 1. Load the trained model
    model = load(input_model_path)
    
    # 2. Load the test features and labels saved by the training step
    X_test = pd.read_csv(input_features_path)
    y_test = pd.read_csv(input_labels_path).squeeze()
    
    # --- Prediction and Evaluation ---
    
    # 3. Generate predictions on the test set
    y_pred = model.predict(X_test)
    
    # 4. Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])
    
    # --- Save Evaluation Report ---
    
    # 5. Create a clean, readable report string
    evaluation_output = f"""
    --- Model Evaluation Report ---
    
    Model: RandomForestClassifier
    Test Set Size: {len(X_test)} samples
    
    Overall Accuracy: {accuracy:.4f}
    
    Classification Report:
    {report}
    """
    
    # 6. Write the report to the output file path Tangle manages
    with open(output_report_path, 'w') as f:
        f.write(evaluation_output)
        
    print(f"Evaluation complete. Report saved to {output_report_path}")

except Exception as e:
    print(f"An error occurred during evaluation: {e}")
    sys.exit(1)
