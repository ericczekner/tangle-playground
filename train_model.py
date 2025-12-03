import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump # Used to save the model artifact
import sys

# Usage: python train_model.py <input_features_path> <input_labels_path> <output_model_path>

input_features_path = sys.argv[1]
input_labels_path = sys.argv[2]
output_model_path = sys.argv[3]

try:
    # --- Load Data ---
    X = pd.read_csv(input_features_path)
    y = pd.read_csv(input_labels_path).squeeze() # Use squeeze() to get Series for target
    
    # --- Split Data ---
    # Use a fixed random state for reproducibility
    # NOTE: The test set is split here, but only the indices are needed for evaluation. 
    # For a simple pipeline, we save the indices in a real-world scenario.
    # For this simple example, we will train on X_train and evaluate on X_test later.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- Model Training ---
    print("Starting Model Training...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model Training Complete.")
    
    # --- Save the Trained Model Artifact ---
    # In a production pipeline, you would also save the X_test/y_test data or indices 
    # to ensure the evaluation step uses the exact same test set.
    # For simplicity here, we save the trained model only.
    dump(model, output_model_path) 
    X_test.to_csv('X_test_for_eval.csv', index=False) # Save the test set for the evaluation component
    y_test.to_csv('y_test_for_eval.csv', index=False, header=['Churn']) # Save the test labels
    
    print(f"Trained model saved to {output_model_path}")

except Exception as e:
    print(f"An error occurred during training: {e}")
    sys.exit(1)
