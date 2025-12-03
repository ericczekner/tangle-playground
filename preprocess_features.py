import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys

# Tangle passes input/output file paths as command-line arguments.
# Usage: python preprocess_features.py <input_raw_data_path> <output_features_path> <output_labels_path>

input_path = sys.argv[1]
output_features_path = sys.argv[2]
output_labels_path = sys.argv[3]

try:
    data = pd.read_csv(input_path)
    
    # --- Data Cleaning and Type Conversion ---
    # Convert 'TotalCharges' to numeric, replacing errors (spaces/blanks) with NaN, then fill NaN with 0
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce').fillna(0)
    
    # --- Feature and Target Separation ---
    # The target variable is 'Churn'
    target = data['Churn']
    features = data.drop('Churn', axis=1)
    
    # --- Encoding Categorical Features ---
    # 1. Binary Encoding (Yes/No columns)
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                   'StreamingMovies', 'PaperlessBilling']
    for col in binary_cols:
        features[col] = features[col].replace({'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0})
        
    # 2. One-Hot Encoding (Multi-class categorical columns)
    categorical_cols = ['gender', 'Contract', 'PaymentMethod', 'InternetService']
    features = pd.get_dummies(features, columns=categorical_cols, drop_first=True)
    
    # --- Target (Label) Encoding ---
    # Convert 'Yes'/'No' Churn labels to 1/0
    le = LabelEncoder()
    target_encoded = le.fit_transform(target)
    
    # --- Save Processed Data ---
    features.to_csv(output_features_path, index=False)
    pd.Series(target_encoded).to_csv(output_labels_path, index=False, header=['Churn'])
    
    print(f"Features saved to {output_features_path} and labels to {output_labels_path}")

except Exception as e:
    print(f"An error occurred during preprocessing: {e}")
    sys.exit(1)
