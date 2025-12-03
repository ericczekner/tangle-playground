import pandas as pd
import sys

# The URL for the raw Telco Customer Churn dataset
DATA_URL = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'

# Tangle passes output file paths as command-line arguments.
# The path for the raw data output is the first argument.
output_path = sys.argv[1] 

try:
    # 1. Fetch the data
    data = pd.read_csv(DATA_URL)
    
    # 2. Drop the 'customerID' column as it's not a useful feature
    data = data.drop('customerID', axis=1)

    # 3. Save the raw data to the path Tangle expects
    data.to_csv(output_path, index=False)
    print(f"Successfully loaded and saved raw data to {output_path}")

except Exception as e:
    print(f"An error occurred during data fetching: {e}")
    sys.exit(1)
