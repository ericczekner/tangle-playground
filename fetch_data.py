# fetch_data.py
import pandas as pd

# Define the URL
DATA_URL = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'

# Load the data
data = pd.read_csv(DATA_URL)

# Save the raw data to a local file path (Tangle will manage this output path)
data.to_csv('raw_data.csv', index=False)
print("Successfully loaded and saved raw_data.csv")
