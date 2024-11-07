import pandas as pd
import os
import requests
from io import StringIO

# Configurable constants
DATA_URL = "https://example.com/your-dataset.csv"  # Replace with the actual dataset URL
OUTPUT_DIR = "./data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dataset.csv")

# Step 1: Data Extraction
def extract_data(url: str) -> pd.DataFrame:
    """
    Extracts data from a URL and loads it into a DataFrame.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if request was successful
        data = pd.read_csv(StringIO(response.text))
        print("Data extraction successful.")
        return data
    except Exception as e:
        print(f"Error during data extraction: {e}")
        return pd.DataFrame()

# Step 2: Data Transformation
def transform_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the data by cleaning and preparing it.
    """
    try:
        # Example transformations: cleaning column names and dropping NA rows
        data.columns = [col.lower().replace(' ', '_') for col in data.columns]
        data = data.dropna()  # Drop rows with missing values
        print("Data transformation successful.")
        return data
    except Exception as e:
        print(f"Error during data transformation: {e}")
        return pd.DataFrame()

# Step 3: Save to Local Storage
def load_data(data: pd.DataFrame, filepath: str):
    """
    Saves the data to a CSV file in the /data directory.
    """
    try:
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)  # Create /data directory if it doesn't exist
        data.to_csv(filepath, index=False)
        print(f"Data successfully saved to {filepath}.")
    except Exception as e:
        print(f"Error during data loading: {e}")

# Main function
def main():
    print("Starting ETL pipeline...")
    
    # Step 1: Extract
    data = extract_data(DATA_URL)
    
    # Step 2: Transform
    if not data.empty:
        data = transform_data(data)
    
    # Step 3: Load
    if not data.empty:
        load_data(data, OUTPUT_FILE)
    
    print("ETL pipeline completed.")

# Entry point check
if __name__ == "__main__":
    main()
