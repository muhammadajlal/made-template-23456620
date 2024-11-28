import os
import pandas as pd
import requests
import subprocess
from pathlib import Path

OUTPUT_DIR = "./data" 

# Expected columns for each output file
EXPECTED_COLUMNS = {
    "merged_macroeconomic_dataset": ["DATE", "CPI", "Inflation", "MORTGAGE30US_MonthlyAvg", "MORTGAGE15US_MonthlyAvg", "RDP_Income"],
    "merged_housing_dataset": ["DATE", "Region", "SF_HomePrice", "All_HomePrice", "SF_RentalPrice", "All_RentalPrice", "SF_RentalDemand", "All_RentalDemand", "H_Mkt_HeatIndex"],
    "final_merged_dataset": ["DATE", "Region", "SF_HomePrice", "All_HomePrice", "SF_RentalPrice", "All_RentalPrice", "SF_RentalDemand", "All_RentalDemand", "H_Mkt_HeatIndex",
                            "CPI", "Inflation", "MORTGAGE30US_MonthlyAvg", "MORTGAGE15US_MonthlyAvg", "RDP_Income"],
}

# Data sources to be tested
DATA_SOURCES = {
    "macroeconomic": [
        {"name": "CPI", "url": "https://tinyurl.com/CPIUcsv"},
        {"name": "Interest Rates (30-Year Fixed Mortgage)", "url": "https://tinyurl.com/mortgagecsv"},
        {"name": "Interest Rates (15-Year Fixed Mortgage)", "url": "https://tinyurl.com/15mortgagecsv"},
        {"name": "Real Disposable Personal Income", "url": "https://tinyurl.com/RDPIcsv"},
    ],
    "housing": [
        {"name": "Single Family Home Prices", "url": "https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv?t=1730644370"},
        {"name": "All Hometypes Combined Prices", "url": "https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv?t=1730644370"},
        {"name": "Single Family Rental Prices", "url": "https://files.zillowstatic.com/research/public_csvs/zori/Metro_zori_uc_sfr_sm_month.csv?t=1730644371"},
        {"name": "All Hometypes Rental Prices", "url": "https://files.zillowstatic.com/research/public_csvs/zori/Metro_zori_uc_sfrcondomfr_sm_sa_month.csv?t=1730644371"},
        {"name": "Single Family Rental Home Demand", "url": "https://files.zillowstatic.com/research/public_csvs/zordi/Metro_zordi_uc_sfr_month.csv?t=1730644371"},
        {"name": "All Hometypes Rental Home Demand", "url": " https://files.zillowstatic.com/research/public_csvs/zordi/Metro_zordi_uc_sfrcondomfr_month.csv?t=1730644371"},
        {"name": "All Hometypes Housing Market Heat Index", "url": "https://files.zillowstatic.com/research/public_csvs/market_temp_index/Metro_market_temp_index_uc_sfrcondo_month.csv?t=1730644371"},
    ]
}

# Data Source Accessibility Test
def test_datasource_access(data_sources):
    print("Test 1/8: Testing Data Source Accessibility...")
    inaccessible_sources = []

    for source_type, datasets in data_sources.items():
        for dataset in datasets:
            url = dataset["url"]
            name = dataset["name"]
            try:
                response = requests.get(url)  # HEAD request for faster response
                if response.status_code != 200:
                    inaccessible_sources.append((source_type, name, url))
            except requests.exceptions.RequestException as e:
                inaccessible_sources.append((source_type, name, url, str(e)))

    # Report results
    if not inaccessible_sources:
        print(f"All data sources are accessible and downloadable. Test Passed!\n{'_'*150}")
    else:
        print(f"The following data sources are not accessible:")
        for item in inaccessible_sources:
            if len(item) == 3: 
                print(f"Type: {item[0]}, Name: {item[1]}, URL: {item[2]}")
            else:
                print(f"Type: {item[0]}, Name: {item[1]}, URL: {item[2]}, Error: {item[3]}\n{'_'*150}")

# Pipeline Execution Test
def test_pipeline_execution():
    """
    Runs the entire data pipeline using `requirements.txt` and `pipeline.py`.
    """
    print("Test 2/8: Executing pipeline...")

    try:
        # Install requirements
        subprocess.run(["python", "-m", "pip", "install", "-r", "./project/requirements.txt", "--no-warn-script-location"], check=True)

        # Run pipeline.py
        subprocess.run(["python", "./project/pipeline.py"], check=True)
    except subprocess.CalledProcessError as e:
        raise AssertionError(f"Pipeline execution failed: {e}")

    print(f"Pipeline executed successfully. Test Passed!\n{'_'*150}")

    
# Intermediate Data Files Existence Test
def test_interemdiate_data_files_exist():
    print("Test 3/8: Looking for the intermediate datatsets in the data directory...")
    required_files = [
        "merged_macroeconomic_dataset.csv", 
        "merged_housing_dataset.csv"
    ]
    for file in required_files:
        file_path = os.path.join(OUTPUT_DIR, file)
        assert os.path.exists(file_path), f"Required file '{file}' not found in {OUTPUT_DIR}.\n{'_'*150}"
    print(f"Found both {required_files} files. Test passed!\n{'_'*150}")

# Final Data File Existence Test
def test_final_data_file_exist():
    print("Test 4/8: Looking for the final datatset in the data directory...")
    required_files = [
        "final_merged_dataset.csv"
    ]
    for file in required_files:
        file_path = os.path.join(OUTPUT_DIR, file)
        assert os.path.exists(file_path), f"Required file '{file}' not found in {OUTPUT_DIR}.\n{'_'*150}"
    print(f"Found the {required_files} file. Test passed!\n{'_'*150}")

# Missing Values Test
def test_NaN_values_in_final_dataset(file_path):
    print("Test 8/8: Checking for missing values in the final dataset...")
    
    # Check if the file exists first
    if not os.path.exists(file_path):
        raise AssertionError(f"Required file 'final_merged_dataset.csv' not found in {OUTPUT_DIR}.")
    
    # Load the dataset into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Check for any missing or NaN values
    missing_values = df.isna().sum().sum()
    
    # If there are any missing values, raise an error
    if missing_values > 0:
        raise AssertionError(f"Found {missing_values} missing values (NaN) in 'final_merged_dataset.csv'.")
    
    print(f"No missing values found in 'final_merged_dataset.csv'. Test passed!\n{'_'*150}")

import pandas as pd

# Datatype test for the final dataset
def test_column_data_types_in_final_dataset(file_path):
    print("Test 7/8: Testing data types for all columns in the 'final_merged_dataset'...")

    # Define the expected columns and their expected data types
    column_data_types = {
        "CPI": "numeric",
        "Inflation": "numeric",
        "MORTGAGE30US_MonthlyAvg": "numeric",
        "MORTGAGE15US_MonthlyAvg": "numeric",
        "RDP_Income": "numeric",
        "DATE": "object",
        "Region": "object",
        "SF_HomePrice": "numeric",
        "All_HomePrice": "numeric",
        "SF_RentalPrice": "numeric",
        "All_RentalPrice": "numeric",
        "SF_RentalDemand": "numeric",
        "All_RentalDemand": "numeric",
        "H_Mkt_HeatIndex": "numeric",
    }

    # Load your final merged dataset (replace with your actual file path)
    final_df = pd.read_csv(file_path)

    # Check each column
    for column, expected_type in column_data_types.items():
        if column in final_df.columns:
            # Validate if the column contains numeric data (either int or float)
            if expected_type == "numeric":
                if not pd.api.types.is_numeric_dtype(final_df[column]):
                    raise AssertionError(f"Column '{column}' is not numeric. Please check the data.")
                else:
                    print(f"Column '{column}' is numeric. Test passed!")

            elif expected_type == "object":
                # Validate if the column contains object (string or categorical) data
                if not pd.api.types.is_object_dtype(final_df[column]):
                    raise AssertionError(f"Column '{column}' is not of type 'object'. Please check the data.")
                else:
                    print(f"Column '{column}' is of type 'object'. Test passed!")
        else:
            raise AssertionError(f"Column '{column}' is missing from the dataset.")
    print(f"All data type checks passed for the specified columns.\n{'_'*150}")

# Data Validation Test
def test_validate_non_empty(file_path):
    """Check if the file is non-empty."""
    print(f"Test 5/8: Checking if files are empty for {file_path}....")
    df = pd.read_csv(file_path)
    if df.empty:
        raise AssertionError(f"The following output file is empty: {file_path}\n{'_'*150}")
    print(f"The following output file is non-empty: {file_path}. Test Passed!\n{'_'*150}")

# Data Validation Test
def test_validate_columns(file_path, expected_columns):
    """Check if the file has the expected columns."""
    print(f"Test 6/8: Validating columns for {file_path}...")
    df = pd.read_csv(file_path)
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        raise AssertionError(f"Missing columns in {file_path}: {missing_columns}\n{'_'*150}")
    print(f"Columns validated successfully for {file_path}. Test Passed!\n{'_'*150}")


def main():
    # Step 1: Test data source accessibility
    test_datasource_access(DATA_SOURCES)

    # Step 2: Run the pipeline 
    test_pipeline_execution()

    # Step 3: Validate intermediate datasets
    test_interemdiate_data_files_exist()

    # Step 4: Validate final dataset existence
    test_final_data_file_exist()

    # Step 5: Validate intermediate datasets (non-empty and column checks)
    intermediate_datasets = {
        "merged_macroeconomic_dataset": EXPECTED_COLUMNS["merged_macroeconomic_dataset"],
        "merged_housing_dataset": EXPECTED_COLUMNS["merged_housing_dataset"],
    }
    for file_key, expected_columns in intermediate_datasets.items():
        file_path = os.path.join(OUTPUT_DIR, f"{file_key}.csv")
        test_validate_non_empty(file_path)
        test_validate_columns(file_path, expected_columns)

    # Step 56 Validate final dataset (non-empty, column name and datatype checks, NaN values check)
    final_dataset_path = os.path.join(OUTPUT_DIR, "final_merged_dataset.csv")
    test_validate_non_empty(final_dataset_path)
    test_validate_columns(final_dataset_path, EXPECTED_COLUMNS["final_merged_dataset"])
    test_column_data_types_in_final_dataset(final_dataset_path)
    test_NaN_values_in_final_dataset(final_dataset_path)

    print("\nAll tests passed successfully!\n Earned Marks: 100%\n" + "_" * 150)

if __name__ == "__main__":
    main()
