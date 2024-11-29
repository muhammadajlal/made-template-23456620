import os
import pandas as pd
import requests
import subprocess
from io import StringIO
from pandas.testing import assert_frame_equal
from pipeline import transform_interest_rate_data
from pipeline import transform_cpi_data
from pipeline import transform_disposable_income_data

OUTPUT_DIR = "./data" 

# Define final mock data
MOCK_DATA = """
DATE,Region,SF_HomePrice,All_HomePrice,SF_RentalPrice,All_RentalPrice,SF_RentalDemand,All_RentalDemand,H_Mkt_HeatIndex,CPI,Inflation,MORTGAGE30US_MonthlyAvg,MORTGAGE15US_MonthlyAvg,RDP_Income
2015-01-01,United States,184535.969904,186014.926817,1263.444318,1230.465523,100.276460,53.473359,50.122169,233.707,-0.1,3.6700,2.9850,13797.7
2015-02-01,Ney York,185330.314211,186816.309447,1269.895092,1236.076292,132.205287,82.834704,51.441924,234.722,-0.0,3.7100,3.0075,13848.0
2015-03-01,Los Angeles,186140.654880,187630.517995,1278.469237,1242.136870,	149.610668,103.671933,52.091247,236.119,-0.1,3.7700,3.0400,13811.3
2015-04-01,Boston,187031.186424,188520.789452,1288.647487,1247.193310,149.610668,108.948715,52.091247,236.599,-0.2,3.6720,2.9420,13842.0
"""
# Load mock data into a DataFrame
mock_df = pd.read_csv(StringIO(MOCK_DATA))

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
    print("Integration Test 1: Testing Accessibility of 11 different data sources...")
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
    print("System Test 1/7: Executing pipeline...")

    try:
        # Install requirements
        subprocess.run(["python", "-m", "pip", "install", "-r", "./project/requirements.txt", "--no-warn-script-location"], check=True)

        # Run pipeline.py
        subprocess.run(["python", "./project/pipeline.py"], check=True)
    except subprocess.CalledProcessError as e:
        raise AssertionError(f"Pipeline execution failed: {e}")

    print(f"Test Passed: Pipeline executed successfully!\n{'_'*150}")

    
# Intermediate Data Files Existence Test
def test_interemdiate_data_files_exist():
    print("System Test 2/7: Looking for the intermediate datatsets in the data directory...")
    required_files = [
        "merged_macroeconomic_dataset.csv", 
        "merged_housing_dataset.csv"
    ]
    for file in required_files:
        file_path = os.path.join(OUTPUT_DIR, file)
        assert os.path.exists(file_path), f"Required file '{file}' not found in {OUTPUT_DIR}.\n{'_'*150}"
    print(f"Test Passed: Found both {required_files} files!\n{'_'*150}")

# Final Data File Existence Test
def test_final_data_file_exist():
    print("System Test 3/7: Looking for the final datatset in the data directory...")
    required_files = [
        "final_merged_dataset.csv"
    ]
    for file in required_files:
        file_path = os.path.join(OUTPUT_DIR, file)
        assert os.path.exists(file_path), f"Required file '{file}' not found in {OUTPUT_DIR}.\n{'_'*150}"
    print(f"Test Passed: Found the {required_files} file!\n{'_'*150}")

# Missing Values Test
def test_NaN_values_in_final_dataset_data_directory(file_path):
    print("System Test 7/7: Checking for missing values in the final dataset...")
    
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
    
    print(f"test Passed: No missing values found in 'final_merged_dataset.csv'!")
    print()
    print()

# Datatype test for the final dataset
def test_column_data_types_in_final_dataset_data_directory(file_path):
    print("System Test 6.1/7: Testing data types for all columns in the 'final_merged_dataset'...")

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
    print(f"test Passed: All data type checks passed for the specified columns.\n{'_'*150}")

    # Datatype test for the final dataset
def test_column_data_types_in_final_dataset_mock_data(data: pd.DataFrame):
    print("Test 6.2/7: Testing data types for all columns in the 'final mock data'...")

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

    # Check each column
    for column, expected_type in column_data_types.items():
        if column in data.columns:
            # Validate if the column contains numeric data (either int or float)
            if expected_type == "numeric":
                if not pd.api.types.is_numeric_dtype(data[column]):
                    raise AssertionError(f"Column '{column}' is not numeric. Please check the data.")
                else:
                    print(f"Column '{column}' is numeric. Test passed!")

            elif expected_type == "object":
                # Validate if the column contains object (string or categorical) data
                if not pd.api.types.is_object_dtype(data[column]):
                    raise AssertionError(f"Column '{column}' is not of type 'object'. Please check the data.")
                else:
                    print(f"Column '{column}' is of type 'object'. Test passed!")
        else:
            raise AssertionError(f"Column '{column}' is missing from the dataset.")
    print(f"test Passed: All data type checks passed for the specified columns in 'final mock data'.\n{'_'*150}")

    

# Data Validation Test
def test_validate_non_empty_data_directory(file_path):
    """Check if the file is non-empty."""
    print(f"System Test 4/7: Checking if files are empty for {file_path}....")
    df = pd.read_csv(file_path)
    if df.empty:
        raise AssertionError(f"The following output file is empty: {file_path}\n{'_'*150}")
    print(f"The following output file is non-empty: {file_path}. Test Passed!\n{'_'*150}")

# Data Validation Test
def test_validate_columns_data_directory(file_path, expected_columns):
    """Check if the file has the expected columns."""
    print(f"System Test 5.1/7: Validating columns for {file_path}...")
    df = pd.read_csv(file_path)
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        raise AssertionError(f"Missing columns in {file_path}: {missing_columns}\n{'_'*150}")
    print(f" Test Passed: Columns validated successfully for {file_path}!\n{'_'*150}")

def test_validate_columns_mock_data(data: pd.DataFrame, expected_columns):
    """Check if the file has the expected columns."""
    print(f"System Test 5.2./7: Validating columns for 'final mock data'...")
    df = data
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        raise AssertionError(f"Missing columns in 'final mock data': {missing_columns}\n{'_'*150}")
    print(f" Test Passed: Columns validated successfully for 'final mock data'!\n{'_'*150}")

# Some unit tests
def test_transform_interest_rate_data():
    print("Unit Test 1: Testing the transform_interest_rate_data function...")
    # Mock input data
    mock_data = pd.DataFrame({
        'DATE': ['2014-12-31', '2015-01-01', '2015-01-15', '2015-02-01'],
        'MORTGAGE30US': [3.9, 3.8, 3.7, 3.6],
        'MORTGAGE15US': [3.1, 3.0, 2.9, 2.8],
    })

    # Expected output for 30-year interest rates
    expected_output_30Y = pd.DataFrame({
        'Year': [2015, 2015],
        'Month': [1, 2],
        'MORTGAGE30US_MonthlyAvg': [3.75, 3.6],
        'DATE': pd.to_datetime(['2015-01-01', '2015-02-01']),
    }).reset_index(drop=True)  # Reset index to RangeIndex for consistency

    # Expected output for 15-year interest rates
    expected_output_15Y = pd.DataFrame({
        'Year': [2015, 2015],
        'Month': [1, 2],
        'MORTGAGE15US_MonthlyAvg': [2.95, 2.8],
        'DATE': pd.to_datetime(['2015-01-01', '2015-02-01']),
    }).reset_index(drop=True)  # Reset index to RangeIndex for consistency

    # Test for 30-year rates
    transformed_30Y = transform_interest_rate_data(mock_data, rate_type='30Y').reset_index(drop=True)
    pd.testing.assert_frame_equal(transformed_30Y, expected_output_30Y, check_dtype=False)

    # Test for 15-year rates
    transformed_15Y = transform_interest_rate_data(mock_data, rate_type='15Y').reset_index(drop=True)
    pd.testing.assert_frame_equal(transformed_15Y, expected_output_15Y, check_dtype=False)
    print(f"Test Passed: transform_interest_rate_data function works as expected!\n{'_'*150}")


def test_transform_cpi_data(): 
    print("Unit Test 2: Testing the transform_cpi_data function...") 
        # Sample data to test the function, including monthly CPI data for two years
    mock_cpi_df = pd.DataFrame({
            'DATE': pd.date_range(start='2013-01-01', end='2015-12-01', freq='MS'),
            'CPIAUCNS': [
                230.28, 230.68, 231.09, 231.66, 231.86, 232.14, 232.95, 233.70, 233.75, 234.15, 234.72, 235.16,
                234.81, 234.89, 235.23, 235.64, 235.72, 235.98, 236.12, 236.68, 236.98, 237.12, 237.65, 238.00,
                236.92, 237.01, 237.11, 237.24, 237.46, 237.65, 238.00, 238.50, 238.60, 238.70, 238.82, 239.10
            ]
        })

    expected_data = pd.DataFrame({
            'DATE': pd.to_datetime([
                '2015-01-01', '2015-02-01', '2015-03-01', '2015-04-01', '2015-05-01', '2015-06-01', '2015-07-01',
                '2015-08-01', '2015-09-01', '2015-10-01', '2015-11-01', '2015-12-01'
            ]),
            'CPI': [
                236.92, 237.01, 237.11, 237.24, 237.46, 237.65, 238.00, 238.50, 238.60, 238.70, 238.82, 239.10
            ],
            'Inflation': [
                0.9, 0.9, 0.8, 0.7, 0.7, 0.7, 0.8, 0.8, 0.7, 0.7, 0.5, 0.5
            ]
        }).set_index('DATE')

        # Transform the data using the function
    transformed_data = transform_cpi_data(mock_cpi_df)

        # Set the index to DATE for comparison
    transformed_data = transformed_data.set_index('DATE').loc['2015-01-01':]

        # Check the output
    pd.testing.assert_frame_equal(transformed_data, expected_data, check_dtype=False)
    print(f"Test Passed: transform_cpi_data function works as expected!\n{'_'*150}")

# Unit test Nrr. 3 for the function transform_disposable_income_data
def test_transform_disposable_income_data():
    print("Unit Test 3: Testing the transform_disposable_income_data function...")
    # Sample data to test the function
    mock_rdp_data = pd.DataFrame({
        'DATE': ['2014-01-01', '2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01'],
        'DSPIC96': [10000, 10500, 11000, 11500, 12000]
    })

    # Expected data after transformation
    expected_data = pd.DataFrame({
        'DATE': pd.to_datetime(['2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01']),
        'RDP_Income': [10500, 11000, 11500, 12000]
    }).reset_index(drop=True)

    # Transform the data using the function
    transformed_data = transform_disposable_income_data(mock_rdp_data).reset_index(drop=True)

    # Check if the transformation is as expected
    pd.testing.assert_frame_equal(transformed_data, expected_data, check_dtype=False)

    print(f"Test passed: transform_disposable_income_data function works as expected!\n{'_'*150}")


def main():
    # Step 1: Test data source accessibility
    print()
    print()
    print(f"{'+'*150}")
    print(f"Starting Integrations Tests...")
    print(f"{'+'*150}")
    test_datasource_access(DATA_SOURCES)
    print()
    print(f"{'+'*150}")
    print(f"Starting System Tests...")
    print(f"{'+'*150}")
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
        test_validate_non_empty_data_directory(file_path)
        test_validate_columns_data_directory(file_path, expected_columns)

    # Step 56 Validate final dataset (non-empty, column name and datatype checks, NaN values check)
    final_dataset_path = os.path.join(OUTPUT_DIR, "final_merged_dataset.csv")
    test_validate_non_empty_data_directory(final_dataset_path)
    test_validate_columns_data_directory(final_dataset_path, EXPECTED_COLUMNS["final_merged_dataset"])
    test_validate_columns_mock_data(mock_df, EXPECTED_COLUMNS["final_merged_dataset"])
    test_column_data_types_in_final_dataset_data_directory(final_dataset_path)
    test_column_data_types_in_final_dataset_mock_data(mock_df)
    test_NaN_values_in_final_dataset_data_directory(final_dataset_path)
    print(f"{'+'*150}")
    print(f"Starting unit tests...")
    print(f"{'+'*150}")
    test_transform_interest_rate_data()
    test_transform_cpi_data()
    test_transform_disposable_income_data()

    print("\nAll tests passed successfully!\n Earned Marks: 100%\n" + "_" * 150)

if __name__ == "__main__":
    main()
