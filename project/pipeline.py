import pandas as pd
import os
import requests
from io import StringIO

# Data Sources Dictionary
DATA_SOURCES = {
    "macroeconomic": [
        {"name": "CPI", "url": "https://tinyurl.com/CPIUcsv"},
        {"name": "Interest Rates (30-Year Fixed Mortgage)", "url": "https://tinyurl.com/mortgagecsv"},
        {"name": "Interest Rates (15-Year Fixed Mortgage)", "url": "https://tinyurl.com/15mortgagecsv"},
        {"name": "Real Disposable Personal Income", "url": "https://tinyurl.com/RDPIcsv"},
        # Add additional macroeconomic URLs here
    ],
    "housing": [
        {"name": "Single Family Home Prices", "url": "https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv?t=1730644370"},
        {"name": "All Hometypes Combined Prices", "url": "https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv?t=1730644370"},
        {"name": "Single Family Rental Prices", "url": "https://files.zillowstatic.com/research/public_csvs/zori/Metro_zori_uc_sfr_sm_month.csv?t=1730644371"},
        {"name": "All Hometypes Rental Prices", "url": "https://files.zillowstatic.com/research/public_csvs/zori/Metro_zori_uc_sfrcondomfr_sm_sa_month.csv?t=1730644371"},
        {"name": "Single Family Rental Home Demand", "url": "https://files.zillowstatic.com/research/public_csvs/zordi/Metro_zordi_uc_sfr_month.csv?t=1730644371"},
        {"name": "All Hometypes Rental Home Demand", "url": " https://files.zillowstatic.com/research/public_csvs/zordi/Metro_zordi_uc_sfrcondomfr_month.csv?t=1730644371"},
        {"name": "All Hometypes Housing Market Heat Index", "url": "https://files.zillowstatic.com/research/public_csvs/market_temp_index/Metro_market_temp_index_uc_sfrcondo_month.csv?t=1730644371"},

        #{"name": "Shared/Cooperative Apartment Price", "url": "https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_condo_tier_0.33_0.67_sm_sa_month.csv?t=1730644370"},
        #{"name": "Shared/Cooperative Apartment Rental Apartment Demand", "url": "https://files.zillowstatic.com/research/public_csvs/zordi/Metro_zordi_uc_condo_month.csv?t=1730644371"},
    ]
}

OUTPUT_DIR = "./data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dataset.csv")
macroeconomic_output_file = os.path.join(OUTPUT_DIR, "merged_macroeconomic_dataset.csv")
housing_output_file = os.path.join(OUTPUT_DIR, "merged_housing_dataset.csv")
final_output_file = os.path.join(OUTPUT_DIR, "final_merged_dataset.csv")


# Extraction of data
def extract_data(url: str) -> pd.DataFrame:
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = pd.read_csv(StringIO(response.text))
        print(f"Data extraction successful from {url}")
        return data
    except Exception as e:
        print(f"Error in data extraction from {url}: {e}")
        return pd.DataFrame()

# Tranformation of Data
# 1. CPI (Consumer Price Index) Data Transformation
def transform_cpi_data(data: pd.DataFrame) -> pd.DataFrame:
    try:
        # Feature Enginnering (Generating YoY Percentage Change)
        YoY_Percentage_Change_Inflation = data['CPIAUCNS'].pct_change(periods=12) * 100
        YoY_Percentage_Change_Inflation = YoY_Percentage_Change_Inflation.round(1)
        data.insert(2, 'YoY_%Change_CPI(Inflation)', YoY_Percentage_Change_Inflation)
        # Filtering out the data for the last 10 years
        data['DATE'] = pd.to_datetime(data['DATE'])  # Convert 'DATE' column to datetime objects
        data = data[data['DATE'] >= '2014-01-01']
        #data.columns = [col.lower().replace(' ', '_') for col in data.columns]
        #data = data.dropna()

       
        # Renaming column for better clarity
        data = data.rename(columns={'CPIAUCNS': 'CPI'})
        print("CPI Data transformation successful.")
        return data
    except Exception as e:
        print(f"Error in data transformation: {e}")
        return pd.DataFrame()

# 2. Interest Rates (30-Year Fixed Mortgage) Data Transformation
def transform_interest_rate_data(data: pd.DataFrame, rate_type: str = '30Y') -> pd.DataFrame:
    try:
        # Select the correct column based on the rate type
        rate_column = 'MORTGAGE30US' if rate_type == '30Y' else 'MORTGAGE15US'
        output_column = f"{rate_column}_MonthlyAvg"
        
        # Convert DATE column to datetime and extract Month and Year
        data['DATE'] = pd.to_datetime(data['DATE'])
        data['Month'] = data['DATE'].dt.month
        data['Year'] = data['DATE'].dt.year

        # Calculate the monthly average
        monthly_avg = data.groupby(['Year', 'Month'], as_index=False)[rate_column].mean()
        monthly_avg['DATE'] = pd.to_datetime(monthly_avg[['Year', 'Month']].assign(DAY=1))
        
        # Rename the column for clarity
        monthly_avg = monthly_avg.rename(columns={rate_column: output_column})

        # Filter for the last 10 years
        monthly_avg = monthly_avg[monthly_avg['DATE'] >= '2014-01-01']

        print(f"Interest Rates Data ({rate_type}) transformation successful.")
        return monthly_avg

    except Exception as e:
        print(f"Error in data transformation: {e}")
        return pd.DataFrame()


# 3. Real Disposable Income Data Transformation
def transform_disposable_income_data(data: pd.DataFrame) -> pd.DataFrame:
    try:
        # Filtering out the data for the last 10 years
        data['DATE'] = pd.to_datetime(data['DATE'])
        data = data[data['DATE'] >= '2014-01-01']
        # Renaming column for better clarity
        data = data.rename(columns={'DSPIC96': 'RDP_Income'})
        print("Disposable Income Data transformation successful.")
        return data
    except Exception as e:
        print(f"Error in data transformation: {e}")
        return pd.DataFrame()

# 4. Zillow Home Price/Rental/Demand Data Transformation    
def transform_zillow_home_price_and_rental_data(data: pd.DataFrame, data_type: str) -> pd.DataFrame:
    try:
        # Filtering top 10 biggest Metropolitan Areas based on size rank
        data = data[data['SizeRank'] <= 10]
        # Setting 'RegionName' as the index so that we can melt the dataframe as we need
        data = data.set_index('RegionName')
        # Dropping unnecessary columns 
        data = data.drop(columns=['RegionID', 'SizeRank', 'RegionType', 'StateName'])
        # Transposing the data into long format
        df_transposed = data.T  
        df_transposed = df_transposed.reset_index() 
        df_transposed.columns = ['DATE'] + data.index.tolist() 
        data = pd.melt(df_transposed, id_vars='DATE', var_name='Region', value_name = data_type) 
        # Filtering out the data for the last 10 years
        data['DATE'] = pd.to_datetime(data['DATE'])
        data = data[data['DATE'] >= '2014-01-01']

        print(f"{data_type} Data transformation successful.")
        return data
    except Exception as e:
        print(f"Error in data transformation: {e}")
        return pd.DataFrame()


# Transformation Dispatcher
TRANSFORMATION_FUNCTIONS = {
    "CPI": transform_cpi_data,
    "Interest Rates (30-Year Fixed Mortgage)": lambda data: transform_interest_rate_data(data, rate_type='30Y'),
    "Interest Rates (15-Year Fixed Mortgage)": lambda data: transform_interest_rate_data(data, rate_type='15Y'),
    "Real Disposable Personal Income": transform_disposable_income_data,
    "Single Family Home Prices": lambda data: transform_zillow_home_price_and_rental_data(data, data_type="SF_HomePrice"),
    "All Hometypes Combined Prices": lambda data: transform_zillow_home_price_and_rental_data(data, data_type="All_HomePrice"),
    "Single Family Rental Prices": lambda data: transform_zillow_home_price_and_rental_data(data, data_type="SF_RentalPrice"),
    "All Hometypes Rental Prices": lambda data: transform_zillow_home_price_and_rental_data(data, data_type="All_RentalPrice"),
    "Single Family Rental Home Demand": lambda data: transform_zillow_home_price_and_rental_data(data, data_type="SF_RentalDemand"),
    "All Hometypes Rental Home Demand": lambda data: transform_zillow_home_price_and_rental_data(data, data_type="All_RentalDemand"),
    "All Hometypes Housing Market Heat Index": lambda data: transform_zillow_home_price_and_rental_data(data, data_type="HeatIndex")
    # Add other datasets and their respective transformation functions here
}

# Main Transformation Dispatcher
def transform_data(data: pd.DataFrame, name: str) -> pd.DataFrame:
    try:
        # Call the appropriate transformation function based on dataset name
        if name in TRANSFORMATION_FUNCTIONS:
            data = TRANSFORMATION_FUNCTIONS[name](data)
        else:
            print(f"No specific transformation function found for {name}. Using default transformation.")
            # Apply any general transformations here if needed, or return the data as is.
        
        return data
    except Exception as e:
        print(f"Error in data transformation for {name}: {e}")
        return pd.DataFrame()


# Loading Data
def load_data(data: pd.DataFrame, data_type: str, index: int, name: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_path = os.path.join(OUTPUT_DIR, f"{data_type}_data_{index}.csv")
    try:
        data.to_csv(file_path, index=False)
        print(f"{name} data loaded successfully into {file_path}\n{'-'*150}")
    except Exception as e:
        print(f"Error in data loading to {file_path}: {e}")

# Making a final merged dataset
def merge_n_clean_transformed_data(macroeconomic_transformed_data, housing_transformed_data):
    final_merged_data = {}

    try:
        
        macroeconomic_merged_data = macroeconomic_transformed_data["CPI"]
        macroeconomic_merged_data = macroeconomic_merged_data.merge(macroeconomic_transformed_data["Interest Rates (30-Year Fixed Mortgage)"][["DATE", "MORTGAGE30US_MonthlyAvg"]], on="DATE", how="inner")
        macroeconomic_merged_data = macroeconomic_merged_data.merge(macroeconomic_transformed_data["Interest Rates (15-Year Fixed Mortgage)"][["DATE", "MORTGAGE15US_MonthlyAvg"]], on="DATE", how="inner")
        macroeconomic_merged_data = macroeconomic_merged_data.merge(macroeconomic_transformed_data["Real Disposable Personal Income"], on="DATE", how="inner")
        
        housing_merged_data = housing_transformed_data["Single Family Home Prices"]
        housing_merged_data = housing_merged_data.merge(housing_transformed_data["All Hometypes Combined Prices"][["DATE", "Region", "All_HomePrice"]], on=["DATE", "Region"], how="left")
        housing_merged_data = housing_merged_data.merge(housing_transformed_data["Single Family Rental Prices"][["DATE", "Region", "SF_RentalPrice"]], on=["DATE", "Region"], how="left")
        housing_merged_data = housing_merged_data.merge(housing_transformed_data["All Hometypes Rental Prices"][["DATE", "Region", "All_RentalPrice"]], on=["DATE", "Region"], how="left")
        housing_merged_data = housing_merged_data.merge(housing_transformed_data["Single Family Rental Home Demand"][["DATE", "Region", "SF_RentalDemand"]], on=["DATE", "Region"], how="left")
        housing_merged_data = housing_merged_data.merge(housing_transformed_data["All Hometypes Rental Home Demand"][["DATE", "Region", "All_RentalDemand"]], on=["DATE", "Region"], how="left")
        housing_merged_data = housing_merged_data.merge(housing_transformed_data["All Hometypes Housing Market Heat Index"][["DATE", "Region", "HeatIndex"]], on=["DATE", "Region"], how="left")    
        
        # Save the merged datasets to separate files
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        if macroeconomic_merged_data is not None:
            macroeconomic_merged_data.to_csv(macroeconomic_output_file, index=False)
            print(f"1st Merge operation Successfull! Merged Macroeconomic dataset saved to {macroeconomic_output_file}\n{'-'*150}")
        else:    
            print("Cannot save merged macroeconomic dataset. Error in merging individual macroeconomic datasets.\n{'-'*100}")
            return None 
    
        # Save the merged housing data to a separate file
        if housing_merged_data is not None:
            housing_merged_data.to_csv(housing_output_file, index=False)
            print(f"2nd Merge Operation Successfull! Merged Housing dataset saved to {housing_output_file}\n{'-'*150}")
        else:
            print("Cannot save merged housing dataset. Error in merging individual housing datasets.\n{'-'*100}")
            return None

        
        # Convert both DATE columns to the first day of the month (because housing data DATE column had last day of the month)
        housing_merged_data["DATE"] = pd.to_datetime(housing_merged_data["DATE"]).dt.to_period("M").dt.start_time
        macroeconomic_merged_data["DATE"] = pd.to_datetime(macroeconomic_merged_data["DATE"]).dt.to_period("M").dt.start_time

        # Merge the macroeconomic and housing datasets and save the final dataset
        final_merged_data = housing_merged_data.merge(macroeconomic_merged_data, on="DATE", how="left")
        if final_merged_data is not None:
            final_merged_data.to_csv(final_output_file, index=False)
            print(f"Final Merge Operation Successfull! Final merged dataset saved to {final_output_file}\n{'-'*150}")
            return final_merged_data
        else:
            print("Cannot make a final dataset! Error in merging both transformed datasets.\n{'-'*100}")
            

    except Exception as e:
        print(f"Error in merging datasets: {e}")
        return None
    

# Main Function of our ETL Pipeline
def main():
    macroeconomic_transformed_data = {}
    housing_transformed_data = {}

    for data_type, datasets in DATA_SOURCES.items():
        for idx, dataset in enumerate(datasets, start=1):
            name = dataset["name"]
            url = dataset["url"]
            print(f"Processing {name} ({data_type} dataset {idx})")
            
            # Extraction
            data = extract_data(url)
            if not data.empty:

                transformed_data = transform_data(data, name)
            
                # Store transformed data
                if data_type == "macroeconomic":
                    macroeconomic_transformed_data[name] = transformed_data
                elif data_type == "housing":
                    housing_transformed_data[name] = transformed_data
            
            # Load transformed data
            load_data(transformed_data, data_type, idx, name)

    # Merging transformed data tables into single dataset
    final_merged_data = merge_n_clean_transformed_data(macroeconomic_transformed_data, housing_transformed_data)

    if final_merged_data is not None:
        print("ETL pipeline completed.")
    else:
        print("ETL pipeline failed during merging.")


# Entry point check
if __name__ == "__main__":
    main()
