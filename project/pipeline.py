import pandas as pd
import os
import requests
import time
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor

# Data Sources Dictionary
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

OUTPUT_DIR = "./data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dataset.csv")
macroeconomic_output_file = os.path.join(OUTPUT_DIR, "merged_macroeconomic_dataset.csv")
housing_output_file = os.path.join(OUTPUT_DIR, "merged_housing_dataset.csv")
final_output_file = os.path.join(OUTPUT_DIR, "final_merged_dataset.csv")


# Extraction of Data
def extract_data(url: str) -> pd.DataFrame:
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = pd.read_csv(StringIO(response.text))
            print(f"Data extraction successful from {url}")
            return data
        except Exception as e:
            print(f"Attempt {attempt} failed for {url}: {e}")
            if attempt < max_attempts:
                print("Retrying...")
                time.sleep(2)  # Wait 2 seconds before retrying
            else:
                print(f"Failed to extract data from {url} after {max_attempts} attempts.")
                return pd.DataFrame()

# Tranformation of Data
# 1. CPI (Consumer Price Index) Data Transformation
def transform_cpi_data(data: pd.DataFrame) -> pd.DataFrame:
    try:
        # Feature Enginnering (Generating YoY Percentage Change)
        YoY_Percentage_Change_Inflation = data['CPIAUCNS'].pct_change(periods=12) * 100
        YoY_Percentage_Change_Inflation = YoY_Percentage_Change_Inflation.round(1)
        data.insert(2, 'Inflation', YoY_Percentage_Change_Inflation)
        # Filtering out the data for the last 10 years
        data['DATE'] = pd.to_datetime(data['DATE'])  # Convert 'DATE' column to datetime objects
        data = data[data['DATE'] >= '2015-01-01']
        # Renaming column for better clarity
        data = data.rename(columns={'CPIAUCNS': 'CPI'})
        print("CPI Data transformation successful.")
        return data
    except Exception as e:
        print(f"Error in final_merged_data transformation: {e}")
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
        monthly_avg = monthly_avg[monthly_avg['DATE'] >= '2015-01-01']

        print(f"Interest Rates Data ({rate_type}) transformation successful.")
        return monthly_avg

    except Exception as e:
        print(f"Error in final_merged_data transformation: {e}")
        return pd.DataFrame()

# 3. Real Disposable Income Data Transformation
def transform_disposable_income_data(data: pd.DataFrame) -> pd.DataFrame:
    try:
        # Filtering out the data for the last 10 years
        data['DATE'] = pd.to_datetime(data['DATE'])
        data = data[data['DATE'] >= '2015-01-01']
        # Renaming column for better clarity
        data = data.rename(columns={'DSPIC96': 'RDP_Income'})
        print("Disposable Income Data transformation successful.")
        return data
    except Exception as e:
        print(f"Error in final_merged_data transformation: {e}")
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
        # Transposing the final_merged_data into long format
        df_transposed = data.T  
        df_transposed = df_transposed.reset_index() 
        df_transposed.columns = ['DATE'] + data.index.tolist() 
        data = pd.melt(df_transposed, id_vars='DATE', var_name='Region', value_name = data_type) 
        # Filtering out the data for the last 10 years
        data['DATE'] = pd.to_datetime(data['DATE'])
        data = data[data['DATE'] >= '2015-01-01']

        print(f"{data_type} Data transformation successful.")
        return data
    except Exception as e:
        print(f"Error in final_merged_data transformation: {e}")
        return pd.DataFrame()

# Transformation Dispatcher helper dictionary
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
    "All Hometypes Housing Market Heat Index": lambda data: transform_zillow_home_price_and_rental_data(data, data_type="H_Mkt_HeatIndex")

}

# Main Transformation Dispatcher
def transform_data(data: pd.DataFrame, name: str) -> pd.DataFrame:
    try:
        # Call the appropriate transformation function based on dataset name
        if name in TRANSFORMATION_FUNCTIONS:
            data = TRANSFORMATION_FUNCTIONS[name](data)
        else:
            print(f"No specific transformation function found for {name}. Using default transformation.")
            # Apply any general transformations here if needed, or return the final_merged_data as is.
        
        return data
    except Exception as e:
        print(f"Error in transformation for {name}: {e}")
        return pd.DataFrame()

# Loading Data
def load_data(data: pd.DataFrame, data_type: str, index: int, name: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_path = os.path.join(OUTPUT_DIR, f"{data_type}_data_{index}.csv")
    try:
        data.to_csv(file_path, index=False)
        print(f"{name} data loaded successfully into {file_path}\n{'-'*150}")
    except Exception as e:
        print(f"Error in loading {name} data to {file_path}: {e}")

# Regression Model to fill missing values in the final_merged_data
def Regress_Missing_Values(final_merged_data: pd.DataFrame, 
                           target_column: str, 
                           time_range_to_predict: tuple, 
                           predictors: list) -> pd.DataFrame:
    try:
        # Extract time range
        start_date, end_date = time_range_to_predict

        # Filter rows with complete target data for training
        train_data_filtered = final_merged_data.dropna(subset=[target_column])

        # Define X_train and y_train
        X_train = train_data_filtered[predictors]
        y_train = train_data_filtered[target_column]

        # Define the prediction time range
        predict_data = final_merged_data[(final_merged_data['DATE'] >= start_date) & 
                                         (final_merged_data['DATE'] <= end_date)]
        X_predict = predict_data[predictors]

        # Fit Gradient Boosting Regressor
        model = GradientBoostingRegressor(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=3)
        model.fit(X_train, y_train)

        # Predict missing values
        predictions = model.predict(X_predict)

        # Fill missing values in the target column
        final_merged_data.loc[(final_merged_data['DATE'] >= start_date) & 
                              (final_merged_data['DATE'] <= end_date), target_column] = predictions

        print(f"Training Gradient Boosting Regressor for {target_column}.............................")
        return final_merged_data

    except Exception as e:
        print(f"Error in regression model for {target_column}: {e}")
        return final_merged_data


# Making a final merged dataset
def merge_n_clean_transformed_data(macroeconomic_transformed_data, housing_transformed_data):
    final_merged_data = {}

    try:
        # Initialize the merged data with None
        macroeconomic_merged_data = None

        # Implementing merge logic which  can still merge the available datasets even some are missing
        if "CPI" in macroeconomic_transformed_data:
            macroeconomic_merged_data = macroeconomic_transformed_data["CPI"]

        if "Interest Rates (30-Year Fixed Mortgage)" in macroeconomic_transformed_data:
            if macroeconomic_merged_data is not None:
                macroeconomic_merged_data = macroeconomic_merged_data.merge(
                    macroeconomic_transformed_data["Interest Rates (30-Year Fixed Mortgage)"][["DATE", "MORTGAGE30US_MonthlyAvg"]],
                    on="DATE",
                    how="inner"
                )
            else:
                macroeconomic_merged_data = macroeconomic_transformed_data["Interest Rates (30-Year Fixed Mortgage)"][["DATE", "MORTGAGE30US_MonthlyAvg"]]

        if "Interest Rates (15-Year Fixed Mortgage)" in macroeconomic_transformed_data:
            if macroeconomic_merged_data is not None:
                macroeconomic_merged_data = macroeconomic_merged_data.merge(
                    macroeconomic_transformed_data["Interest Rates (15-Year Fixed Mortgage)"][["DATE", "MORTGAGE15US_MonthlyAvg"]],
                    on="DATE",
                    how="inner"
                )
            else:
                macroeconomic_merged_data = macroeconomic_transformed_data["Interest Rates (15-Year Fixed Mortgage)"][["DATE", "MORTGAGE15US_MonthlyAvg"]]

        if "Real Disposable Personal Income" in macroeconomic_transformed_data:
            if macroeconomic_merged_data is not None:
                macroeconomic_merged_data = macroeconomic_merged_data.merge(
                    macroeconomic_transformed_data["Real Disposable Personal Income"],
                    on="DATE",
                    how="inner"
                )
            else:
                macroeconomic_merged_data = macroeconomic_transformed_data["Real Disposable Personal Income"]

        # Merge available housing datasets
        housing_merged_data = None

        if "Single Family Home Prices" in housing_transformed_data:
            housing_merged_data = housing_transformed_data["Single Family Home Prices"]

        for dataset_key, column_subset in [
            ("All Hometypes Combined Prices", ["DATE", "Region", "All_HomePrice"]),
            ("Single Family Rental Prices", ["DATE", "Region", "SF_RentalPrice"]),
            ("All Hometypes Rental Prices", ["DATE", "Region", "All_RentalPrice"]),
            ("Single Family Rental Home Demand", ["DATE", "Region", "SF_RentalDemand"]),
            ("All Hometypes Rental Home Demand", ["DATE", "Region", "All_RentalDemand"]),
            ("All Hometypes Housing Market Heat Index", ["DATE", "Region", "H_Mkt_HeatIndex"]),
        ]:
            if dataset_key in housing_transformed_data:
                if housing_merged_data is not None:
                    housing_merged_data = housing_merged_data.merge(
                        housing_transformed_data[dataset_key][column_subset],
                        on=["DATE", "Region"],
                        how="left"
                    )
                else:
                    housing_merged_data = housing_transformed_data[dataset_key][column_subset]    
        
        # Save the merged datasets to separate files
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Save the merged macroeconomic data to a separate file
        if macroeconomic_merged_data is not None:
            macroeconomic_merged_data.to_csv(macroeconomic_output_file, index=False)
            print(f"1st Merge operation Successfull! Merged Macroeconomic dataset saved to {macroeconomic_output_file}\n{'-'*150}")
        else:    
            print("Error in merging individual macroeconomic datasets! No Macroeconomic Data found. \n{'-'*100}")
            return None 
    
        # Save the merged housing data to a separate file
        if housing_merged_data is not None:
            housing_merged_data.to_csv(housing_output_file, index=False)
            print(f"2nd Merge Operation Successfull! Merged Housing dataset saved to {housing_output_file}\n{'-'*150}")
        else:
            print("Error in merging individual housing datasets! No Housing Data found.\n{'-'*100}")
            return None

        
        # Convert both DATE columns to the first day of the month (because housing final_merged_data DATE column had last day of the month)
        housing_merged_data["DATE"] = pd.to_datetime(housing_merged_data["DATE"]).dt.to_period("M").dt.start_time
        macroeconomic_merged_data["DATE"] = pd.to_datetime(macroeconomic_merged_data["DATE"]).dt.to_period("M").dt.start_time

        # Merge the macroeconomic and housing datasets 
        final_merged_data = housing_merged_data.merge(macroeconomic_merged_data, on="DATE", how="left")

        # Fill missing values in the final_merged_data using Linear Regression
        final_merged_data['Year'] = final_merged_data['DATE'].dt.year
        final_merged_data['Month'] = final_merged_data['DATE'].dt.month
        le = LabelEncoder()
        final_merged_data['Region_Encoded'] = le.fit_transform(final_merged_data['Region'])
        final_merged_data['Region'] = le.inverse_transform(final_merged_data['Region_Encoded'])

        # implementing prework for to use regression model
        predictors = ['Year', 'Month', 'SF_HomePrice', 'All_HomePrice', 'Region_Encoded']
        target_columns = ['SF_RentalDemand', 'All_RentalDemand', 'H_Mkt_HeatIndex']
        available_target_columns = [col for col in target_columns if col in final_merged_data.columns]

        # Define date ranges for each target column
        target_date_ranges = {
            #'SF_RentalPrice': ('2014-01-01', '2014-12-01'),
            #'All_RentalPrice': ('2014-01-01', '2014-12-01'),
            'SF_RentalDemand': ('2015-01-01', '2020-05-01'),
            'All_RentalDemand': ('2015-01-01', '2020-05-01'),
            'H_Mkt_HeatIndex': ('2015-01-01', '2019-12-01')
        }
        # implementing a loop to counter if any of the target column or predictor is missing
        for target_column in target_columns:
            # Dynamically adjust target columns
            if target_column not in final_merged_data.columns:
                print(f"Skipping {target_column}: Column missing in the dataset.")
                continue

            # Dynamically adjust predictors
            available_predictors = [col for col in predictors if col in final_merged_data.columns]
            if len(available_predictors) == 0:
                print(f"Skipping {target_column}: No predictors available.")
                continue

            # Get the date range for the target column
            time_range_to_predict = target_date_ranges.get(target_column, None)
            if not time_range_to_predict:
                print(f"Skipping {target_column}: Date range not defined.")
                continue

            try:
                # Regressing Missing Values
                final_merged_data = Regress_Missing_Values(
                final_merged_data, 
                target_column=target_column, 
                time_range_to_predict=time_range_to_predict, 
                predictors=available_predictors
                )
                print(f"Regressing {target_column} using predictors: {available_predictors} "
                f"for date range {time_range_to_predict}.")
            except Exception as e:
                print(f"Failed to regress {target_column}: {e}")
        print(f"Yayy! Regression Model Ready\n{'-'*150}")

        # Drop temporary columns
        final_merged_data.drop(columns=['Year', 'Month', 'Region_Encoded'], inplace=True)

        # For the sake of completeness we will put a ffil and bffil operation to handle future cases e.g., There were 14 values missing in only All_RentalPrice Column.
        final_merged_data.ffill(inplace=True)
        final_merged_data.bfill(inplace=True)
        # Save the final merged dataset with predictions
        if final_merged_data is not None:
            final_merged_data.to_csv(final_output_file, index=False)
            print(f"Final Merge Operation (Housing + Macroeconomic Data) Successful! Final merged dataset saved to {final_output_file}")

            if any(final_merged_data[col].isna().any() for col in available_target_columns):
                print(f"Failed to fill missing values in columns {available_target_columns}.")
            else:
                print(f"All the Missing values in columns {available_target_columns} filled using Gradient Boosting Regression Model.")
                print(f"All other missing values filled using ffill and bfill methods.\n{'-'*150}")
            return final_merged_data
        else:
            print("Cannot make a final dataset! Error in merging both transformed datasets.")

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
            if data.empty:
                print(f"Skipping {name} ({data_type} dataset {idx}) due to extraction failure.\n{'-'*150}")
                continue  # Skip to the next dataset if extraction fails

            # Transformation
            try:
                transformed_data = transform_data(data, name)
            except Exception as e:
                print(f"Error transforming {name}: {e}\n{'-'*150}")
                continue  # Skip to the next dataset if transformation fails

            # Store transformed data
            if data_type == "macroeconomic":
                macroeconomic_transformed_data[name] = transformed_data
            elif data_type == "housing":
                housing_transformed_data[name] = transformed_data

            # Load transformed data
            try:
                load_data(transformed_data, data_type, idx, name)
            except Exception as e:
                print(f"Error loading {name}: {e}\n{'-'*150}")

    # Merging transformed data tables into a single dataset
    try:
        final_merged_data = merge_n_clean_transformed_data(macroeconomic_transformed_data, housing_transformed_data)
        if final_merged_data is not None:
            print(f"ETL pipeline completed successfully.")
            print(f"Data is ready for anlaysis.\n{'-'*150}")
        else:
            print(f"ETL pipeline failed during merging.\n{'-'*150}")
    except Exception as e:
        print(f"Error during merging: {e}\n{'-'*150}")


# Entry point check
if __name__ == "__main__":
    main()
