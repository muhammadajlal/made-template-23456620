# Project Plan

## Title
<!-- Give your project a short title. -->
Analyzing Effect of Selected Macroeconomic Indicators on Housing Affordability.

## Main Question

<!-- Think about one main question you want to answer based on the data. -->
A case study of top 10 Metroplitans of USA: How macro-economic indicators (inflation, interest rates, unemployment, and income) has effected the housing affordability over last 10 years.

## Description

<!-- Describe your data science project in max. 200 words. Consider writing about why and how you attempt it. -->
As a student living in Germany, and having experienced overal invrease in inflation and the housing crisis in metropolitan areas shows how much importance the Housing Sector have in an Economy because house rentals take a large chunk out of our disposable incomes so if rents are high it means one would be left with less money to spend, and this effect gets amplified with with rising inflation as the money loses its value over time. This project would be very insightful for the people who want to leave Germany due to language barriers or other preferences and move to USA (a dream destination for many) as this study attempts to help these people make informed choices based on statistcs. This study would use some macro-economic inicators as a rough measure of economic stability in USA and would also show how bad/good the Housing market is performing in recent years because it also entails how hard/easy is it to start a new life in USA.

Analysis Steps:
1. Analyzing CPI with single home prices and all homes prices.
2. Analyzing CPI with single home rent prices and all home rent prices.
3. Analyzing 30-year-fixed mortgage rates with the demand of single family and all homes combined.
4. Analyzing real-median-household growth income with CPI, home prices, and home rental prices.
5. Analyzing real-disposable-personal-income growth income with CPI, home prices, and home rental prices.
6. (Tentative) Analyzing the effect of unemployment rate on house affordability. 

## Datasources

<!-- Describe each datasources you plan to use in a section. Use the prefic "DatasourceX" where X is the id of the datasource. -->

### Datasource1: ExampleSource
* Metadata URL: https://fred.stlouisfed.org/
* Data URL: https://tinyurl.com/CPIInflationcsv (CPI)
* Data URL: https://tinyurl.com/RMHIncomecsv (Median Household Income)
* Data URL: https://tinyurl.com/RDPIcsv (Real Disposable Personal Income)
* Data URL: https://tinyurl.com/mortgagecsv (Mortgage Fixed Rates)
* Data URL: https://fred.stlouisfed.org/release?t=&et=&rid=113&ob=pv&od=&tg=&tt=&pageID=2 (Unemployment Rate Data) - Optional
* Data Type: CSV

Release: Consumer Price Index (Inflation)
Source: U.S. Bureau of Labor Statistics   
Units:  Index 1982-1984=100
Frequency:  Monthly

Release: Real Disposable Personal Income
Source: U.S. Bureau of Economic Analysis
Units:  Billions of Chained 2017 Dollars
Frequency:  Annual


Release: Real Median Household Income
Source: U.S. Census Bureau  
Units:  2023 C-CPI-U Dollars
Frequency:  Annual

Release: 30-Year Fixed Rate Mortgage Average
Source: Freddie Mac   
Units:  Percent
Frequency:  Weekly

Release: Unemployment Rates
Source: U.S. Bureau of Labor Statistics   
Units:  Percent
Frequency:  Monthly

### Datasource2: ExampleSource
* Metadata URL: https://www.zillow.com/research/data/
* Data URL: https://files.zillowstatic.com/research/public_csvs/zori/Metro_zori_uc_sfr_sm_month.csv?t=1730644371 (Single Family Rentals)
* Data URL: https://files.zillowstatic.com/research/public_csvs/zori/Metro_zori_uc_sfrcondomfr_sm_sa_month.csv?t=1730644371 (All Homes Combined Rentals)
* Data URL: https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv?t=1730644370 (All Homes Combined Prices)
* Data URL: https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv?t=1730644370 (Single Family Home Prices)
* Data URL: https://files.zillowstatic.com/research/public_csvs/zordi/Metro_zordi_uc_sfr_month.csv?t=1730644371 (Single Family Rental Home Demand)
* Data URL: https://files.zillowstatic.com/research/public_csvs/zordi/Metro_zordi_uc_sfrcondomfr_month.csv?t=1730644371 (All Hometypes Combined Rental Demand)
* Data Type: CSV

#### Short description of the DataSource:
The Zillow Economic Research team publishes a variety of real estate metrics including median home values and rents, inventory, sale prices and volumes, negative equity, home value forecasts and many more. Most datasets are available at the neighborhood, ZIP code, city, county, metro, state and national levels, and many include data as far back as the late 1990s. All data accessed and downloaded from this page is free for public use by consumers, media, analysts, academics and policymakers, consistent with our published Terms of Use. Proper and clear attribution of all data to Zillow is required.

#### We will use three different datasets built using a different index, each index is for a different purpose as described below.
1. Home Prices: Zillow Home Value Index (ZHVI): A measure of the typical home value and market changes across a given region and housing type. It reflects the typical value for homes in the 35th to 65th percentile range.
2. Home Rental Prices: Zillow Observed Rent Index (ZORI): A smoothed measure of the typical observed market rate rent across a given region. ZORI is a repeat-rent index that is weighted to the rental housing stock to ensure representativeness across the entire market, not just those homes currently listed for-rent. The index is dollar-denominated by computing the mean of listed rents that fall into the 35th to 65th percentile range for all homes and apartments in a given region, which is weighted to reflect the rental housing stock.
3. Rental Home Demand: Zillow Observed Renter Demand Index (ZORDI): A measure of the typical observed rental market engagement across a region. ZORDI tracks engagement on Zillow’s rental listings to proxy changes in rental demand. The metric is smoothed to remove volatility. 


## Work Packages

<!-- List of work packages ordered sequentially, each pointing to an issue with more details. -->

1. Select Datasets
2. Build an automated data pipeline
3. perform Exploratory Data Analysis and Feature Engineering
4. Make Analysis Report
5. Optional: Statistical Modelling for Prediction of Trends
6. Reporting on findings
