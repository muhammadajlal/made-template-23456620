# Case Study of Top 10 Metropolitans of USA: Relation of Macroeconomic Indicators (Inflation, Interest Rates, and Income) & Housing Affordability
![image alt](https://github.com/muhammadajlal/made-template-23456620/blob/main/project/picture.png?raw=true)

## Main Question

<!-- Think about one main question you want to answer based on the data. -->
Analyzing the Relationship of Selected Macroeconomic Indicators & Housing Affordability in the USA.

## Description

<!-- Describe your data science project in max. 200 words. Consider writing about why and how you attempt it. -->
As a result of globalization and interconnected economies, we are witnessing an enormous outflow of skilled workers from developing countries towards developed economies like Germany and the USA. Usually, the ground realities of developed economies are not transparent from the home countries of expats, hence it becomes difficult for the skilled workers to make informed decisions while relocating to developed countries. The author particularly wants to analyze housing affordability over the last 10 years in the top 10 Metropolitans of the US. Housing affordability is the most important thing during relocation for expat skilled workers because rental prices constitute a disproportionate chunk of their disposable incomes as compared to other expenses like food, etc. Moreover, the condition of the housing market can be seen as a proxy measure to measure the health of an economy if analyzed side by side with Macroeconomic indicators. This study will provide an outlook on the US economy for people interested in moving there.

Analysis Steps:
1. Analyzing CPI/Inflation vis-a-vis Home Prices, Rental Prices, and Rental Demand.
2. Analyzing Income vis-a-vis Home Prices, Rental Prices, and Rental Demand.
3. Analyzing 30/15-Year Fixed Mortgage Rates vis-a-vis Home Prices, Rental Prices, and Rental Demand.
4. Analyzing Disposable Income growth trends with CPI, Home Prices, and Rental Prices.

## Datasources

<!-- Describe each datasources you plan to use in a section. Use the prefic "DatasourceX" where X is the id of the datasource. -->

### Datasource 1: Macroeconomic Data
* Metadata URL: https://fred.stlouisfed.org/
* Data URL: https://tinyurl.com/CPIInflationcsv (CPI)
* Data URL: https://tinyurl.com/RDPIcsv (Real Disposable Personal Income)
* Data URL: https://tinyurl.com/mortgagecsv (30Y Mortgage Fixed Rates)
* Data URL: https://tinyurl.com/15mortgagecsv (15Y Mortgage Fixed Rates)
* Data Type: CSV
<!-- * Data URL: https://tinyurl.com/RMHIncomecsv (Median Household Income) -->
<!-- * Data URL: https://fred.stlouisfed.org/release?t=&et=&rid=113&ob=pv&od=&tg=&tt=&pageID=2 (Unemployment Rate Data) - Optional -->

#### Short description of the DataSource:
Federal Reserve Bank of St. Louis (FRED) USA, is a database of more than 800,000 economic data series from over 100 sources covering issues and information relating to banking, business, consumer and producer price indices, employment, population, exchange rates, gross domestic product, interest rates, trade and international transactions, and U.S. financial data. In general, the Federal Reserve Bank of St. Louis encourages using FRED data and associated materials to support policymakers, researchers, journalists, teachers, students, businesses, and the general public. FRED provides data and data services to the public for non-commercial, educational, and personal uses subject to a few prohibitions.

All the macroeconomic data is sourced from the official website of the Federal Reserve Bank of St. Louis (FRED), USA. FRED collects data from different government departments e.g., the U.S. Bureau of Labor Statistics (BLS), etc and then adds some value to these data sources to make them analysis-ready. For example,  BLS provides most of the CPI data in chunks and FRED compiles all the data into a time series data with some value additions like smoothing the data for seasonality, etc. Although we have downloaded everything from FRED the original departmental sources of different indicators are given below.  

- Consumer Price Index (Inflation).  
Source: U.S. Bureau of Labor Statistics.  
Units:  Index 1982-1984=100.   
Frequency:  Monthly

- Real Disposable Personal Income.  
Source: U.S. Bureau of Economic Analysis.  
Units:  Billions of Chained 2017 Dollars.  
Frequency:  Monthly.  

- Real Median Household Income.  
Source: U.S. Census Bureau.    
Units:  2023 C-CPI-U Dollars.  
Frequency:  Annual.  

- 30-Year Fixed Mortgage Rate Average.  
Source: Freddie Mac.  
Units:  Percent.  
Frequency:  Weekly

- 15-Year Fixed Mortgage Rate Average.  
Source: Freddie Mac.  
Units:  Percent.  
Frequency:  Weekly 

<!-- - Unemployment Rates.  
Source: U.S. Bureau of Labor Statistics.     
Units:  Percent.  
Frequency:  Monthly  -->

### Datasource 2: Housing Data
* Metadata URL: https://www.zillow.com/research/data/
* Data URL: https://files.zillowstatic.com/research/public_csvs/zori/Metro_zori_uc_sfr_sm_month.csv?t=1730644371 (Single Family Rentals)
* Data URL: https://files.zillowstatic.com/research/public_csvs/zori/Metro_zori_uc_sfrcondomfr_sm_sa_month.csv?t=1730644371 (All Homes Combined Rentals)
* Data URL: https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv?t=1730644370 (All Homes Combined Prices)
* Data URL: https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv?t=1730644370 (Single Family Home Prices)
* Data URL: https://files.zillowstatic.com/research/public_csvs/zordi/Metro_zordi_uc_sfr_month.csv?t=1730644371 (Single Family Rental Home Demand)
* Data URL: https://files.zillowstatic.com/research/public_csvs/zordi/Metro_zordi_uc_sfr_month.csv?t=1731509126 (Single Family Rental Demand)
* Data URL: https://files.zillowstatic.com/research/public_csvs/zordi/Metro_zordi_uc_sfrcondomfr_month.csv?t=1730644371 (All Hometypes Combined Rental Demand)
* Data URL: https://files.zillowstatic.com/research/public_csvs/market_temp_index/Metro_market_temp_index_uc_sfrcondo_month.csv?t=1730644371 (Housing Market Heat Index)
* Data Type: CSV

#### Short description of the DataSource:
The Zillow Economic Research team publishes a variety of real estate metrics including median home values and rents, inventory, sale prices and volumes, negative equity, home value forecasts and many more. Most datasets are available at the neighborhood, ZIP code, city, county, metro, state and national levels, and many include data as far back as the late 1990s. All data accessed and downloaded from this page is free for public use by consumers, media, analysts, academics and policymakers. Proper and clear attribution of all data to Zillow is required.

#### We will use three different datasets built using a different index, each index is for a different purpose as described below.
1. Home Prices: Zillow Home Value Index (ZHVI): A measure of the typical home value and market changes across a given region and housing type. It reflects the typical value for homes in the 35th to 65th percentile range.
2. Home Rental Prices: Zillow Observed Rent Index (ZORI): A smoothed measure of the typical observed market rate rent across a given region. ZORI is a repeat-rent index that is weighted to the rental housing stock to ensure representativeness across the entire market, not just those homes currently listed for-rent. The index is dollar-denominated by computing the mean of listed rents that fall into the 35th to 65th percentile range for all homes and apartments in a given region, which is weighted to reflect the rental housing stock.
3. Rental Home Demand: Zillow Observed Renter Demand Index (ZORDI): A measure of the typical observed rental market engagement across a region. ZORDI tracks engagement on Zillow’s rental listings to proxy changes in rental demand. The metric is smoothed to remove volatility. 
4. Housing Markey heat Index: A measure that aims to capture the balance of for-sale supply and demand in a given market for all types of homes e.g., a higher number means the market is more tilted in favor of sellers.

___
___
___

# Methods of Advanced Data Engineering Template Project

This template project provides some structure for your open data project in the MADE module at FAU.
This repository contains (a) a data science project that is developed by the student over the course of the semester, and (b) the exercises that are submitted over the course of the semester.

To get started, please follow these steps:
1. Create your own fork of this repository. Feel free to rename the repository right after creation, before you let the teaching instructors know your repository URL. **Do not rename the repository during the semester**.

## Project Work
Your data engineering project will run alongside lectures during the semester. We will ask you to regularly submit project work as milestones, so you can reasonably pace your work. All project work submissions **must** be placed in the `project` folder.

### Exporting a Jupyter Notebook
Jupyter Notebooks can be exported using `nbconvert` (`pip install nbconvert`). For example, to export the example notebook to HTML: `jupyter nbconvert --to html examples/final-report-example.ipynb --embed-images --output final-report.html`


## Exercises
During the semester you will need to complete exercises using [Jayvee](https://github.com/jvalue/jayvee). You **must** place your submission in the `exercises` folder in your repository and name them according to their number from one to five: `exercise<number from 1-5>.jv`.

In regular intervals, exercises will be given as homework to complete during the semester. Details and deadlines will be discussed in the lecture, also see the [course schedule](https://made.uni1.de/).

### Exercise Feedback
We provide automated exercise feedback using a GitHub action (that is defined in `.github/workflows/exercise-feedback.yml`). 

To view your exercise feedback, navigate to Actions → Exercise Feedback in your repository.

The exercise feedback is executed whenever you make a change in files in the `exercise` folder and push your local changes to the repository on GitHub. To see the feedback, open the latest GitHub Action run, open the `exercise-feedback` job and `Exercise Feedback` step. You should see command line output that contains output like this:

```sh
Found exercises/exercise1.jv, executing model...
Found output file airports.sqlite, grading...
Grading Exercise 1
	Overall points 17 of 17
	---
	By category:
		Shape: 4 of 4
		Types: 13 of 13
```
