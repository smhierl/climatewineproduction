# Impact of Climate Change on Wine Production in California: A County Level Analysis

[The Wine Institute](https://wineinstitute.org/about-us/) - which represents about 1000 wineries and affiliated businesses and is the 'only US organization advocating for wine at the state, federal and international levels' - wanted to have a better picture of the potential impact of climate natural variability and long term trends on wine production in the State of California. More specifically, they wanted to understand whether an increase in average temperature and a reduction in precipitation would correspond with a substantial drop in wine production. Because of their economical value, the Institute was particularly interested in the following counties: Napa, Sonoma and San Joaquin. Serving as members of the Wine Institute Data Science Team, we embraced the challenge by developing a multiple linear regression model that mapped the relationship between weather metrics and wine production, with the intent of applying this model to future weather predicted by two different forward-looking climate scenarios: RCP 4.5, an intermediate scenario, and RCP 8.5, a worst case scenario. We anticipate that both scenarios will be associated with a decline in wine yield due to higher temperatures and lower precipitation, with a higher level of decline for 8.5 as compared to 4.5.


## Introduction
The major components of the Earth’s atmosphere are Nitrogen at 78.08%, Oxygen at 20.95% and several other gasses at lower concentrations, including greenhouse gasses (GHG) such as water vapor, carbon dioxide, methane and nitrous oxide. Despite this lower concentration, greenhouse gasses and changes in levels present have significant implications for Earth’s temperature. Some of the incoming sunlight is reflected back to space without being absorbed by surfaces with a high reflectance or albedo, such as our Arctic regions. However, some of the electromagnetic solar radiations hit the surface and is reflected back as infrared radiation (IR) - and part of this IR is captured by greenhouse gases.Without the presence in the atmosphere of greenhouse gasses, without their interaction with the infrared radiation, the average temperature on Earth would be about 0°F (-18°C). Indeed, all IR would be completely reflected back into space. However, to an increase of the greenhouse atmospheric concentration, corresponds an enhancement of the greenhouse effect leading to a modification of the Earth energy balance and thus, an increase of its temperature.

Due to a variety of geographic and meteorological factors, California has significant exposure to climate change. In particular, California’s agricultural industry has substantial amounts of risk associated with drought and extreme weather events. This should be of particular concern for policy makers as California is the largest agricultural producer and exporter in the United States, with roughly $49.1 billion of revenue being generated in [2020](https://www.cdfa.ca.gov/Statistics/).

At the economic and cultural heart of California’s agricultural industry is the production and sale of wine. According to the [Wine Institute](https://wineinstitute.org/press-releases/california-wine-sales-hit-40-billion-in-2020-despite-pandemic/), despite the pandemic California wineries “experienced an increase in sales by volume in 2020, totaling an estimated retail value of $40 billion.” However, there are multiple studies that show how climate natural variability and change threaten both wine quality and production. For instance, a study released in 2011 from Cornell University('Climate change associated effects on grape and wine quality and production') shows how under extreme hot temperatures, already observed in other regions as a result of climate change, “vine metabolism may be inhibited, leading to reduced metabolite accumulations which may affect wine aroma and color.” Other potential impacts include: an increased number of grape diseases and other pests due to temperature rise (Nemail et al. 2001), a decline in quality due to fewer cool nights (Elkjer 2008), and a decline in productivity or yield as a result of extreme hot days, with temperatures above 35C (Diffenbaugh 2006)

According to a Stanford University study (Jonathan Gatto et al. March 2009), the elements of climate that are most important for wine growth are “temperature, variability of day and night temperatures, the difference between winter and summer temperatures, sunlight, rainfall, humidity and wind.” In fact, specific grape varieties can only grow within limited temperature ranges. For instance, Chardonnay and Pinot Noir both need an average growing season temperature of 15-17C, while Cabernet Sauvignon and Sangiovese need a slightly higher temperature range of 17-19C. In fact grapes can be labeled as either cool and warm varieties and are constrained by the climate of their growing regions.

The same study reports that temperature and precipitation variability are the most important climate factors that will affect grapes and wine price, quality, and quantity in the future. In the same study, the authors have interviewed representatives of the wine industry indicating that i) in the short term water supply will be the most pressing challenge. It is important to notice that the water footprint of wine is  substantial. According to a study by the Global Footprint Network, the average water footprint of a bottle of wine is 632 liters. ii) In the long term, temperature variability will be the greatest concern. The interviewed farmers reported that in recent years, higher temperatures have forced growers to pick grapes sooner when sugar levels are too high, which in turn causes a higher alcohol content in wines, negatively affecting taste and quality.

Another component is that roughly 70% of laborers in California are immigrants who are in the United States illegally from countries south of the border who are also greatly threatened by climate change. This could indirectly threaten the quality, price, and yield of wine in California, highlighting the importance of a comprehensive adaptation strategy for the industry.


## Executive Summary

The following notebooks require the use of Pandas, Matplotlib, and Sklearn.

#### 01_Data Import and Cleaning 
__Data Sources__
There were many potential options for sourcing data, the best choice being the National Oceanic and Atmospheric Administration [(NOAA)](https://www.ncdc.noaa.gov/cag/). The thoroughness, transparency, and accessibility made it a clear choice. It provided both the climate features that seemed relevant to the problem being addressed, and the twenty year’s worth of data with minimal missing values. The maximum, minimum, and average temperature, as well as precipitation records were collected for each county, and then combined in [this notebook](https://git.generalassemb.ly/shierl/project-group-project/blob/1eb3dbaa93b882217ca321f7e43b7418d723d093/01_data_import_and_cleaning/Imports_And_Cleaning.ipynb.) The wine production was similarly sourced from a government database, as the [USDA National Agricultural Statistics Service](https://www.nass.usda.gov/Statistics_by_State/California/Publications/AgComm/) contained information on most major crops grown in California by county, dating back to 1980. The crop data from 2000-2020 was collected and combined into a data frame of all years, which was then narrowed down to exclusively wine grapes.   
__Cleaning and Merging__
As previously noted, the data was obtained from government agencies and collected by experts in the field. While NOAA data contained the information necessary for the model, it also contained extraneous columns in need of removal prior to the modeling process. Additionally, it was determined that only top-producing wine counties would be included in the dataset. Further cleaning included renaming columns, correcting data types, filtering the timespan to 2000-2020, and merging into one dataframe. 

Comparatively, the production data from the USDA National Agricultural Statistics Service was more relevant, and therefore did not require as much cleaning. The unit column was dropped, as all wine grape production was measured in tons. Furthermore, the crop name and commodity code columns were dropped due to redundancies. Because of this, the amount of missing data was relatively low, and the bulk of the cleaning was relegated to formatting. These minor formatting issues–such as mismatched spaces in the column names and string columns–were corrected before a merge could be performed. There were four rows missing data, so it was determined that they could be dropped without impact to the model. 


#### 02_EDA and Feature Engineering
We sourced data on wine-producing counties in California from 2000-2020 in two primary categories: wine grape production and climate data. Within production, observations included harvested acres, production, yield (production/acre), price per unit (ton), and total economic valuation per county. The climate data included monthly precipitation and minimum, maximum, and average temperature observations by county. 

Because the wine production data was collected yearly, we transformed the monthly climate observations into annual measures - averaged across the entirety of the year or solely the growing season, from April to September.

We selected yield to be the target variable for our model, defined as tons produced per acre harvested, due to its isolation from unobserved market factors that may impact value or affect total quantity of production.

In the process of EDA, we specifically looked at correlation to determine which variables we wanted to include in our model. The final variables were selected for their correlation with yield and lack of collinearity with each other - for this reason either the season or annual variable was picked for each metric, but not both (with the exception of annual and seasonal temperature averages, which showed a lower level of collinearity).

We narrowed down the counties included in our data frame to the top ten counties as measured by total economic value per year, as trends in these counties would have the greatest impact on the state economy. Dummy variables for each county were added to account for unobserved county specific factors.

#### 03_Production Model
Having selected the features to be included in our analysis, we then considered a variety of models to determine through comparison which one would be the best fit for the data and problem statement at hand. These models include the following:

> Linear Regression
> K-Nearest Neighbors
> Random Forest
> AdaBoost
> GradientBoost

We ran each model on the same selection of train and test data, and on both scaled and unscaled data where appropriate. In evaluating model performance, as indicated by Cross Val Score, Train Score, and Test Score, we determined Linear Regression to be the best model with which to proceed. As the Linear Regression model had comparable scores on both the scaled and unscaled data, we chose to proceed with unscaled data for ease of understanding later results.

We further refined the linear regression model through the use of a GridSearchCV, and found the best parameters were those assigned as default to the model.

#### 04_Future Projections
In this notebook, we load in the model developed in the previous stage and run predictions on two sets of future predicted climate data, formatted identically as our existing climate data for the three counties with the highest economic value of wine production: Napa, Sonoma, and San Joaquin. This data comes from the Representative concentration pathways or RCP, and simulates two possible future greenhouse gas emissions scenarios:  i) RCP 4.5 which is described by the Intergovernmental Panel on Climate Change or [IPCC] (https://www.ipcc-data.org/) as a moderate scenario, in which emissions peaks around 2040 and then decline; ii) the RCP 8.5 which is the highest baseline emissions scenario in which emissions will keep rising till the end of this century. 

## Data Dictionary

|Feature|Type|Dataset|Description|
|---|---|---|---|
|**year**|*int*|final_merged_data.csv|Year|
|**county code**|*int*|final_merged_data.csv|Code of county|
|**county**|*str*|final_merged_data.csv|Name of county|
|**harvested Acres**|*int*|final_merged_data.csv|Number of acres of wine grapes harvested|
|**yield**|*float*|final_merged_data.csv|Total production/Acres Harvested, measured in tons per Acre|
|**production**|*int*|final_merged_data.csv| Total tons produced by county |
|**Price P/U**|*float*|final_merged_data.csv|Cost per ton of wine grapes|
|**Value**|*int*|final_merged_data.csv|Dollar amount of wine grapes produced|
|**annual_precip**|*float*|final_merged_data.csv|Inches of precipitation in given year|
|**annual_tavg**|*float*|final_merged_data.csv|Average temperature in given year|
|**annual_tmin**|*float*|final_merged_data.csv|Minimum temperature in given year|
|**annual_tmax**|*float*|final_merged_data.csv|Maximum temperature in given year|
|**annual_var**|*float*|final_merged_data.csv| Difference in temperature day to day for year |
|**season_precip**|*float*|final_merged_data.csv|Temperature max for season|
|**season_tavg**|*float*|final_merged_data.csv|Temperature average for season|
|**season_tmin**|*float*|final_merged_data.csv|Temperature min for season|
|**season_tmax**|*float*|final_merged_data.csv|Temperature max for season|
|**season_var**|*float*|final_merged_data.csv|Difference in temperature day to day for season|
|**time Prec**|*Pandas Datetime*|historical_sonoma.csv|Date precipitation data was sourced|
|**mean Prec**|*float*|historical_sonoma.csv| Mean Precipitation |
|**mean Tmax**|*float*|historical_sonoma.csv|Average maximum temperature in a given year|
|**mean Tmin**|*float*|historical_sonoma.csv|Average minimum temperature in a given year|
|**time Prec**|*Pandas Datetime*|historical_sj.csv|Date precipitation data was sourced|
|**mean Prec**|*float*|historical_sj.csv|Mean Precipitation|
|**mean Tmax**|*float*|historical_sj.csv|Average maximum temperature in a given year|
|**mean Tmin**|*float*|historical_sj.csv|Average minimum temperature in a given year|
|**time Prec**|*Pandas Datetime*|historical_napa.csv|Date precipitation data was sourced|
|**mean Prec**|*float*|historical_napa.csv|Mean Precipitation|
|**mean Tmax**|*float*|historical_napa.csv|Average maximum temperature in a given year|
|**mean Tmin**|*float*|historical_napa.csv|Average minimum temperature in a given year|

As previously noted, the data was obtained from government agencies and collected by experts in the field. While NOAA data contained the information necessary for the model, it also contained extraneous columns in need of removal prior to the modeling process. Additionally, it was determined that only top-producing wine counties would be included in the dataset. Further cleaning included renaming columns, correcting data types, filtering the timespan to 2000-2020, and merging into one dataframe. 

Comparatively, the production data from the USDA National Agricultural Statistics Service was more relevant, and therefore did not require as much cleaning. The unit column was dropped, as all wine grape production was measured in tons. Furthermore, the crop name and commodity code columns were dropped due to redundancies. Because of this, the amount of missing data was relatively low, and the bulk of the cleaning was relegated to formatting. These minor formatting issues–such as mismatched spaces in the column names and string columns–were corrected before a merge could be performed. There were four rows missing data, so it was determined that they could be dropped without impact to the model.


## Conclusion  
We identified some level of relationship between observed weather and wine yield per acre. From our model performance, it seems very possible to predict yield per acre based on these weather observations. Further improvements to our model could be done through a more extensive data set - incorportating either a more detailed or larger volume of data, either on a geographic or temoporal scale. We could also incorporate additional information into our model to account for other potential factors, such as adaptive behavior, crop pests or disease, or soil composition. 
The predictive power of our model, in considering the next 100 years of climate conditions, is very limited. This is largely due to the random nature of weather and the very gradual nature of long term climate trends. The model still holds potential on this front, potentially with more detailed future projects or on a more limited time scale.
