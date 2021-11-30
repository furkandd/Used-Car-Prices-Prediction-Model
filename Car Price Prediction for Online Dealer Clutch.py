#!/usr/bin/env python
# coding: utf-8

# # Building a User Car Prediction Model for Clutch
# <br>
# <br>
# In this project, I will do some analysis on a sample data of used vehicles and deploy a Machine Learning model to predict vehicle prices by following the steps below.
# <br>
# <br>
# 1. <b>Import Packages:</b> We'll be importing relevant libraries and packages.
# <br>
# 2. <b>Data Upload:</b> We are using a sample data scrape provided by Clutch.
# <br>
# 3. <b>Business Problem:</b> We'll frame the problem based on the dataset description.
# <br>
# 4. <b>Exploratory Data Analsis (EDA) I:</b> We'll carry out an exploratory analysis to figure out the important features and creating new combination of features.
# <br>
# 5. <b>Data Cleaning and EDA II:</b> We'll clean the raw data and address any data quality issues. 
# <br>
# 6. <b>Data Analysis:</b> At this step, we are going to be analyzing the cleaned data and providing insights. 
# <br>
# 7. <b>Preprocessing and ML Modelling:</b> We'll process the data for ML models and do some feature engineering before we choose our ML algorithm. 
# <br>

# ### Step 1: Import Packages

# In[124]:


import datetime

import numpy as np
import pandas as pd

from wordcloud import WordCloud, STOPWORDS 

import re

import warnings
warnings.filterwarnings('ignore')


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
import statsmodels.api as sm


# ### Step 2: Data Upload

# In[377]:


cars = pd.read_csv("vehicles.csv")

#Pandas normally shows a limited number of columns on Jupyter notebooks.
#The line below is enforcing it to display all the columns available. 
pd.set_option("display.max_columns", 15)
cars.head(5)


# ### Step 3: Business Problem
# 
# The dataset provided by Clutch online dealership contains market data obtained from listing websites. The dataset contains <b>price</b> variable which is continuous data and tells us about the asking price of each car in the dataset.
# 
# Our aim here is to analyze the data and <b>predict the price </b> of a car given that we have other attributes of that vehicle.

# ### Step 4: Exploratory Data Analsis (EDA)
# <br>
# In this part, we will follow the steps below and do our EDA to get more information about our data.
# <br>
# 
# * Check data size
# * Check data type of columns
# * Check for duplicate entries
# * Check for null values
# * Check for data entry error and outliers
# * Look for the frequency distribution in categorical columns
# * Initial insights

# #### Check data size

# In[126]:


print("No. of rows: "+ str(cars.shape[0]))
print("No. of columns: "+ str(len(cars.columns)))


# We have a total of <b> 52,236</b> rows. Let's see what types of data we have under columns and see if we need to any cleaning and transformation. 

# #### Check data types of columns

# In[379]:


pd.set_option("display.max_rows", 10)
cars.info()


# #### Checking for duplicates
# 

# In[5]:


# Checking for duplicate entries but turns out there is none!
cars.duplicated().sum()


# #### Checking for NULL values
# 
# As seen below, there are a lot null values that should be handled carefully. We'll address these issues going forward.

# In[382]:


pd.set_option("display.max_rows",None)
print("No. of rows:",cars.shape[0])
print("\n")
print("No. of nulls by columns")
cars.isnull().sum()


# #### Check for data entry error and outliers
# 
# Let's see the data types again.

# In[7]:


cars.select_dtypes(include=['object', 'category']).columns


# Maximum and minimum values for all the numeric values below will be investigated in the future steps with focus on outliers.

# In[8]:


cars_numeric_fields = ['year','mileage','price','passengers']
cars[cars_numeric_fields].describe().apply(lambda s: s.apply('{0:.0f}'.format))


# The data was collected from <b>April 2, 2021</b> to <b>May 3, 2021.</b>

# In[9]:


print(cars.first_date_seen.min())
print(cars.first_date_seen.max())
print(cars.last_date_seen.min())
print(cars.last_date_seen.max())


# #### Look for the category distribution in categorical columns

# In[385]:


pd.set_option("display.max_rows",20)

cars["make"].value_counts()


# In[386]:


cars["model"].value_counts()


# In[387]:


cars.trim.value_counts()


# In[388]:


cars.color.value_counts()


# In[389]:


cars.body_type.value_counts()


# In[390]:


cars.drivetrain.value_counts()


# In[391]:


cars.transmission.value_counts()


# In[392]:


cars.fuel_type.value_counts()


# In[393]:


cars.engine.value_counts()


# In[394]:


cars.city.value_counts()


# In[395]:


cars.seller_name.value_counts()


# #### Initial insights
# 
# Looking at the initial EDA above, we can say that dataset is not the cleanest in the world. But it is an interesting one to work on for sure!
# 
# We will clean it by fixing data entry errors, imputing NULL values, clearing outliers, handling categorical values by string manipulations and some feature engineering in further steps.

# ### Step 5: Data Cleaning and EDA II
# <br>
# In this part, we'll clean the raw data and address any data quality issues by following the steps below.
# <br>
# 
# * Correct data type issues
# * Handle categorical variables with string manipulations
# * Impute null values in categorical values, drop if needed
# * Fix issues in numerical columns
# * Look for new variables
# <br>

# ####  Correct data type issues
# 
# * Let's see the datatypes again and decide what to do!
# 

# In[396]:


cars.info()


#  * First off the bat, let's get the datefields correct. 

# In[22]:


cars.first_date_seen=pd.to_datetime(cars.first_date_seen)
cars.last_date_seen=pd.to_datetime(cars.last_date_seen)


# * <b>vin</b> and <b>carfax_url</b> fields have unique values for each and every data point so it's helpful to convert them to boolean. This info will also be relavant in our further analysis.

# In[397]:


cars['vin_provided'] = cars.vin.apply(lambda x: 0 if pd.isnull(x) else 1)
cars['carfax_provided'] = cars.carfax_url.apply(lambda x: 0 if pd.isnull(x) else 1)
cars['is_private'] = cars.is_private.apply(lambda x: 0 if x is False else 1)
cars= cars.drop(columns=['vin','carfax_url'])


# * <b>Trim</b> level has a lot of unique text info for each data point but it is not eligible to be converted into a categorical or a numeric column. So, our best bet is to merge it with the description column so that we can get some insights from the merged field.

# In[398]:


cars["description"] = cars["trim"].map(str)+ ' ' + cars["description"].map(str)
cars= cars.drop(columns=['trim'])


# #### Handle categorical variables with string manipulations

# * Color field is an important one but has a lot of unique values that needs a bit of string manipulation.

# In[399]:


cars.color.value_counts()


# In[400]:


cars.loc[cars['color'].str.contains('white',na=False,flags=re.IGNORECASE), 'color'] = 'White'
cars.loc[cars['color'].str.contains('black',na=False,flags=re.IGNORECASE), 'color'] = 'Black'
cars.loc[cars['color'].str.contains('red',na=False,flags=re.IGNORECASE), 'color'] = 'Red'
cars.loc[cars['color'].str.contains('silver',na=False,flags=re.IGNORECASE), 'color'] = 'Silver'
cars.loc[cars['color'].str.contains('gray',na=False,flags=re.IGNORECASE), 'color'] = 'Gray'
cars.loc[cars['color'].str.contains('grey',na=False,flags=re.IGNORECASE), 'color'] = 'Gray'
cars.loc[cars['color'].str.contains('charcoal',na=False,flags=re.IGNORECASE), 'color'] = 'Gray'
cars.loc[cars['color'].str.contains('granite',na=False,flags=re.IGNORECASE), 'color'] = 'Gray'
cars.loc[cars['color'].str.contains('blue',na=False,flags=re.IGNORECASE), 'color'] = 'Blue'
cars.loc[cars['color'].str.contains('metallic',na=False,flags=re.IGNORECASE), 'color'] = 'Metallic'
cars.loc[cars['color'].str.contains('steel',na=False,flags=re.IGNORECASE), 'color'] = 'Metallic'
cars.loc[cars['color'].str.contains('green',na=False,flags=re.IGNORECASE), 'color'] = 'Green'
cars.loc[cars['color'].str.contains('brown',na=False,flags=re.IGNORECASE), 'color'] = 'Brown'

cars["color"] =              np.where(cars.groupby(["color"])["color"].transform('count') >= 50, 
                      cars.color, 'Other')
cars.loc[cars.color=='Not Specified',"color"]="Other"
cars.color.value_counts()


# * There are a lot of car manufacturers under <b>make</b> field and the size of the unique list makes it harder to analyze. So, I am going to mask them by tagging smaller brands as 'Other'. This is totally a random threshold. I initially set it to <b>brands with # of records > 100</b> but that was filtering out many sport vehicles and I did not want to exclude them. Instead, I remove those under 15.

# In[401]:


cars["make"] =              np.where(cars.groupby(["make"])["make"].transform('count') >= 15, 
                      cars.make, 'Other')
cars.loc[cars['make'].str.contains('mercedes',na=False,flags=re.IGNORECASE), "make"] = 'Mercedes'
cars.make.value_counts()


# * Can't do anything about the <b>model</b> field. There are many unique fields for each car brand and it is not feasible to do any sory of transformation on it. I'll just display the unique values here!

# In[402]:


cars.model.str.title().value_counts()


# * <b>Body_type</b> field might help us drive insights from the dataset but it requires a bit of string manipulations. If I have time, I will write a separate function for the code below because this type of cleaning is required for other fields as well. 

# In[403]:


cars.loc[cars['body_type'].str.contains('truck',na=False,flags=re.IGNORECASE), 'body_type'] = 'Truck'
cars.loc[cars['body_type'].str.contains('cabriolet',na=False,flags=re.IGNORECASE), 'body_type'] = 'Convertible'
cars.loc[cars['body_type'].str.contains('cab',na=False,flags=re.IGNORECASE), 'body_type'] = 'Truck'
cars.loc[cars['body_type'].str.contains('wagon',na=False,flags=re.IGNORECASE), 'body_type'] = 'Station Wagon'
cars.loc[cars['body_type'].str.contains('super crew',na=False,flags=re.IGNORECASE), 'body_type'] = 'Truck'
cars.loc[cars['body_type'].str.contains('cutaway',na=False,flags=re.IGNORECASE), 'body_type'] = 'Truck'
cars.loc[cars['body_type'].str.contains('avant',na=False,flags=re.IGNORECASE), 'body_type'] = 'Station Wagon'

cars.body_type.value_counts()


# * <b>Drive_Train</b> columns is also useful one. It also needs a bit of string manipulations. A little domain knowledge might come in handy at this part of the analysis. You can also do online research about what the main drive_train types are and what keywords are used interchangibly for those categories.   

# In[30]:


cars.drivetrain.value_counts()


# In[404]:


cars.loc[cars.drivetrain == '2WD', 'drivetrain'] = "FWD" # this one is just an assumption. It could be RWD as well!
cars.loc[cars.drivetrain.isin(['4X4','4x4']), 'drivetrain'] = "4WD"
cars.drivetrain.value_counts()


# * <b> Transmission </b> field is also a valueable one. We are applying our good old method to do some transformation here as well. 

# In[405]:


cars.transmission=cars.transmission.astype('str')

cars.loc[cars['transmission'].str.contains('Automatic'), 'transmission'] = 'Automatic'
cars.loc[cars['transmission'].str.contains('Manual'), 'transmission'] = 'Manual'
cars.loc[cars['transmission'].str.contains('Sequential'), 'transmission'] = 'Manual'
cars.loc[cars['transmission'].str.contains('CVT'), 'transmission'] = 'CVT'

cars.loc[cars.transmission == 'nan', 'transmission'] = None


# * Let's also look at the <b>fuel_type</b> field. 

# In[406]:


cars['fuel_type'] = cars['fuel_type'].replace(['Premium Unleaded','Gasoline','Gasoline Fuel','Regular Unleaded'],'Gas')
cars['fuel_type'] = cars['fuel_type'].replace(['E85- Gasoline(Flex Fuel)'],'Flexible')
cars['fuel_type'] = cars['fuel_type'].replace(['Other/Don’t Know'],'Other')
cars['fuel_type'] = cars['fuel_type'].replace(['Gasoline Hybrid','Gasoline - Hybrid'],'Hybrid')


# * For the <b>engine </b> field, we are going to do our transformation. However, due to the way that data points were entered, it is not too feasible to work on every unique record. Instead, we will focus on transforming the field so in a way that we can extract <b># of cylinders</b> as a close proxy in a separate field in the further steps.   

# In[407]:


cars.loc[cars['engine'].str.contains("4 cyl",na=False,flags=re.IGNORECASE), "engine"] = '4 Cylinder'
cars.loc[cars['engine'].str.contains("4 cyclinder",na=False,flags=re.IGNORECASE), "engine"] = '4 Cylinder'
cars.loc[cars['engine'].str.contains("4-cyclinder",na=False,flags=re.IGNORECASE), "engine"] = '4 Cylinder'
cars.loc[cars['engine'].str.contains("4cyl",na=False,flags=re.IGNORECASE), "engine"] = '4 Cylinder'
cars.loc[cars['engine'].str.contains("6 cyl",na=False,flags=re.IGNORECASE), "engine"] = '6 Cylinder'
cars.loc[cars['engine'].str.contains("6cyl",na=False,flags=re.IGNORECASE), "engine"] = '6 Cylinder'
cars.loc[cars['engine'].str.contains("8cyl",na=False,flags=re.IGNORECASE), "engine"] = '8 Cylinder'
cars.loc[cars['engine'].str.contains("8 cylinder",na=False,flags=re.IGNORECASE), "engine"] = '8 Cylinder'
cars.loc[cars['engine'].str.contains("v6",na=False,flags=re.IGNORECASE), "engine"] = '6 Cylinder'
cars.loc[cars['engine'].str.contains("v-6",na=False,flags=re.IGNORECASE), "engine"] = '6 Cylinder'
cars.loc[cars['engine'].str.contains("3 cylinder",na=False,flags=re.IGNORECASE), "engine"] = '3 Cylinder'
cars.loc[cars['engine'].str.contains("5 cylinder",na=False,flags=re.IGNORECASE), "engine"] = '5 Cylinder'
cars.loc[cars['engine'].str.contains("5cyl",na=False,flags=re.IGNORECASE), "engine"] = '5 Cylinder'
cars.loc[cars['engine'].str.contains("3cyl",na=False,flags=re.IGNORECASE), "engine"] = '3 Cylinder'
cars.loc[cars['engine'].str.contains("v8",na=False,flags=re.IGNORECASE), "engine"] = '8 Cylinder'
cars.loc[cars['engine'].str.contains("v-8",na=False,flags=re.IGNORECASE), "engine"] = '8 Cylinder'

cars.loc[~cars.engine.isin(['4 Cylinder','6 Cylinder','8 Cylinder','3 Cylinder','5 Cylinder']),'engine'] = None


# * Nothing interesting in this <b>seller_name</b> field because there are all small dealers except for a few big chains. If we had more data points from CLUTCH, we might have been able to calculate a new field and see if it is significant at all. But unfortunately we have not! Also, the largest group in the output below is 'Private' sellers. We already have a separate column for this information. That's why we'll drop this column in the next steps.

# In[408]:


cars.loc[cars['seller_name'].str.contains("toyota",na=False,flags=re.IGNORECASE), "seller2"] = 'toyota'
cars.loc[cars['seller_name'].str.contains("honda",na=False,flags=re.IGNORECASE), "seller2"] = 'honda'
cars.loc[cars['seller_name'].str.contains("clutch",na=False,flags=re.IGNORECASE), "seller2"] = 'clutch'
cars.loc[cars['seller_name'].str.contains("dodge",na=False,flags=re.IGNORECASE), "seller2"] = 'dodge'
cars.loc[cars['seller_name'].str.contains("private",na=False,flags=re.IGNORECASE), "seller2"] = 'private'
cars.loc[cars['seller_name'].str.contains("ford",na=False,flags=re.IGNORECASE), "seller2"] = 'ford'
cars.loc[cars['seller_name'].str.contains("hyundai",na=False,flags=re.IGNORECASE), "seller2"] = 'hyundai'


cars.seller2.value_counts()


# ####  Impute null values, drop them if needed

# We have a lot of null values. We will use two techniques to get rid of them or impute them from other fields where applicable. Main NULL elimination techniques we use are: 
# * Backward, forward imputation
# * Taking group mean
# 
# First, let's see how many we got null values in each field.

# In[409]:


cars.isnull().sum()


# In[410]:


cars['body_type']=cars.groupby('model')['body_type'].ffill().bfill()
cars.body_type.value_counts()


# In[411]:


cars['drivetrain']=cars.groupby(['make','model'])['drivetrain'].ffill().bfill()

cars.drivetrain.value_counts()


# In[412]:


cars['transmission']=cars.groupby(['make','model'])['transmission'].ffill().bfill()
cars.transmission.value_counts()


# In[413]:


cars['fuel_type']=cars.groupby(['make','model'])['fuel_type'].ffill().bfill()
cars["fuel_type"].value_counts()


# In[414]:


cars['engine']=cars.groupby(['make','model','body_type'])['engine'].ffill().bfill()

cars["cylinders"]=cars.engine.str.replace('Cylinder','',regex=True).astype(int)
cars= cars.drop(columns=['engine'])
cars.cylinders.value_counts()


# In[416]:


cars["city"] = cars.city.str.title() #Fixing UPPER, LOWER case inconsistencies.

cars.loc[cars.city == 'Richmond', 'city'] = 'Richmond Hill' #City was mistakenly recorded as Richmond (BC) instead of 
## Richmond Hill (ON). I checked the longitude and latitude values to make sure and fix it manually.


toronto_gta=['Toronto','North York','Scarborough','Etobicoke','York'
            'Burlington','Oakville','Milton','Halton Hills',
            'Mississauga','Brampton','Caledon',
             'Vaughan','Concord','Unionville','Markham','Newmarket','Richmond Hil','King','Stouffville',
             'East Gwillimbury','Whitchurch-Stouffville','Aurora','Georgina',
             'Pickering','Ajax','Whitby','Uxbridge','Brock','Scugog','Oshawa','Clarington'
            ] 
## There a lot of unique values so instead we are creating a new, boolean column to check if the car is in Toronto GTA.

cars["toronto_gta"]= np.where(cars.city.str.contains('|'.join(toronto_gta)), 1, 0)

cars.toronto_gta.value_counts()


# In[417]:


cars=cars.drop(columns=['seller_name','seller2']) # Dropping seller name columns b/c we're not getting any info from them.  


# #### Fix issues in numerical columns

# In[418]:


cars.describe()


# * <b>Year</b>

# In[419]:


cars[cars.year== 1914] #Checking if 1914 is a data entry or an actual data point.


# These two cars look like they are duplicates and they are from Baltimore, a US city out of the scope of this analysis. Let's remove them.
# 

# In[420]:


cars=cars[cars.city.str.title()!='Baltimore']


# Now, it is better that we bring an age field created from the year column.

# In[421]:


cars["age"]=2021-cars["year"]


# In[422]:


fig,axs=plt.subplots(nrows=2)
fig.set_size_inches(20,10)
sns.barplot(x='age',y='price',data=cars,ax=axs[0])
sns.barplot(cars.groupby('age').count()['price'].index,cars.groupby('age').count()['price'].values,ax=axs[1])
axs[0].set_title('Figure 3')
axs[1].set_title('Figure 4')
axs[1].set_ylabel('Number of cars')
plt.tight_layout()
plt.show()


# Looking at the graphs above, it makes a lot of sense to me that at this point we should create a new field that would account old, classic cars or in broader terms vehicles that increase in value as they get old unlike the new cars?
# 
# In most provinces in Canada, the minimum age for a car to get vintage license plate is 30.

# In[423]:


cars.loc[cars.age > 30, "vintage"] = 1
cars.loc[cars.vintage != 1, "vintage"] = 0  
cars.vintage.value_counts()


# * <b>Mileage</b>

# In[424]:


pd.set_option('display.float_format', lambda x: '%.0f' % x) #Just a formatting script

cars.mileage.describe()


# In[425]:


cars.loc[(cars.mileage==0) & (cars.year < 2019),'mileage']=None # Assuming that old cars can't have 0 km.


# In[426]:


pd.set_option('display.float_format', lambda x: '%.0f' % x)
cars.mileage.describe()


# In[427]:


## Again we are subsetting our data. This is not by intuition. I checked the vehicle with the max mileage(3965000).
## Turns out it is a data error with null values in other fields too. That's why I removed it and limited our 
## analysis to only those below 3,000,000 altough this, too, smells fishy. We'll take care of it in the next steps!

cars= cars[cars['mileage'] < 3000000] 
cars.mileage.describe()


# In[182]:


cars[cars['mileage'] > 1000000]

#looks like people were putting an extra digit by mistake. We can divide those by 10 and fix this issue.


# In[428]:


cars.loc[(cars.mileage>1000000),'mileage']=cars.mileage/10


# In[429]:


cars.mileage.describe()


# In[430]:


cars.loc[(cars.mileage>500000) & (cars.year > 2015),:] #Let'see what cars we have next in terms of highest odomoter reading.


# In[431]:


cars.loc[(cars.mileage>500000) & (cars.year > 2015),'mileage']=cars.mileage/10 ## same issue with relatively new cars.


# In[432]:


cars.mileage.describe()


# In[433]:


cars[cars.mileage==999999] # Who is this Number-9-lover?


# In[434]:


cars.loc[(cars.mileage==999999),'mileage']=None ## A data error by the looks of it.


# In[435]:


cars.mileage.describe()
# Checked other cars with odomoter readings of upwards of 500,000. Althoug they look a little suspicious too,
# I won't remove them b/c they are all trucks. I have always used sedan. Not so much insight into larger vehicles.
# Someone with more domain knowledge might be able to handle this better. 


# In[436]:


cars.loc[(cars.mileage>0) & (cars.mileage < 500)& (cars.year < 2015),:] 
# Next, we are investigating olders vehicles with seemingly low KM on them. 


# In[437]:


cars.loc[(cars.mileage>0) & (cars.mileage < 500) & (cars.year < 2015),'mileage']=None

# We're rendering them all NULL. We'll later impute them with other NULL values by using back and forward filling
# based on other fields.


# In[438]:


cars['mileage'] = cars.groupby(['make','year'])['mileage'].apply(lambda x: x.fillna(x.mean()))
cars.mileage.isnull().sum()


# In[439]:


cars['mileage'] = cars.groupby(['year'])['mileage'].apply(lambda x: x.fillna(x.mean()))
## still some null values that's why we are now using avg odometer readings of the vehicles with the same year.
# Depreciation should be more or less the same for the cars with same age.


# * <b>Price</b> 

# In[440]:


cars.price.describe()


# In[441]:


cars[cars.price>500000] # Surprisingly,nothing looks fishy! All look sport cars and we'll keep them.


# In[442]:


cars[cars.price<500] #Checking to see if there is any data errors on cars cheaper than 500$.


# In[443]:


cars.loc[(cars.price<500) & (cars.description.str.contains('parts')),:]

# We're assuming that cheap cars might be up for sale for their parts. If this is the case, we are not interested in them.


# In[444]:


indexes = cars.loc[(cars.price<500) & (cars.description.str.contains('parts')),:].index

cars.drop(indexes,inplace=True) # Turns out some of them has 'parts' keyword in the ad description so we can remove them.


# In[445]:


cars[cars.price.isnull()]


# In[446]:


cars['price'] = cars.groupby(['make','model','year'])['price'].apply(lambda x: x.fillna(x.mean()))

# Remaining NULL values are filled by group average based on make, model and year.


# * <b> Passengers </b>

# In[447]:


cars.passengers.describe()


# In[448]:


## turns out this car (Subaru Crosstek) has actually a capacity of 5 passengers. Manually fixing it.

cars.loc[cars.passengers == 17, 'passengers'] = 5  


# In[449]:


## turns out 1744 vehicles has mistakenly 0 passenger capacity. This gotta be a mistake. We'll fix that by first
## converting them into Null then filling null values with most frequent value in respective groups.

cars.loc[cars.passengers == 0, 'passengers'] = None

cars['passengers']=cars.groupby(['make','model','body_type'])['passengers'].ffill().bfill()
cars.passengers.value_counts()


# In[450]:


cars=cars.drop(columns=['city','province']) # we drop the city column. 
## We also drop province because it is all Ontario except for around 100 data points from BC (Creston)
## and Newfoundland (Clarenville and Grandfalls-Windsor).


# ### Step 6: Data Analysis

# In[451]:


def classify_price(x):    
    if x<10000:
        return "<10000"
    
    if x >= 10000 and x <= 30000:
        return "10000-30000"
    
    if x >= 30000 and x <= 50000:
        return "30000-50000"    
    
    if x>50000:
        return ">50000"
    return "NA"


# In[452]:


cars['classifications'] = cars.price.apply(classify_price)


# In[453]:


# Effect of make to price
piv = pd.pivot_table(cars,index=['make'],columns='classifications',values=[],aggfunc=len)
piv.fillna(0, inplace=True)
piv = piv[['<10000','10000-30000','30000-50000','>50000']]
piv = piv.div(piv.sum(axis=1), axis = 0) * 100


# In[454]:


plt.subplots(figsize=(10,15))
sns.heatmap(piv,cmap="Blues",annot=True,fmt=".2f");
plt.xlabel("Price Bracket")
plt.ylabel("Make");
plt.title("Percent of Cars in Each Price Bracket Based on Make");
plt.tick_params(right=True, top=True, labeltop=True,rotation=0);


# <b>Insight 1:</b> Premium luxury vehicles such as Aston Martin, Bentley, Rolls-Royce, Ferrari, Tesla, Porsche, Maserati and Lamborghini have overwhelming majority of their cars listed upwards of 50,000\\$. Brands such as Toyota, Honda, Hyundai, Kia and Nissan have their cars mostly listed under the second tier, 10,000\\$ - 30,000\\$. The brands like Saturn, Saab, Suzuki, Smart and Fiat have almost all of their listings under 10,000\\$. This could be due to the lower perception of brand quality in the used market in Canada or perhaps in Ontario in particular.

# In[455]:


plt.subplots(figsize=(15,5))
g = sns.regplot(x="age",y="price",data=cars)
regline = g.get_lines()[0]
regline.set_color('red')
plt.ticklabel_format(style='plain', axis='y')
plt.title("Relationship Between Model Age and Price");


# <b>Insight 2:</b> As expected, price goes down as the vehicle ages, sort of insight that is also backed up by the downward sloping curve on the graph above.
# 
# <b>Insight 3:</b> Our dataset contains relatively older cars that skew the data a little on the right edge above the red trendline. This should be due to vintage cars for which we already created a new variable earlier. But despite this the Insight 2 still holds. 

# In[456]:


plt.subplots(figsize=(15,5))
g = sns.regplot(x="mileage",y=np.log(cars.price),data=cars) #taking log of the price
regline = g.get_lines()[0]
regline.set_color('red')
plt.title("Relationship Between Odometer Distance and Price");


# In[457]:


cars.head()


# <b>Insight 4:</b> The trend is obvious that prices go down as car accumulates more mileage on their odometers.

# In[458]:


plt.subplots(figsize=(15,5))
sns.boxplot(x=cars.is_private, y=np.log(cars.price));
plt.ylabel("Log-Price")
plt.show() 


# <b>Insight 5:</b>
#     
# * Private sellers tend to have lower prices.

# In[459]:


plt.subplots(figsize=(15,5))
sns.boxplot(x=cars.color, y=np.log(cars.price));
plt.ylabel("Log-Price")
plt.show() 


# <b>Insight 6:</b>
#     
# * Selling price does not vary a lot across different body colors. While Yellow, Black and White cars look slightly more expensive, gold-color vehicles sells cheaper than the rest. But again, it is not too much of a difference.

# In[460]:


plt.subplots(figsize=(15,5))
sns.boxplot(x=cars.body_type, y=np.log(cars.price));
plt.ylabel("Log-Price")
plt.show() 


# 
# <b>Insight 7:</b>
# * Body type is an important differentiator. Roadsters are by far the most expensive cars according to the data. This makes sense because sport cars mostly target more wealthy consumers. Trucks follows roadsters on a descending order. Hatchbacks are listed as most affordable cars while sedans follow hatchback in affordability. The graph also displays that two other sporty-looking body types, namely convertibles and coupes, have larger variance, meaning that their prices could go up and down by a lot.

# In[461]:


plt.subplots(figsize=(15,5))
sns.boxplot(x=cars.drivetrain, y=np.log(cars.price));
plt.ylabel("Log-Price")
plt.show() 


# 
# <b>Insight 8:</b>
# * Car prices look similar across different drive train types. That being said, 4WD cars have higher prices than average whereas FWD vehicles stand out as the most affordable type.

# In[462]:


plt.subplots(figsize=(15,5))
sns.boxplot(x=cars.transmission, y=np.log(cars.price));
plt.ylabel("Log-Price")
plt.show() 


# 
# <b>Insight 9:</b>
# * Transmission types do not make any difference when it comes to price but Manual vehicles sell cheaper than automatic and CVTs in general.

# In[463]:


plt.subplots(figsize=(15,5))
sns.boxplot(x=cars.fuel_type, y=np.log(cars.price));
plt.ylabel("Log-Price")
plt.show() 


# 
# <b>Insight 10:</b>
# * As per fuel type, electric cars are more expensive as expected because they are made with high technology. Meanwhile, hybrid and diesel vehicles are above average too. Technology aspect applies for hybrid cars too. About diesel vehicles, I assume it is due to large engine vehicles like pickups using diesel fuels.

# In[464]:


plt.subplots(figsize=(15,5))
sns.boxplot(x=cars.passengers, y=np.log(cars.price));
plt.ylabel("Log-Price")
plt.show() 


# 
# <b>Insight 11:</b>
# * In terms of passenger capacity, the most popular option is 5. It is the category that is closer to the average the most too. Moving further from 5 to both sides increase price. This should be due to the fact that cars with lower passenger capacity tend to be sport cars and those with higher passenger seats are larger cars like minivans which require a bigger engine with high cost. 

# In[465]:


plt.subplots(figsize=(15,5))
sns.boxplot(x=cars.vin_provided, y=np.log(cars.price));
plt.ylabel("Log-Price")
plt.show() 


# 
# <b>Insight 12:</b>
# * An interesting result: if a seller provides VIN number in the listing, its selling price tend to be higher than the others most of the time. Looks like trust is not something you get for free!

# In[466]:


plt.subplots(figsize=(15,5))
sns.boxplot(x=cars.carfax_provided, y=np.log(cars.price));
plt.ylabel("Log-Price")
plt.show() 


# <b>Insight 13:</b>
# * Same thing for CARFAX report. If it is available, selling prices is mostly higher.

# In[467]:


plt.subplots(figsize=(15,5))
sns.boxplot(x=cars.cylinders, y=np.log(cars.price));
plt.ylabel("Log-Price")
plt.show() 


# 
# 
# <b>Insight 14:</b>
# * More cylinders is associated with higher prices. Should be a result of bigger cars with large engine or/and that of sport cars.  

# In[468]:


plt.subplots(figsize=(15,5))
sns.boxplot(x=cars.vintage, y=np.log(cars.price));
plt.ylabel("Log-Price")
plt.show() 


# 
# 
# <b>Insight 15:</b>
# * This is surprising because one can expect vintage cars to be most expensive. However, it is also noteworthy that sport vehicles that skew the data are mostly new cars so two fields should be cancelling out each other's effect. It would be interesting to see the impact a combination of both fields might have. We'll explore that in the modelling phase!   

# In[469]:


text_list = list(cars.description.astype('str'))
text = '-'.join(text_list)
my_stop_words =["car","power","vehicle","available","Please","drive","price","great","call","vehicles",
                "Control","customer"]+list(STOPWORDS)
wordcloud = WordCloud(stopwords=my_stop_words, collocations=False).generate(text)


# In[470]:


plt.figure(figsize=(20,12))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# <b>Insight 16:</b>
# 
# As you might recall already, I merged trim and description fields earlier because they both contain long strings that is not feasible for numeric or categorical transformation. However, the strings stored in both fields are still valuable in the sense that we can capture most common keywords and try to understand what sort of information sellers put on most listings. This would help us know more about popular features on used vehicles and also help us perform a better data collection in further projects.
# 
# By applying a WordCloud, a popular tool for sentiment analysis, I have been able to see:
# * Cars with heated seats seem to be on demand. The word cloud also captured the words 'Rear' and 'Front'. Those could be in the sense of wheels as well as seats.
# * The words 'Certified', 'Pre', 'Owned' could be associated with Certified Pre Owned Vehicles. It should be an on-demand feature on used cars.
# * Warranty on used cars is also popular. One might say that most used cars come with a warranty. It could be useful to have a boolean field that checks whether a car has warranty or not. 
# * CARFAX is important and it is good that we already have a separate field for that. The word 'Safety' is also popular in a similar vein.
# * Credit is another common tag that the Wordcloud has captured. Should be purely due to the fact that 3/4 of the vehicles on our dataset is listed by dealers.
# * Camera, bluetooth, sunroof, leather(seats perhaps) are other features that most cars seem to have.

# In[471]:


searchfor = ['one owner', 'single owner','one-owner', '1 owner','sole owner','only owner'] #potential tags

cars.loc[cars['description'].str.contains('|'.join(searchfor),na=False,flags=re.IGNORECASE), "one_owner"] = 1
cars.loc[cars.one_owner != 1, 'one_owner'] = 0  
cars.one_owner.value_counts()


# In[472]:


plt.subplots(figsize=(15,5))
sns.boxplot(x=cars.one_owner, y=np.log(cars.price));
plt.ylabel("Log-Price")
plt.show() 


# <b>Insight 17:</b>
# 
# From my own experience being in the market for a used car recently, I think an important factor people seem to justifiably associate prices is the number of previous users. I figured through my personal experience that the fewer previous owners a car has the more expensive its asking price will be. The principle applies to the accident status: If a car is accident free, it could be higher in price.
# 
# * A <b>No. of owners</b> column would come in handy. In fact, the code above checks whether the improved description field has such info. Understandably, I had to limit my scope to look for certain vehicles, perhaps those with single owners.
# * And voila! Cars with one owner tags in the description are associated with higher prices.

# #### Adding new features based on the word cloud
# * <b>Heated Seats</b>

# In[473]:


cars.loc[cars['description'].str.contains('heated seats',na=False,flags=re.IGNORECASE), "heated_seats"] = 1
cars.loc[cars.heated_seats != 1, 'heated_seats'] = 0  
cars.heated_seats.value_counts()


# In[474]:


plt.subplots(figsize=(15,5))
sns.boxplot(x=cars.heated_seats, y=np.log(cars.price));
plt.ylabel("Log-Price")
plt.show() 


# * <b>Sunroof</b>

# In[475]:


cars.loc[cars['description'].str.contains('sunroof',na=False,flags=re.IGNORECASE), "sunroof"] = 1
cars.loc[cars.sunroof != 1, 'sunroof'] = 0  
cars.sunroof.value_counts()


# In[476]:


plt.subplots(figsize=(15,5))
sns.boxplot(x=cars.sunroof, y=np.log(cars.price));
plt.ylabel("Log-Price")
plt.show() 


# * <b>Camera</b>

# In[477]:


cars.loc[cars['description'].str.contains('camera',na=False,flags=re.IGNORECASE), "camera"] = 1
cars.loc[cars.camera != 1, 'camera'] = 0  
cars.camera.value_counts()


# In[478]:


plt.subplots(figsize=(15,5))
sns.boxplot(x=cars.camera, y=np.log(cars.price));
plt.ylabel("Log-Price")
plt.show() 


# * One thing I worried a lot about when looking for a used vehicle was whether it had any accients. Let's create a new field called <b>accident_free</b>.

# In[479]:


searchfor = ['no accident', 'accident free','no-accident', 'accident-free','no damage','no-damage','no dent','no-dent'] 
# I included no damage tags as well because they are essentially measuring the similar thing.

cars.loc[cars['description'].str.contains('|'.join(searchfor),na=False,flags=re.IGNORECASE), "accident_free"] = 1
cars.loc[cars.accident_free != 1, 'accident_free'] = 0  
cars.accident_free.value_counts()


# In[480]:


plt.subplots(figsize=(15,5))
sns.boxplot(x=cars.accident_free, y=np.log(cars.price));
plt.ylabel("Log-Price")
plt.show() 


# * <b> Rebuilt status</b>

# In[481]:


cars.loc[cars['description'].str.contains('rebuilt',na=False,flags=re.IGNORECASE), "rebuilt"] = 1
cars.loc[cars.rebuilt != 1, 'rebuilt'] = 0  
cars.rebuilt.value_counts()


# In[303]:


plt.subplots(figsize=(15,5))
sns.boxplot(x=cars.rebuilt, y=np.log(cars.price));
plt.ylabel("Log-Price")
plt.show() 


# In[483]:


cars.loc[cars['description'].str.contains('clean title',na=False,flags=re.IGNORECASE), "clean_title"] = 1
cars.loc[cars.clean_title != 1, 'clean_title'] = 0  
cars.clean_title.value_counts()


# In[484]:


plt.subplots(figsize=(15,5))
sns.boxplot(x=cars.clean_title, y=np.log(cars.price));
plt.ylabel("Log-Price")
plt.show() 


# <b> Insight 18:</b> This is by far the most interesting result from our new variables. The result should have been the other way around. I assume the way we created this column is to blame! We have been able to get title info from only 500 rows (1%) and tagged the rest as NOT HAVING CLEAN TITLE.

# In[485]:


cars.loc[cars['description'].str.contains('leather seat',na=False,flags=re.IGNORECASE), "leather_seats"] = 1
cars.loc[cars.leather_seats != 1, 'leather_seats'] = 0  
cars.leather_seats.value_counts()


# In[486]:


plt.subplots(figsize=(15,5))
sns.boxplot(x=cars.leather_seats, y=np.log(cars.price));
plt.ylabel("Log-Price")
plt.show() 


# In[487]:


cars.loc[cars['description'].str.contains('bluetooth',na=False,flags=re.IGNORECASE), "bluetooth"] = 1
cars.loc[cars.bluetooth != 1, 'bluetooth'] = 0  
cars.bluetooth.value_counts()


# In[488]:


plt.subplots(figsize=(15,5))
sns.boxplot(x=cars.bluetooth, y=np.log(cars.price));
plt.ylabel("Log-Price")
plt.show() 


# In[490]:


# Sketching our correlating matrix to see if there are any further insights we can drive.

cars_corr = cars.corr()

# figure size
plt.figure(figsize=(16,8))

# heatmap
sns.heatmap(cars_corr, cmap="YlGnBu", annot=True)
plt.show()


# <b>Insight 19:</b> As displayed in the correlation matrix above, <b>price</b> looks correlated with <b>age</b>, <b>mileage</b> and <b>cylinder</b> variables. Correlation is often confused with causation. The graph above only tells us that there are positive or negative relationship between price and the other highlighted variables. And that's it. It does not imply that cylinders are what make vehicles expensive. There could be a 3rd variable, total engine volume for instance (we removed this variable from our dataset earlier because it had data quality issues.) that is causing prices to get higher and since cylinders are engine size are related, they have a correlation. That being said, causation is often hard to detect and correlation is good enough for basic relationship analysis until we can locate causation!
# 
# Another important point to make here is that this matrix might show weak correlation between our target variable, price in this case, and other variables. However, we should not rule out the possibility that some variables have collective relationship with the targer variable. That is, a combination of several different fields might be relevant to exlain price differences and such a quest warrants further and deeper modelling.

# ### Step 7: Preprocessing for Modelling
# 
# We will now process the data and do some feature engineering for our machine learning models.

# In[491]:


cars.head()


# * Removing unnecessary columns from my dataset

# In[492]:


cars_new= cars.drop(columns=['id','first_date_seen','last_date_seen','year','model',
                             'description','longitude','latitude','classifications'])


# In[493]:


cars_new.head()


# #### Let's remove some outliers

# * In order to have the make field a feasible predictor, I remove vehicles under the 'Other' category that I earlier created for brands which have less than 15 listings under them. They make up around 78 listings in total.

# In[494]:


cars_new=cars_new[cars_new.make!='Other']


# In[495]:


cars_new.describe()


# * Let's see if we have potential outliers for our target variables by the Interquartile range method.
# 
#  <b>Price</b> 

# In[496]:


sns.boxplot(np.log(cars_new.price)) # A lot of outliers actually. Let's remove them for our model.


# In[497]:


#Define a function to determine outlier boundaries
def outlier_limits(col):
  Q3, Q1 = np.nanpercentile(col, [75, 25])
  IQR = Q3 - Q1
  UL = Q3 + 1.5*IQR   #upper limit
  LL = Q1 - 1.5*IQR   #lower limit
  return UL, LL


# In[498]:


#Apply the function to your data
outlier_limits(cars_new.price)
UL, LL = outlier_limits(cars_new["price"])


# In[499]:


cars_new=cars_new[(cars_new.price > LL) & (cars_new.price  < UL)]


# In[500]:


sns.boxplot(cars_new.price)


# In[501]:


cars_new.price.describe()


#  <b>Mileage</b> 

# In[502]:


sns.boxplot(np.log(cars_new.price)) # A lot of duplicates here as well. Let's remove them for our model.


# In[503]:


outlier_limits(cars_new.mileage)
UL, LL = outlier_limits(cars_new["mileage"])


# In[504]:


cars_new=cars_new[(cars_new.mileage > LL) & (cars_new.mileage  < UL)]


# In[505]:


sns.boxplot(cars_new.mileage)


# * Let's get our dummy variables for the categorical variables.

# In[506]:


dummy_columns=['make','color','body_type','drivetrain','transmission','fuel_type']

cars_new_expanded = pd.get_dummies(cars_new)
cars_new_expanded.head()


# In[507]:


pd.set_option('display.float_format', lambda x: '%.6f' % x) #Just a formatting script

std = StandardScaler()
cars_new_expanded_std = std.fit_transform(cars_new_expanded)
cars_new_expanded_std = pd.DataFrame(cars_new_expanded_std, columns = cars_new_expanded.columns)
print(cars_new_expanded_std.shape)
cars_new_expanded_std.head()


# In[508]:


X_train, X_test, y_train, y_test = train_test_split(cars_new_expanded_std.drop(columns = ['price']), cars_new_expanded_std[['price']])
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ### Step 8: Modelling
# 
# With dummy variables added to our dataset, I ended up with a lot of variables on my hand. For this reason, I am using SelectKBest function from sklearn to choose optimal number of variables from the dataset.
# 
# I randomly decide that the final number of variables could be anything from 3 to 100.

# In[509]:


column_names = cars_new_expanded_std.drop(columns = ['price']).columns

no_of_features = []
r_squared_train = []
r_squared_test = []

for k in range(3, 100, 2):
    selector = SelectKBest(f_regression, k = k)
    X_train_transformed = selector.fit_transform(X_train, y_train)
    X_test_transformed = selector.transform(X_test)
    regressor = LinearRegression()
    regressor.fit(X_train_transformed, y_train)
    no_of_features.append(k)
    r_squared_train.append(regressor.score(X_train_transformed, y_train))
    r_squared_test.append(regressor.score(X_test_transformed, y_test))
    
sns.lineplot(x = no_of_features, y = r_squared_train, legend = 'full')
sns.lineplot(x = no_of_features, y = r_squared_test, legend = 'full')


# We get an R score of around 0.725 with around 60 variables. It goes up to 0.75 with nearly 100 variables. It is not worth having extra 45 variables for an 0.025 additional points.
# 
# I keep k as 60 and let the machine choose best variables.

# In[510]:


selector = SelectKBest(f_regression, k = 60)
X_train_transformed = selector.fit_transform(X_train, y_train)
X_test_transformed = selector.transform(X_test)
column_names[selector.get_support()]


# In[511]:


def regression_model(model):
    """
    Will fit the regression model passed and will return the regressor object and the score
    """
    regressor = model
    regressor.fit(X_train_transformed, y_train)
    score = regressor.score(X_test_transformed, y_test)
    return regressor, score


# In[512]:


model_performance = pd.DataFrame(columns = ["Features", "Model", "Score"])

models_to_evaluate = [LinearRegression(), Ridge(), Lasso(), SVR(), RandomForestRegressor(), MLPRegressor()]

for model in models_to_evaluate:
    regressor, score = regression_model(model)
    model_performance = model_performance.append({"Features": "Linear","Model": model, "Score": score}, ignore_index=True)

model_performance


# ### Conclusion:
# I got a maximum R^2 score of 0.877 with DecisionTreeRegressor. If I had more time, I would explore further transformations and fine-tuning like polynomial features and try to increase model performance. 
# 
# I would also love to deploy this model with a user-friendly interface.
# 
# Lastly, I would be curious to have more historic data and account for COVID's impact on the used car market. We've all witnessed that the pandemic has hit all the broader auto industry hard. The dataset provided had been collected in the early days of COVID.

# In[ ]:




