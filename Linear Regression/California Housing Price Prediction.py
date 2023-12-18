#!/usr/bin/env python
# coding: utf-8

# # Linear Regression Project: California Housing Price Prediction
# 
# A real estate agent and wants some help predicting housing prices for california. Task is to create a model for her that allows to put in a few features of a house and returns back an estimate of what the house would sell for. I decided that Linear Regression might be a good path to solve this problem! The agent gives some information about a bunch of houses in regions of the United States,it is all in the data set: ("California_Housing.csv").
# 
# ### The data contains the following columns:
# 
# longitude (signed numeric - float) : Longitude value for the block in California, USA
# 
# latitude (numeric - float ) : Latitude value for the block in California, USA
# 
# housing_median_age (numeric - int ) : Median age of the house in the block
# 
# total_rooms (numeric - int ) : Count of the total number of rooms (excluding bedrooms) in all houses in the block
# 
# total_bedrooms (numeric - float ) : Count of the total number of bedrooms in all houses in the block
# 
# population (numeric - int ) : Count of the total number of population in the block
# 
# households (numeric - int ) : Count of the total number of households in the block
# 
# median_income (numeric - float ) : Median of the total household income of all the houses in the block
# 
# ocean_proximity (numeric - categorical ) : Type of the landscape of the block
# 
# median_house_value (numeric - int ) : Median of the household prices of all the houses in the block

# # Check out the data

# ### Import Libraries

# In[223]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[224]:


house_price=pd.read_csv('California-housing.csv')


# In[225]:


house_price.head()


# In[226]:


house_price.info()


# In[227]:


house_price.describe()


# # Look for missing value in dataset

# In[228]:


sns.heatmap(house_price.isnull(),yticklabels=False,cbar=False)


# total_bedrooms columns have missing values 

# In[229]:


house_price[house_price['total_bedrooms'].isnull()].head()


# In[230]:


house_price['total_bedrooms'].fillna(house_price['total_bedrooms'].mean(),inplace=True)


# No Missing values left

# In[231]:


sns.heatmap(house_price.isnull(),yticklabels=False,cbar=False)


# # Exploratory Data Analysis 

# Plotting the relation between all the features

# Plotting a distribution plot of Price  

# In[68]:


sns.set_style('whitegrid')
sns.displot(house_price['median_house_value'],kde=True,bins=20)
plt.xlabel('House Prices')
plt.show()


# Graph of Population vs Ocean Proximity

# In[107]:


plt.figure(figsize=(10,8))
sns.barplot(data=house_price,x='ocean_proximity',y='population')
plt.xlabel('Ocean Proximity')
plt.ylabel('Population')
plt.show()


# Graph of Ocean Proximity vs House Price

# In[112]:


plt.figure(figsize=(10,8))
sns.boxplot(data=house_price,x='ocean_proximity',y='median_house_value',)
plt.xlabel('Ocean Proximity')
plt.ylabel('House Price')
plt.show()


# Graph of Household Income vs Population

# In[151]:


plt.figure(figsize=(10,8))
sns.jointplot(data=house_price,x='median_income',y= 'population')
plt.xlabel('Household Income')
plt.ylabel('Population')
plt.show()


# In[152]:


house_price.head()


# # Convert categorical column in the dataset to numerical data

# Getting dummie values for ocean_proximity column

# In[232]:


house_price=pd.get_dummies(house_price,columns=['ocean_proximity'],dtype=int)


# In[233]:


house_price.head()


# In[234]:


house_price=house_price.apply(round)


# In[235]:


house_price.head()


# # Training a Linear Regression Model

# Defining Features (X) and Target Values(y)

# In[236]:


X=house_price.drop('median_house_value',axis=1)
y=house_price['median_house_value']


# In[237]:


from sklearn.model_selection import train_test_split


# In[238]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)


# In[239]:


from sklearn.linear_model import LinearRegression


# In[240]:


lm=LinearRegression()


# Fitting the model with training data

# In[241]:


lm.fit(X_train,y_train)


# # Model Evaluation 

# In[242]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# # Predictions from our Model

# In[243]:


predictions = lm.predict(X_test)


# In[244]:


plt.scatter(y_test,predictions)


# In[245]:


sns.displot((y_test-predictions),bins=30,kde=True)


# ## Regression Evaluation Metrics
# 
# 
# Here are three common evaluation metrics for regression problems:
# 
# **Mean Absolute Error** (MAE) is the mean of the absolute value of the errors:
# 
# $$\frac 1n\sum_{i=1}^n|y_i-\hat{y}_i|$$
# 
# **Mean Squared Error** (MSE) is the mean of the squared errors:
# 
# $$\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2$$
# 
# **Root Mean Squared Error** (RMSE) is the square root of the mean of the squared errors:
# 
# $$\sqrt{\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2}$$
# 
# 
# 

# In[246]:


from sklearn import metrics


# In[249]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

