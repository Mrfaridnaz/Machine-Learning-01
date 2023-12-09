#!/usr/bin/env python
# coding: utf-8

# In[72]:


import pandas as pd
import random


# In[73]:


data = {'Category': ['A', None, 'C', 'A', 'B', None, 'A', 'B', None, 'A'],
        'Value': [10, 20, 30, 40, None, 60, 70, 80, None, 100]}
df = pd.DataFrame(data)


# In[74]:


df


# In[75]:


def fill_missing_with_random(df, column_name):
    # Create a list of unique values in the specified column
    unique_values = df[column_name].dropna().unique()
    
    # Iterate through the DataFrame and fill missing values with random values from the same column
    for index, row in df.iterrows():
        if pd.isna(row[column_name]):
            df.at[index, column_name] = random.choice(unique_values)


# In[71]:


fill_missing_with_random(df, 'Category')


# In[76]:


# Fill missing categorical values in the 'Category' column
fill_missing_with_random(df, 'Category')

# Display the updated DataFrame
print(df)


# ### Real Data

# In[108]:


df=pd.read_csv("train.csv", usecols = ['GarageQual','FireplaceQu','SalePrice'])


# In[109]:


df1= df


# In[111]:


df.shape, df1.shape


# In[112]:


df1.isnull().sum()


# In[113]:


df1['FireplaceQu'].value_counts()


# In[114]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


# In[115]:


df["FireplaceQu"].value_counts().plot(kind = 'bar', color = 'green')


# In[116]:


df1["FireplaceQu1"].value_counts().plot(kind = 'bar')


# In[103]:


df1["FireplaceQu1"].value_counts()


# In[67]:


df["FireplaceQu"].value_counts().plot(kind = 'kde')


# In[81]:


df1=pd.read_csv("train.csv", usecols = ['GarageQual','FireplaceQu','SalePrice'])


# In[82]:


df1.rename(columns={'GarageQual': 'GarageQual1', 'FireplaceQu':'FireplaceQu1','SalePrice':'SalePrice1'}, inplace=True)


# In[83]:


def fill_missing_with_random(df1, column_name):
    # Create a list of unique values in the specified column
    unique_values = df1[column_name].dropna().unique()
    
    # Iterate through the DataFrame and fill missing values with random values from the same column
    for index, row in df1.iterrows():
        if pd.isna(row[column_name]):
            df1.at[index, column_name] = random.choice(unique_values)


# In[85]:


fill_missing_with_random(df1, 'FireplaceQu1')


# In[86]:


df1


# In[87]:


df1.isnull().sum()


# In[93]:


df1["FireplaceQu1"].value_counts().plot(kind = 'kde')
df["FireplaceQu"].value_counts().plot(kind = 'kde')


# In[ ]:




