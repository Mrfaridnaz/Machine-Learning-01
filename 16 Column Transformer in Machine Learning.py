#!/usr/bin/env python
# coding: utf-8

# ### Columns Transformer

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder


# In[3]:


df = pd.read_csv("covid_toy.csv")


# In[4]:


df.sample(4)


# In[42]:


df["cough"].value_counts()


# In[43]:


df["city"].value_counts()


# In[44]:


df.isnull().sum()


# In[ ]:


gender | Nominal Data | One Hot Encoding 
Fever | Numerical data | Missing vlaues | Simple Imputer 
Cough | Ordinal Data | Ordinal Encoder 
City | Nominal Data | One Hot Encoding 
has_covid | nominal O/P | Label Encoding 


# In[62]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df.drop(columns = ['has_covid']),df['has_covid'],test_size=0.2)


# In[46]:


x_train


# ### 1. Without using the Column Tranformation

# In[47]:


# adding simple imputer to fever col
# Fill the missing values with its mean

si = SimpleImputer()
x_train_fever = si.fit_transform(x_train[['fever']])

# also the test data
x_test_fever = si.fit_transform(x_test[['fever']])

x_train_fever.shape


# In[48]:


# OrdinalEncoding --> cough

oe = OrdinalEncoder(categories = [['Mild','Strong']])
x_train_cough = oe.fit_transform(x_train[['cough']])

# also the test data
x_test_cough = oe.fit_transform(x_test[['cough']])

x_train_cough.shape


# In[49]:


x_train_cough


# In[52]:


# OneHotEncoding --> gender, city
# We will remove the first column from gender, city
# we will get 1 col from gender and 3 col from city

ohe = OneHotEncoder(drop = 'first' , sparse = False)
x_train_gender_city = ohe.fit_transform(x_train[['gender', 'city']])

# also the test data
x_test_gender_city = ohe.fit_transform(x_test[['gender', 'city']])

x_train_gender_city.shape


# In[54]:


x_train_gender_city

1 col for gender | 3 cols for city


# In[55]:


# Extraction Age
x_train_age = x_train.drop(columns = ['gender','fever','cough','city']).values

# also the test data
x_test_age = x_test.drop(columns = ['gender','fever','cough','city']).values

x_train_age  # this is separted Age col


# In[56]:


# add all the cols


x_train_transform = np.concatenate((x_train_age, x_train_fever, x_train_gender_city, x_train_cough), axis = 1)

# Also the test Data
x_test_transform = np.concatenate((x_test_age, x_test_fever, x_test_gender_city, x_test_cough), axis = 1)



# In[57]:


x_train_transform


# In[58]:


x_train_transform.shape


# ### Using the Column Tranformation

# In[59]:


from sklearn.compose import ColumnTransformer


# In[66]:


transformer = ColumnTransformer(transformers=[
    ('tnf1', SimpleImputer(), ['fever']),
    ('tnf2', OrdinalEncoder(categories=[['Mild', 'Strong']]), ['cough']),  
    
    # Replace 'column_name_here' with the actual column name
    ('tnf3', OneHotEncoder(sparse=False, drop='first'), ['gender', 'city'])
], remainder='passthrough')


# In[67]:


transformer.fit_transform(x_train)


# In[68]:


transformer.fit_transform(x_train).shape


# In[1]:


transformer


# In[69]:


transformer.transform(x_test).shape


# In[70]:


transformer.transform(x_train).shape


# In[ ]:


Done

