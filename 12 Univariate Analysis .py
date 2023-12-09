#!/usr/bin/env python
# coding: utf-8

# ### Independent analysis regarding the Column

# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np

import plotly.express as px
import bokeh


import scipy.stats as stats
from sklearn import datasets, linear_model

import statsmodels.api as sm


# In[3]:


import pandas as pd
df=pd.read_csv('titanic_train.csv')


# In[4]:


df.head()


# In[5]:


df.columns
# Print all the Column name


# In[ ]:


# In univariate you can take any one column to analys


# In[ ]:


# Basis understanding regarding the Coulumn or variable
# Categorical var || Numeric var ||


# In[7]:


cat_feature = [columns for columns in df.columns if df[columns].dtype == "O"]

# Print the Column that are object data type, "O" for Object Capital O
# Print the table with objective data type


# In[8]:


cat_feature


# In[11]:


df[cat_feature]


# In[12]:


# Print the Column that are Numeric data type, non-objective

Num_feature =[columns for columns in df.columns if df[columns].dtype!= "O"]


# In[14]:


df[Num_feature]


# In[15]:


# Performing the univariant analysis
# Two types of the Columns here (Cat_Column and Num_column)


# ## Categorical Columns

# In[16]:


cat_feature


# ### 1. Count Plot

# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=df, x="Embarked")
plt.show()


# In[20]:


sns.countplot(data=df, x="Sex")


# In[21]:


df["Sex"].value_counts()


# In[23]:


df["Sex"].value_counts().plot(kind = 'bar')


# ### 2. Pie Chart

# In[24]:


df["Sex"].value_counts().plot(kind = 'pie')


# In[25]:


df["Pclass"].value_counts().plot(kind='pie', autopct='%.2f')


# ## Numerical Column

# ### Histogram

# In[26]:


plt.hist(df["Age"])

# distribution regarding the age
# A histogram is a graph showing frequency distributions.

# It is a graph showing the number of observations within each given interval.


# ## 2. Distplot

# In[27]:


sns.distplot(df["Age"])

# Bins are the interval


# In[ ]:


# We can see from the graph, the probability of the people whose age between 20 to 40
# Most of the people 20 to 40
# There is interval by 10 year


# In[28]:


# create an another graph with 5 year of interval

plt.hist(df["Age"], bins = 5)


# ## 3. Boxplot

# In[30]:


sns.boxplot(df["Age"], orient = 'h')

# It represents the disperssion of the data in terms of quantile


# In[ ]:


# Lower fence || Q1 || Mean || IQR || Q3 || Upper fence || Outliers
# Box plot is imp for Univariant Analysis


# In[31]:


# Skweness of the Data


# In[ ]:


(100-38.910778230082705)
#61.08 % of the Data is Normal Distributed


# In[ ]:




