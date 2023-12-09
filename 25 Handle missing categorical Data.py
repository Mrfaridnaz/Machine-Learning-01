#!/usr/bin/env python
# coding: utf-8

# ### 1. Categorical Data
#     Two types of technique
# 1.  Replace with most frequent
# 2.  Create a new category named missing
# 
# #### Most frequent Value Imputation
# 1. Mode (missing value filled by mode)
#     keep in mind:
#     Mode should be more than times than        
# others.
# 
# #### Missing category Imputation.
#    If the Data with more than 10% misisng value , dont use Mode,
# make a new category named missing.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("train.csv")


# In[3]:


df.columns


# In[4]:


df=pd.read_csv("train.csv", usecols = ['GarageQual','FireplaceQu','SalePrice'])


# In[5]:


df.head()


# In[8]:


# Check missing values

df.isnull().mean()*100


# ### Frequent Value Imputation

# In[9]:


df['GarageQual'].value_counts().plot(kind='bar')
plt.xlabel('GarageQual')
plt.ylabel('Number of House')


# In[10]:


df['GarageQual'].mode()


# In[25]:


# PDF for sale price where 'GarageQual' is TA
# PDF for sale price where 'GarageQual' is null

fig = plt.figure()
ax = fig.add_subplot(111)

df[df['GarageQual']=='TA']['SalePrice'].plot(kind = 'kde',ax=ax)

df[df['GarageQual'].isnull()]['SalePrice'].plot(kind = 'kde',ax=ax)

lines, labels = ax.get_legend_handles_labels()
labels = ['House with TA','House with NA']
ax.legend(lines, labels, loc = 'best')

plt.title('GarageQual')


# In[ ]:


# Change missing NA values with TA and check the chnanges


# In[13]:


# Store 'SalePrice' kde where 'GarageQual' == TA

temp = df[df['GarageQual']=='TA']['SalePrice']


# In[14]:


# Fill missing values by TA
df['GarageQual'].fillna('TA', inplace = True)


# In[15]:


df['GarageQual'].value_counts().plot(kind = 'bar')


# In[16]:


fig = plt.figure()
ax = fig.add_subplot(111)

temp.plot(kind = 'kde', ax=ax)

# Distributaion of the variable after imputation
df[df['GarageQual']=='TA']['SalePrice'].plot(kind = 'kde',ax=ax, color = 'red')


lines, labels = ax.get_legend_handles_labels()
labels = ['Original varibale','Imputed varibale']
ax.legend(lines, labels, loc = 'best')

# Add title
plt.title('GarageQual')


# In[ ]:


9:00


# In[32]:


df['FireplaceQu'].value_counts().plot(kind = 'bar')


# In[35]:


df['FireplaceQu'].mode()


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111)

df[df['FireplaceQu']=='Gd']['SalePrice'].plot(kind = 'kde',ax=ax)

df[df['FireplaceQu'].isnull()]['SalePrice'].plot(kind = 'kde',ax=ax)

lines, labels = ax.get_legend_handles_labels()
labels = ['House with Gd','House with NA']
ax.legend(lines, labels, loc = 'best')

plt.title('FireplaceQu')


# In[ ]:


10:00


# In[ ]:




