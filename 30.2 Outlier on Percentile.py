#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('weight-height.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df['Height'].describe()


# In[6]:


import seaborn as sns


# In[7]:


sns.distplot(df['Height'])


# In[9]:


sns.boxplot(df['Height'], orient = 'h')


# In[14]:


upper_limit = df['Height'].quantile(0.99)
upper_limit


# In[15]:


lower_limit = df['Height'].quantile(0.01)
lower_limit


# In[16]:


new_df = df[(df['Height'] <= 74.78) & (df['Height'] >= 58.13)]


# In[17]:


new_df['Height'].describe()


# In[18]:


df['Height'].describe()


# In[19]:


sns.distplot(new_df['Height'])


# In[21]:


sns.boxplot(new_df['Height'], orient = 'h')


# In[22]:


# Capping --> Winsorization
df['Height'] = np.where(df['Height'] >= upper_limit,
        upper_limit,
        np.where(df['Height'] <= lower_limit,
        lower_limit,
        df['Height']))


# In[23]:


df.shape


# In[24]:


df['Height'].describe()


# In[25]:


sns.distplot(df['Height'])


# In[27]:


sns.boxplot(df['Height'], orient = 'h')


# In[ ]:




