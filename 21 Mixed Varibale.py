#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv("Mix_titanic.csv")


# In[3]:


df.head()


# In[4]:


df['number'].unique()


# In[8]:


df['number'].value_counts().plot(kind='bar')


# In[13]:


# Extract numerical part

df['number_numerical'] = pd.to_numeric(df["number"], errors = 'coerce', downcast = 'integer')


# In[16]:


# Extract the categorical part

df['number_categorical'] = np.where(df['number_numerical'].isnull(), df['number'],np.nan)


# In[17]:


df.head()


# In[18]:


df['Cabin'].unique()


# In[20]:


df['Ticket'].unique()


# In[22]:


df['Cabin_num'] = df['Cabin'].str.extract('(\d+)') # capture numerical part
df['Cabin_cat'] = df['Cabin'].str[0] # capture the first letter


# In[23]:


df.head()


# In[25]:


df['Cabin_cat'].unique()


# In[29]:


df['Cabin_cat'].value_counts().plot(kind = 'bar')


# In[30]:


# Extract the last bit of ticket as number

df['ticket_num'] = df['Ticket'].apply(lambda s: s.split()[-1])
df['ticket_num'] = pd.to_numeric(df['ticket_num'],
                                errors = 'coerce',
                                downcast = 'integer')

# Extract the first part of ticket as category
df['ticket_cat'] = df['Ticket'].apply(lambda s: s.split()[0])
df['ticket_cat'] = np.where(df['ticket_cat'].str.isdigit(),np.nan,
                           df['ticket_cat'])


# In[31]:


df.head()


# In[ ]:




