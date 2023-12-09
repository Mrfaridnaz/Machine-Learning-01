#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Missing values
 1. Remove the row
 2. Impute the data or fill
    a. Univariant (use simple imputer class)
       i. Numerical (mean, median, ramdom,end of distribution)
       ii. categorical (mode, missing)
    b. Multivariant
       i. KNN imputer
       ii. iterative imputer (a class)


# In[ ]:


Complete-case analysis (CCA), also called "List-wise deletion" of cases, consists in discarding
observation(rows) where values in any of the variables (cols) are missing.

Complete case analysis means literally analyzing only those observation for which there is 
information in all variables in the dataset.


# In[ ]:


Assumption for CCA

1. Missing completely at random (MCAR)


# In[ ]:


Advantages

Read frorm the vedio 10:00 


# In[ ]:


get_ipython().run_line_magic('pinfo', 'CCA')

1. Data should be missing MCAR (5 % data mising not more than 5)

   


# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[9]:


df = pd.read_csv("data_science_job.csv")


# In[10]:


df.head()


# In[11]:


df.isnull().mean()*100


# In[12]:


df.shape


# In[ ]:


# apply CCA where the missing data is less than 5%.


# In[13]:


# print the cols name where the missing data is between 0% to 5%.

cols = [var for var in df.columns if df[var].isnull().mean() < 0.05 and df[var].isnull().mean()>0]
cols


# In[14]:


# This is the DataFrame in which the cols has missing values less tha 5%
df[cols].sample(5)


# In[ ]:


# Delete the rows with missing data
df[cols].dropna()


# In[17]:


# Remaining rows
len(df[cols].dropna())


# In[18]:


# Total rows
len(df)


# In[19]:


# How much the data remains after removing missing data rows

# remaining rows/total rows

len(df[cols].dropna())/len(df)*100


# In[21]:


new_df = df[cols].dropna() # Now this is the new dataframe
df.shape , new_df.shape


# In[22]:


# Draw the Histogram for whole dataset and it will print only for numeric data type

new_df.hist(bins = 50, density = True, figsize =(12,12))
plt.show()


# In[ ]:


# If you have numerial data , check the histogram before CCA and after CCA.


# In[26]:


fig = plt.figure()
ax = fig.add_subplot(111)

# Orignal data
df['training_hours'].hist(bins = 50, ax=ax, density = True, color = 'red')

# Data after CCA, the argument alpha makes the color transparent, so we can see the overlay of the
# 2 distributoions

new_df['training_hours'].hist(bins = 50, ax=ax, color = 'green',density = True, alpha = 0.8)


# In[19]:


fig = plt.figure()
ax = fig.add_subplot(111)

# Original Data
df['training_hours'].plot.density(color = 'red')

# Data after CCA
new_df['training_hours'].plot.density(color = 'green')

# we can see here , the Original and new_df line overlap each other.
# We can say that the missing data is completely at random


# In[20]:


fig = plt.figure()
ax = fig.add_subplot(111)

# Original Data
df['city_development_index'].plot.density(color = 'red')

# Data after CCA
new_df['city_development_index'].plot.density(color = 'green')


# In[21]:


fig = plt.figure()
ax = fig.add_subplot(111)

# Orignal data
df['city_development_index'].hist(bins = 50, ax=ax, density = True, color = 'red')

# Data after CCA, the argument alpha makes the color transparent, so we can see the overlay of the
# 2 distributoions

new_df['city_development_index'].hist(bins = 50, ax=ax, color = 'green',density = True, alpha = 0.8)


# In[22]:


fig = plt.figure()
ax = fig.add_subplot(111)

# Orignal data
df['experience'].hist(bins = 50, ax=ax, density = True, color = 'red')

# Data after CCA, the argument alpha makes the color transparent, so we can see the overlay of the
# 2 distributoions

new_df['experience'].hist(bins = 50, ax=ax, color = 'green',density = True, alpha = 0.8)


# ### Categorical Columns
# The ratio should be same or almost same after removing the data

# In[42]:


df['education_level'].value_counts()


# In[36]:


df['enrolled_university'].value_counts()/len(df)


# In[37]:


df['enrolled_university'].value_counts()/len(new_df)


# In[29]:


temp = pd.concat([
    # percentage of observation per category, orignal data
    df['enrolled_university'].value_counts()/len(df),
    
    # percentage of observation per category, CCA data
    df['enrolled_university'].value_counts()/len(new_df)
    
    
],
axis =1)

# add cols names
temp.columns = ['orignal','CCA']
temp


# In[34]:


df['enrolled_university'].value_counts()


# In[30]:


temp = pd.concat([
    # percentage of observation per category, orignal data
    df['education_level'].value_counts()/len(df),
    
    # percentage of observation per category, CCA data
    df['education_level'].value_counts()/len(new_df)
    
    
],
axis =1)

# add cols names
temp.columns = ['orignal','CCA']
temp


# In[26]:


temp = pd.concat([
    # percentage of observation per category, orignal data
    df['experience'].value_counts()/len(df),
    
    # percentage of observation per category, CCA data
    df['experience'].value_counts()/len(new_df)
    
    
],
axis =1)

# add cols names
temp.columns = ['orignal','CCA']
temp


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




