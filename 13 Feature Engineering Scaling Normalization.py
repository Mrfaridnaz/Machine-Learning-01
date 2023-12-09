#!/usr/bin/env python
# coding: utf-8

# ### Steps in Machine Learning

# #### 5.Feature Engineering:

# In[ ]:


A.Feature Transformation.
  •Missing values.
  •Handling categorical Feature.
  •Outliers Detection.

  •Feature Scaling.
        Types:
        a. Standardization
        b. Normalization
           i. Min max scaler 
           ii. Robust scaler

B.Feature Construction.
C.Feature Selection.
D.Feature Extraction.


# ### Normalization:

# In[ ]:


Normalization:
    Normalization is a technique often applied as part of data prepration
    for machine learning. The goal of normalization is to change the value of numeric
    columns in the dataset to use a common scale, without distorting differences in 
    the ranges of values of lossing information.


# In[ ]:


1. Min Max Scale
2. Mean normalization
3. Max absolute 
4. Robust Scaling


# #### 1. Min Max Scale

# In[ ]:


Xi' = Xi-Xmin/Xmax-Xmin

It will Scale distribution betweeen 0 to 1 by min-max Scale formula


# In[10]:


import numpy as np # Linear Algebra
import pandas as pd # Data Processing
import matplotlib.pyplot as plt
import seaborn as sns


# In[11]:


df = pd.read_csv('wine_data.csv', header = None, usecols = [0,1,2])
df.columns= ['Class label', 'Alcohol','Malic acid']


# In[12]:


df


# In[13]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df.drop('Class label',axis = 1), 
                                                 df["Class label"],test_size = 0.3,  
                                                 random_state = 0)                          


# In[14]:


x_train.shape , x_test.shape


# In[15]:


x_train


# In[19]:


from sklearn.preprocessing import MinMaxScaler

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Fit the scaler to your data and transform it
scaler.fit_transform(x_train)


# In[20]:


# Transform both training and testing data using the scaler
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Now x_train_scaled and x_test_scaled contain your scaled data
# It takes the DataFrame and returns the numpy array


# In[25]:


# covert the numpy array into DataFrame
x_train_scaled = pd.DataFrame(x_train_scaled, columns = x_train.columns)
x_test_scaled = pd.DataFrame(x_test_scaled, columns = x_test.columns)


# In[26]:


x_train_scaled


# In[24]:


np.round(x_train_scaled)


# In[15]:


np.round(x_train_scaled.describe(),1)

# Min value will be for both and Max is 1 


# In[17]:


import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# This is before scaling
ax1.scatter(x_train["Alcohol"], x_train["Malic acid"], c=y_train)
ax1.set_title("Before Scaling")

ax2.scatter(x_train_scaled["Alcohol"], x_train_scaled["Malic acid"], c=y_train)
ax2.set_title("After Scaling")

plt.show()


# In[18]:


import seaborn as sns
fig, (ax1, ax2) = plt.subplots (ncols=2, figsize=(12, 5))

# before scaling

ax1.set_title('Before Scaling')

sns.kdeplot(x_train['Alcohol'], ax=ax1)

sns.kdeplot(x_train['Malic acid'], ax=ax1)

# after scaling

ax2.set_title('After Standard Scaling')

sns.kdeplot(x_train_scaled['Alcohol'], ax=ax2)

sns.kdeplot(x_train_scaled['Malic acid'], ax=ax2)
plt.show()


# In[20]:


import matplotlib.pyplot as plt
import seaborn as sns

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# Before Scaling
ax1.set_title("Alcohol distribution before scaling")
sns.kdeplot(x_train["Alcohol"], ax=ax1)

# After Scaling
ax2.set_title("Alcohol distribution after scaling")
sns.kdeplot(x_train_scaled["Alcohol"], ax=ax2)

plt.show()


# #### 2. Mean normalization

# In[ ]:


Xi' = Xi-Xmean/Xmax-Xmin


# In[ ]:


Rarely used, there is no class in sklearn for this.
you will have to make the code by yourself
It is needed where the cencered data is required.


# #### 3. MaxAbsulute Scaling

# In[ ]:


Xi' = Xi/|Xmax|


# In[ ]:


There is a class in sklearn named maxabsscaler
used in parshe data mean the data who has numbers of zeros


# #### 4. Robust Scaling

# In[ ]:


Xi' = Xi-Xmedian/IQR


# In[ ]:


RobustScaler class named

it is robust to outliers


# ### Normalization VS Standardization

# In[ ]:


get_ipython().run_line_magic('pinfo', 'required')
2. Most of the problems are solved by Standardization.
3. Normalization is used where the Min and Max values are known.


# In[ ]:





# In[ ]:





# In[ ]:




