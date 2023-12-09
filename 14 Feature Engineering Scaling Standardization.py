#!/usr/bin/env python
# coding: utf-8

# ### Feature Engineering
# 

# #### 1.  Feature Transformation

# In[ ]:


1. Missing Values
2. Handling Categorical Values
3. Outliers Detection


# #### 2. Feature Scaling

# In[ ]:


2. Feature Scaling:
    Feature scaling is a technique to understand the independent feature present in the data 
    in a fixed range.
    
    Types:
        a. Standardization
        b. Normalization
           i. Min max scaler
           ii. Robust scaler


# ##### a. Standardization or Z-Score Normalization

# In[ ]:


x' = xi-x'mean/std

After transformation the x value,
the new x' has Mean = 0 and std=1
mean will be for both coulmns at (0,0)


# In[ ]:


You will have train test split before Standadization


# ##### Train_Test_Split

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv("PlacementCGPA.csv")


# In[3]:


df.shape


# In[4]:


df=df.iloc[:100,1:]


# In[5]:


df


# In[11]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df.drop('Placement',axis = 1), 
                                                 df["Placement"],test_size = 0.3,  
                                                 random_state = 0)
                                                          
                                                          
                                                         


# In[12]:


x_train.shape , x_test.shape


# #### StandarScaler

# In[20]:


from sklearn.preprocessing import StandardScaler

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the scaler on the training data
# Machine learns only from the training data and transform both train and test
scaler.fit(x_train)

# Transform both training and testing data using the scaler
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Now x_train_scaled and x_test_scaled contain your scaled data
# It takes the DataFrame and returns the numpy array


# In[21]:


scaler.mean_ # Age mean and the salary mean


# In[25]:


# covert the numpy array into DataFrame
x_train_scaled = pd.DataFrame(x_train_scaled, columns = x_train.columns)
x_test_scaled = pd.DataFrame(x_test_scaled, columns = x_test.columns)


# In[26]:


x_train # Before scaling


# In[27]:


x_train_scaled # After scaling


# In[28]:


np.round(x_train_scaled.describe(),1)


# ### Effect of Scaling

# In[29]:


import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# This is before scaling
ax1.scatter(x_train["CGPA"], x_train["IQ"])
ax1.set_title("Before Scaling")

ax2.scatter(x_train_scaled["CGPA"], x_train_scaled["IQ"], color='red')
ax2.set_title("After Scaling")

plt.show()


# In[39]:


import seaborn as sns
fig, (ax1, ax2) = plt.subplots (ncols=2, figsize=(12, 5))

# before scaling

ax1.set_title('Before Scaling')

sns.kdeplot(x_train['CGPA'], ax=ax1)

sns.kdeplot(x_train['IQ'], ax=ax1)

# after scaling

ax2.set_title('After Standard Scaling')

sns.kdeplot(x_train_scaled['CGPA'], ax=ax2)

sns.kdeplot(x_train_scaled['IQ'], ax=ax2)
plt.show()


# ### Comparison of Distributions

# In[42]:


import matplotlib.pyplot as plt
import seaborn as sns

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# Before Scaling
ax1.set_title("CGPA distribution before scaling")
sns.kdeplot(x_train["CGPA"], ax=ax1)

# After Scaling
ax2.set_title("CGPA distribution after scaling")
sns.kdeplot(x_train_scaled["CGPA"], ax=ax2)

plt.show()


# In[43]:


import matplotlib.pyplot as plt
import seaborn as sns

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# Before Scaling
ax1.set_title("CGPA distribution before scaling")
sns.kdeplot(x_train["IQ"], ax=ax1)

# After Scaling
ax2.set_title("CGPA distribution after scaling")
sns.kdeplot(x_train_scaled["IQ"], ax=ax2)

plt.show()


# ### Normalization:

# In[ ]:


Normalization:
    Normalization is a technique often applied as part of data prereation
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

You will get Scaled distribution betweeen 0 to 1 by min-max Scale formula.
First and last values always will be 0 and 1 respecively.


# In[ ]:


import numpy as np # Linear Algebra
import pandas as pd # Data Processing
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('wine_data.csv', header = None, usecols = [0,1,2])
df.columns= ['Class label', 'Alcohol','Malic acid']


# In[ ]:


df


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df.drop('Class label',axis = 1), 
                                                 df["Class label"],test_size = 0.3,  
                                                 random_state = 0)
                                                          
                                                          
                                                         


# In[ ]:


x_train.shape , x_test.shape


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Fit the scaler to your data and transform it
scaler.fit_transform(x_train)


# In[ ]:


# Transform both training and testing data using the scaler
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Now x_train_scaled and x_test_scaled contain your scaled data
# It takes the DataFrame and returns the numpy array


# In[ ]:


# covert the numpy array into DataFrame
x_train_scaled = pd.DataFrame(x_train_scaled, columns = x_train.columns)
x_test_scaled = pd.DataFrame(x_test_scaled, columns = x_test.columns)


# In[ ]:


x_train


# In[ ]:


np.round(x_train_scaled.describe(),1)

# Min value will be for both and Max is 1 


# In[ ]:


import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# This is before scaling
ax1.scatter(x_train["Alcohol"], x_train["Malic acid"], c=y_train)
ax1.set_title("Before Scaling")

ax2.scatter(x_train_scaled["Alcohol"], x_train_scaled["Malic acid"], c=y_train)
ax2.set_title("After Scaling")

plt.show()


# In[ ]:


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


# In[ ]:


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


# #### Normalization VS Standardization

# In[ ]:


get_ipython().run_line_magic('pinfo', 'required')
2. Most of the problems are solved by Standardization.
3. Normalization is used where the Min and Max values are known.


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




