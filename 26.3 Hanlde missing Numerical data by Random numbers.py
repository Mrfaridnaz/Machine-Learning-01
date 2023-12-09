#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('titanic_train.csv', usecols = ['Age','Fare','Survived'])


# In[3]:


df


# In[4]:


df.isnull().mean()


# In[5]:


x=df.drop(columns = ['Survived'])
y = df['Survived']


# In[6]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 2)


# In[7]:


x_train.shape, x_test.shape


# In[9]:


x_train.isnull().mean()*100

# 20% missning values in age


# In[15]:


x_train['Age'].plot(kind='kde')


# In[25]:


df1 = pd.read_csv('titanic_train.csv', usecols = ['Age','Fare','Survived'])


# In[26]:


df1.rename(columns={'Age': 'Age1', 'Fare':'Fare1','Survived':'Survived1'}, inplace=True)


# In[27]:


x1=df1.drop(columns = ['Survived1'])
y1 = df1['Survived1']


# In[28]:


x_train1, x_test1, y_train1, y_test1 = train_test_split(x1,y1,test_size=0.2,random_state = 2)


# In[29]:


x_train1


# In[30]:


# 3. Identify columns with missing values
columns_with_missing_values = x_train1.columns[x_train1.isnull().any()]


# In[32]:


# 4. Fill missing values with random values from the same column
for column in columns_with_missing_values:
    # Generate random values from the same column
    random_values = x_train1[column].dropna().sample(x_train1[column].isnull().sum(), replace=True)
    
    # Replace missing values with random values
    x_train1.loc[x_train1[column].isnull(), column] = random_values.values


# In[37]:


x_train1.isnull().sum()


# In[40]:


sns.distplot(x_train["Age"],label = 'Original', hist = False)
sns.distplot(x_train1["Age1"],label = 'imputed', hist = False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




