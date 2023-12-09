#!/usr/bin/env python
# coding: utf-8

# In[ ]:


1. Handling Missing Numerical Data
   1. Univariant imputation
      if you fill any missing value in a col by using the same cols
        mean , meadian, any other value
   2. Multivariant imputation
      if you fill any missing value in a col by using the other cols
      KNN , Iterater imputer


# In[ ]:


1. Mean and median imputation for good result.
   mean used for normally distributed missing data
    median used for skwed missing data


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

 #usecols = ['Age','Fare','Survived'])


# In[13]:


df = pd.read_csv('titanic_train.csv')


# In[14]:


df['family'] = df['SibSp']+df['Parch']


# In[15]:


df = df.drop(columns=['PassengerId','Pclass','Name','SibSp','Ticket','Cabin'])


# In[16]:


df = df.drop(columns=['Embarked','Parch'])


# In[17]:


df = df.drop(columns=['Sex'])


# In[18]:


df.head()


# In[19]:


df.isnull().mean()


# In[22]:


# Input data
x=df.drop(columns = ['Survived'])


# In[23]:


# output data
y = df['Survived']


# In[25]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 2)


# In[26]:


x_train.shape, x_test.shape


# In[28]:


x_train.isnull().mean()*100

# "Age" is Numerical col having 20% missning values.


# In[32]:


mean_age = x_train['Age'].mean() # 29.78590
median_age = x_train['Age'].median() # 28.75


# In[35]:


# asign new columns, age col is filling by mean and median

x_train['Age_mean'] = x_train['Age'].fillna(mean_age)
x_train['Age_median'] = x_train['Age'].fillna(median_age)


# In[36]:


x_train.sample(5)


# In[37]:


# The chnage in variance after imputation 

print('Orignal Age variable variance',x_train['Age'].var())
print('Age variance after mean imputation', x_train['Age_mean'].var())
print('Age variance after median imputation', x_train['Age_median'].var())


# In[37]:


# Change in the distribution

fig = plt.figure()
ax=fig.add_subplot(111)

# Orignal Variable distribution
x_train['Age'].plot(kind='kde', ax=ax)

# Variable imputed with the Mean
x_train['Age_mean'].plot(kind='kde', ax=ax, color = 'red')

# Variable imputed with the Median
x_train['Age_median'].plot(kind='kde', ax=ax, color = 'green')


# In[38]:


x_train.cov()


# In[39]:


x_train.corr()


# In[41]:


x_train[['Age','Age_mean','Age_median']].boxplot()


# ## Using Sklearn

# In[43]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 2)


# In[46]:


imputer1 = SimpleImputer(strategy = 'mean') # Mean apply on "Age"
imputer2 = SimpleImputer(strategy = 'median') # Meadian apply on "Median"


# In[ ]:


# trf = ColumnTransformer([(),()] ,remainder = 'passthrough')
trf is a variable that is being assigned the result of creating a ColumnTransformer 
ColumnTransformer([(), ()], remainder='passthrough'): This is the constructor call 
for the ColumnTransformer object. It takes two main arguments:
    


# In[47]:


# trf = ColumnTransformer([(),()] ,remainder = 'passthrough')

trf = ColumnTransformer([
    ('imputer1',imputer1,['Age']),    # Apply imputer1 to the 'Age' column
    ('imputer2',imputer2,['Fare'])    # Apply imputer2 to the 'Fare' column
],remainder = 'passthrough')


# In[48]:


trf.fit(x_train)


# In[49]:


# Mean value for Age
trf.named_transformers_['imputer1'].statistics_


# In[51]:


# Median valuie for Fare
trf.named_transformers_['imputer2'].statistics_


# In[52]:


x_train = trf.transform(x_train)
x_test = trf.transform(x_test)


# In[54]:


x_train


# In[ ]:


Note Points
1. Fit always on training data
2. find mean and median on Training data
3. Transform on x_test use mean and median of Traing data
4. If the data is missing randomly than use mean and median.


# ### Arbitrary value Imputation

# In[ ]:


here, you can fill all missing values by a same sinlge value
mostly used for filling missed catogorical value
 


# In[ ]:


Advantages:
    1. Easy to apply
Disadvantages:
    2. PDF distortion.
    3. variance.
    4. covaraince changes
    
If the data is not missing randomly.


# In[51]:


df


# In[52]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 2)


# In[54]:


x_train['Age_99'] = x_train["Age"].fillna(99)
x_train['Age_minus1'] = x_train["Age"].fillna(-1)

x_train['Fare_999'] = x_train["Age"].fillna(999)
x_train['Fare_minus1'] = x_train["Age"].fillna(-1)


# In[55]:


# After imputation check the chnage in variance

print('Orignal Age variable variance',x_train['Age'].var())
print('Age variance after mean imputation', x_train['Age_99'].var())
print('Age variance after median imputation', x_train['Age_minus1'].var())

print('Orignal Age variable variance',x_train['Age'].var())
print('Age variance after mean imputation', x_train['Fare_999'].var())
print('Age variance after median imputation', x_train['Fare_minus1'].var())


# In[56]:


# Chang in the distribution

fig = plt.figure()
ax=fig.add_subplot(111)

# Orignal Variable distribution
x_train['Age'].plot(kind='kde', ax=ax)

# Variable imputed with the Mean
x_train['Age_99'].plot(kind='kde', ax=ax, color = 'red')

# Variable imputed with the Median
x_train['Age_minus1'].plot(kind='kde', ax=ax, color = 'green')


# In[ ]:





# In[57]:


# Chang in the distribution

fig = plt.figure()
ax=fig.add_subplot(111)

# Orignal Variable distribution
x_train['Age'].plot(kind='kde', ax=ax)

# Variable imputed with the Mean
x_train['Fare_999'].plot(kind='kde', ax=ax, color = 'red')

# Variable imputed with the Median
x_train['Fare_minus1'].plot(kind='kde', ax=ax, color = 'green')


# In[ ]:


chekc for cov and corr


# ### Using Sklearn

# In[58]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 2)


# In[59]:


imputer1 = SimpleImputer(strategy = 'constant',fill_value=99)
imputer2 = SimpleImputer(strategy = 'constant',fill_value=999)


# In[60]:


trf = ColumnTransformer([
    ('imputer1',imputer1,['Age']),
    ('imputer2',imputer2,['Fare'])
],remainder = 'passthrough')


# In[61]:


trf.fit(x_train)


# In[62]:


trf.named_transformers_['imputer1'].statistics_


# In[63]:


trf.named_transformers_['imputer2'].statistics_


# In[64]:


x_train = trf.transform(x_train)
x_test = trf.transform(x_test)


# In[65]:


x_train


# ### End of Distribution imputation

# In[ ]:





# ### Random Sample Imputation

# In[ ]:




