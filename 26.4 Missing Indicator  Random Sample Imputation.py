#!/usr/bin/env python
# coding: utf-8

# ## Random Imputation

# In[ ]:


1. Preserve the variance of the variable(but why?)
2. Memory heavy for deployment, as we need to store the orignal training set to extract values
   from and replace the NA in coming observations.
3. Well suited for linear models as it does not distort the distribution, regardless of the % of NA    


# In[ ]:


# Filling the missing values by Random numbers
  it can be applied for both categorical as well as Numerical
    this feature is only in pandas
  : Data distribution remains same after imputation  


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


# In[6]:


y = df['Survived']


# In[7]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 2)


# In[8]:


x_train.shape, x_test.shape


# In[9]:


x_train.isnull().mean()

# 20% missning values in age


# In[10]:


# Create a new col named 'Age_imputed' for Train and test

x_train['Age_imputed'] = x_train['Age']
x_test['Age_imputed'] = x_test['Age']


# In[11]:


x_train


# In[12]:


x_train['Age_imputed']


# In[13]:


x_train


# In[14]:


x_train['Age_imputed']


# In[15]:


# print all the null values 148
x_train['Age_imputed'][ x_train['Age_imputed'].isnull()]


# In[19]:


# delete the rows that have null values
x_train['Age'].dropna()


# In[51]:


712-148


# In[20]:


x_train['Age_imputed'].isnull().sum() # count the null values/Deleted


# In[55]:


x_train['Age'].dropna().sample(x_train['Age_imputed'].isnull().sum()).values


# In[22]:


# fill the missing values in age and select randomly any value from the 'Age'
# First part is showing the missing values present in 'Age_imputed' and replaced by R.H.S.

# R.H.S ==>> x_train['Age'].dropna() it will drop all missig values in 'Age'
# RHS will generate randomly 148 values that is equal to missing values

x_train['Age_imputed'][x_train['Age_imputed'].isnull()] = x_train['Age'].dropna().sample(x_train['Age_imputed'].isnull().sum()).values
x_test['Age_imputed'][x_test['Age_imputed'].isnull()] = x_test['Age'].dropna().sample(x_test['Age_imputed'].isnull().sum()).values


# In[24]:


# Age_imputed missing values have been filled
x_train


# In[25]:


sns.distplot(x_train["Age"],label = 'Original', hist = False)
sns.distplot(x_train["Age_imputed"],label = 'imputed', hist = False)


# In[66]:


print('Original variable variance:', x_train["Age"].var())
print('variance after random imputation:', x_train["Age_imputed"].var())


# In[69]:


x_train[['Fare','Age','Age_imputed']].cov()


# In[71]:


x_train[['Age','Age_imputed']].boxplot()


# In[ ]:





# In[78]:


# 3. Identify columns with missing values
columns_with_missing_values = x_train.columns[x_train.isnull().any()]


# In[79]:


x_train.columns[x_train.isnull().any()]


# In[65]:


# 4. Fill missing values with random values from the same column
for column in columns_with_missing_values:
    # Generate random values from the same column
    random_values = x_train[column].dropna().sample(x_train[column].isnull().sum(), replace=True)
    
    # Replace missing values with random values
    x_train.loc[x_train[column].isnull(), column] = random_values.values


# In[66]:


# 3. Identify columns with missing values
columns_with_missing_values = x_test.columns[x_test.isnull().any()]

# 4. Fill missing values with random values from the same column
for column in columns_with_missing_values:
    # Generate random values from the same column
    random_values = x_test[column].dropna().sample(x_test[column].isnull().sum(), replace=True)
    
    # Replace missing values with random values
    x_test.loc[x_test[column].isnull(), column] = random_values.values


# In[67]:


print(x_train)


# In[68]:


x_train.isnull().sum()


# In[69]:


x_test.isnull().sum()


# In[39]:


sns.distplot(x_train["Age"],label = 'Original', hist = False)
sns.distplot(x_train["Age_imputed"],label = 'imputed', hist = False)


# In[ ]:





# In[ ]:





# In[80]:


sample_value = x_train['Age'].dropna().sample(1, random_state=int(Observation['Fare']))


# In[ ]:


17:00


# In[45]:


data = pd.read_csv('house-train.csv', usecols = ['GarageQual','FireplaceQu','SalePrice'])


# In[46]:


data.head()


# In[47]:


data.isnull().mean()*100


# In[48]:


x=data
y = data['SalePrice']


# In[49]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 2)
x_train.shape, x_test.shape


# In[50]:


# Create a new col named 'GarageQual_imputed' for Train and test

x_train['GarageQual_imputed'] = x_train['GarageQual']
x_test['GarageQual_imputed'] = x_test['GarageQual']

x_train['FireplaceQu_imputed'] = x_train['FireplaceQu']
x_test['FireplaceQu_imputed'] = x_test['FireplaceQu']


# In[51]:


x_train.sample(5)


# In[52]:


x_train['GarageQual_imputed'][x_train['GarageQual_imputed'].isnull()] = x_train['GarageQual'].dropna().sample(x_train['GarageQual_imputed'].isnull().sum()).values
x_test['GarageQual_imputed'][x_test['GarageQual_imputed'].isnull()] = x_test['GarageQual'].dropna().sample(x_test['GarageQual_imputed'].isnull().sum()).values


x_train['FireplaceQu_imputed'][x_train['FireplaceQu_imputed'].isnull()] = x_train['FireplaceQu'].dropna().sample(x_train['FireplaceQu_imputed'].isnull().sum()).values
x_test['FireplaceQu_imputed'][x_test['FireplaceQu_imputed'].isnull()] = x_test['FireplaceQu'].dropna().sample(x_test['FireplaceQu_imputed'].isnull().sum()).values


# In[53]:


temp = pd.concat(
    [x_train['GarageQual'].value_counts() / len(x_train['GarageQual'].dropna()),
     x_train['GarageQual_imputed'].value_counts() / len(x_train)
    ],
    axis=1
)

temp.columns = ['Original', 'Imputed']


# In[54]:


temp


# In[ ]:





# In[55]:


temp = pd.concat(
    [x_train['FireplaceQu'].value_counts() / len(x_train['FireplaceQu'].dropna()),
     x_train['FireplaceQu_imputed'].value_counts() / len(data)
    ],
    axis=1
)

temp.columns = ['Original', 'Imputed']

# check data or df


# In[56]:


temp


# In[59]:


for category in x_train['FireplaceQu'].dropna().unique():
    sns.distplot(x_train[x_train['FireplaceQu']==category]['SalePrice'], hist = False, label = category)
plt.show()    


# In[60]:


for category in x_train['FireplaceQu_imputed'].dropna().unique():
    sns.distplot(x_train[x_train['FireplaceQu_imputed']==category]['SalePrice'], hist = False, label = category)
plt.show()    


# In[ ]:





# # Missing Indicator

# In[ ]:


You will create a new col for every missing data in col
new col has only two values True or False


# In[61]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


# In[64]:


df = pd.read_csv('titanic_train.csv', usecols = ['Age','Fare','Survived'])


# In[65]:


df.head()


# In[66]:


x=df.drop(columns = ['Survived'])


# In[67]:


y = df['Survived']


# In[68]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 2)


# In[69]:


x_train.head()


# In[72]:


# fill the missing values by mean in train and test
from sklearn.impute import SimpleImputer


si = SimpleImputer()
x_train_trf = si.fit_transform(x_train)
x_test_trf = si.transform(x_test)


# In[73]:


x_train_trf

# There is no missing values in 'Age'


# In[74]:


# Accuracy without using missing indicator

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(x_train_trf,y_train)

y_pred = clf.predict(x_test_trf)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[87]:


pip install missingno


# In[89]:


from sklearn.impute import MissingIndicator

mi = MissingIndicator()
mi.fit(x_train)


# In[90]:


mi.features_


# In[91]:


x_train_missing = mi.transform(x_train)


# In[92]:


x_train_missing


# In[93]:


x_test_missing = mi.transform(x_test)


# In[94]:


x_test_missing 


# In[96]:


# asign 'x_train_missing' col values in 'Age_NA'
x_train['Age_NA'] = x_train_missing


# In[97]:


# same for the test
x_test['Age_NA'] = x_test_missing


# In[98]:


x_train


# In[100]:


si = SimpleImputer()

x_train_trf2 = si.fit_transform(x_train)
x_test_trf2 = si.transform(x_test)


# In[101]:


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(x_train_trf2,y_train)

y_pred = clf.predict(x_test_trf2)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

# Accuracy is increased by 2%


# In[102]:


# without using the missing indicator class
si = SimpleImputer(add_indicator = True)


# In[105]:


# import again Train test
x_train = si.fit_transform(x_train)


# In[106]:


x_test = si.fit_transform(x_test)


# In[107]:


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(x_train_trf2,y_train)

y_pred = clf.predict(x_test_trf2)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[ ]:


# If you use pandas workflow than use missing indicator
# If you are using sklearn , you can mention the parameter in Simpleimputer Class
  si = SimpleImputer(add_indicator = True)
    you dont need to import the class
    you dont need to make an object
    you dont need to merge the col that you receive from object
    


# In[ ]:





# ### Automatically select value for imputation

# In[108]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression


# In[ ]:


df


# In[ ]:




