#!/usr/bin/env python
# coding: utf-8

# ### Scikit Learn Pipeline

# In[ ]:


Pipelines makes it easy to apply the same preprocessing to train and test


# In[ ]:


Pipelines chains together multiple steps so that the output of each step is used
as input to the next step


# In[ ]:





# ### 1 Without using pipeline

# In[28]:


import numpy as np
import pandas as pd


# In[29]:


from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier # for prediction


# In[ ]:


We have to create a model that predict that the person will survived or not by providing the 
inputs


# In[30]:


df= pd.read_csv("titanic_train.csv")


# In[31]:


df.sample(5)


# In[9]:


# Drop the Columns

df.drop(columns = ['PassengerId','Name','Ticket','Cabin'],inplace= True)


# In[10]:


df.sample(3)


# ### 1. Step

# In[11]:


x_train,x_test, y_train,y_test = train_test_split(df.drop(columns = ['Survived']), 
                                                  df['Survived'], test_size = 0.2, 
                                                  random_state = 42)


# In[12]:


x_train


# In[13]:


y_train  # Survived


# In[14]:


x_train.shape


# In[15]:


x_test.shape


# In[16]:


y_train.shape


# In[319]:


y_test.shape


# In[18]:


df.isnull().sum()


# In[19]:


x_train['Age'] # Missing values


# In[21]:


# Applying imputation for mising values |'Age' and 'Embarked '

si_age = SimpleImputer() # filling the values by mean default
si_embarked = SimpleImputer(strategy = 'most_frequent')

x_train_age = si_age.fit_transform(x_train[['Age']])
x_train_embarked = si_embarked.fit_transform(x_train[['Embarked']])

x_test_age = si_age.transform(x_test[['Age']])
x_test_Embarked = si_embarked.transform(x_test[['Embarked']])


# In[22]:


x_train_age # Age col with filled values


# In[23]:


x_train_embarked # with filling values


# In[ ]:


In above 2 steps we have filled missing values in 'age' and 'embarked'
and asingned two cols in "x_train_age" and "x_train_embarked"

"x_train_age"
"x_train_embarked"


# In[24]:


x_test_Embarked.shape


# In[ ]:


One Hot Encoding | sex and Embarked 9:00
    changing the categorical cols to numerical cols


# In[25]:


# We can not pass 'Sex' and 'Embarked' together becs 'Embarked' has missing values


from sklearn.preprocessing import OneHotEncoder

ohe_sex = OneHotEncoder(sparse=False, handle_unknown='ignore')
ohe_embarked = OneHotEncoder(sparse=False, handle_unknown='ignore')

# Fit and transform the training data
x_train_sex = ohe_sex.fit_transform(x_train[['Sex']])
x_train_embarked = ohe_embarked.fit_transform(x_train_embarked)

# Transform the testing data using the same encoders
x_test_sex = ohe_sex.transform(x_test[['Sex']])
x_test_embarked = ohe_embarked.transform(x_test_Embarked)



# In[26]:


x_train_embarked


# In[27]:


x_test_sex


# In[ ]:


# There are two columns for 'male' and 'female'
# we didnt remove the 'fisrt' column here to remove the multicolinearity
# becs we are using the decision tree not linear model


# In[333]:


x_train_sex.shape


# In[334]:


x_train_embarked.shape


# In[335]:


x_test_sex.shape


# In[336]:


x_test_Embarked.shape


# In[ ]:


Now we have three array for "Age", "Sex" and "Embarked"


# In[337]:


x_train_rem = x_train.drop(columns = ['Sex','Age','Embarked'])

# droping from x_train


# In[345]:


x_train_rem


# In[338]:


x_test_rem = x_test.drop(columns = ['Sex','Age','Embarked'])


# In[339]:


x_train_transform = np.concatenate((x_train_rem,x_train_age, x_train_sex,x_train_embarked),axis=1)
x_test_transform = np.concatenate((x_test_rem,x_test_age, x_test_sex,x_test_embarked),axis=1)


# In[348]:


x_train_transform


# In[349]:


x_test_transform


# In[350]:


x_test_transform.shape


# In[346]:


x_train_transform.shape


# In[347]:


clf = DecisionTreeClassifier()
clf.fit(x_train_transform, y_train)


# In[352]:


y_pred = clf.predict(x_test_transform)


# In[353]:


y_pred.shape


# In[354]:


y_train.shape


# In[355]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[ ]:


The model is created and it will predict that the passaender will die or survived by given data


# In[362]:


import pickle
pickle.dump(ohe_sex, open('models/ohe_sex.pkl','wb'))
pickle.dump(ohe_embarked,open('models/ohe_embarked.pkl','wb'))
pickle.dumb(clf,open('models/clf.pkl','wb'))


# ## 2 Using Pipeline

# In[ ]:


Steps
1. we have two cols with missing data 
  'age' | 'Embarked'
    
    impute the missing values by using 1 col transformer and its o/p use as input for next step
    for 2nd col transformer its work to OneHotEncoding the 'Sex' and 'Embarked'
    and its o/p used for "scaling" the 3rd col transformation and than "feature selection"
    select best 5 cols and then train the model using DecisionTree.


# In[32]:


# Drop the Columns

df.drop(columns = ['PassengerId','Name','Ticket','Cabin'],inplace= True)


# In[34]:


df.sample()


# In[33]:


x_train,x_test, y_train,y_test = train_test_split(df.drop(columns = ['Survived']), 
                                                  df['Survived'], test_size = 0.2, 
                                                  random_state = 42)


# In[35]:


x_train


# #### 1. Step

# In[ ]:


Fill the missing values 1st cols transfomer |  'age' | 'Embarked'

# The next step will follow the previuos dataframe


# In[36]:


# Imputation Transformer

trf1 = ColumnTransformer([
    ('impute_age',SimpleImputer(), [2]), 
    ('impute_embarked', SimpleImputer(strategy = 'most_frequent'),[6])
], remainder = 'passthrough')

# 2 and 6 are the index for 'Age' becs it is array form not dataFrame , 
# In pipeline call it by index number rather than col name.


# In[37]:


# One Hot Encoding | 'Sex' and 'Embarked'

trf2 = ColumnTransformer([
    ('ohe_sex_embarked', OneHotEncoder(sparse = False, handle_unknown = 'ignore'),[1,6])
], remainder = 'passthrough')

# using decision tree here, so dont use 'first drop'


# In[45]:


# feature Scaling for all columns 0 to 10 , becs we have 10 cols after OneHotEncoding

trf3 = ColumnTransformer([
    ('scale', MinMaxScaler(), slice(0,10))
])


# In[49]:


# Feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

trf4 = SelectKBest(score_func = chi2, k=8)

# k=8 mean using top 8 cols


# In[50]:


# Train the Model

from sklearn.tree import DecisionTreeClassifier

trf5 = DecisionTreeClassifier()


# ### Create Pipeline

# In[52]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# In[53]:


pipe = Pipeline([
    ('trf1',trf1),
    ('trf2',trf2),
    ('trf3',trf3),
    ('trf4',trf4),
    ('trf5',trf5),
])


# In[ ]:


Done


# In[ ]:


30:00 min


# ### Pipeline VS make_pipiline

# In[ ]:




