#!/usr/bin/env python
# coding: utf-8

# ### k-nearest neighbors algorithm

# In[23]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score


# In[24]:


df = pd.read_csv('titanic_train.csv')[['Age','Pclass','Fare','Survived']]


# In[25]:


df.head()


# In[26]:


df.isnull().mean() * 100


# In[27]:


X = df.drop(columns=['Survived'])
y = df['Survived']


# In[28]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[29]:


X_train.head()


# In[30]:


X_train.isnull().sum()


# In[31]:


knn = KNNImputer(n_neighbors=3,weights='distance') # fill null values

X_train_trf = knn.fit_transform(X_train) # Numpy Array
X_test_trf = knn.transform(X_test) # Numpy Array


# In[19]:


lr = LogisticRegression()

lr.fit(X_train_trf,y_train)

y_pred = lr.predict(X_test_trf)

accuracy_score(y_test,y_pred)


# In[55]:


# Comparision with Simple Imputer --> mean

si = SimpleImputer()

X_train_trf2 = si.fit_transform(X_train)
X_test_trf2 = si.transform(X_test)


# In[56]:


lr = LogisticRegression()

lr.fit(X_train_trf2,y_train)

y_pred2 = lr.predict(X_test_trf2)

accuracy_score(y_test,y_pred2)


# In[ ]:




