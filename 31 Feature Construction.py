#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

import seaborn as sns


# In[11]:


df = pd.read_csv('titanic_train.csv')[['Age','Pclass','SibSp','Parch','Survived']]


# In[12]:


df.head()


# In[5]:


df.dropna(inplace=True)


# In[13]:


df.head()


# In[14]:


X = df.iloc[:,0:4]
y = df.iloc[:,-1]


# In[15]:


X.head()


# In[9]:


np.mean(cross_val_score(LogisticRegression(),X,y,scoring='accuracy',cv=20))

# cross_val_score(...): This function is part of the scikit-learn library, and it is used for 
# performing cross-validation on a machine learning model. Cross-validation is a technique used to 
# assess how well a model generalizes to new data. It involves splitting the dataset into multiple
# subsets (folds), training the model on some of the folds and testing it on others, and then 
# repeating this process to ensure robust performance evaluation.


# In[ ]:


# LogisticRegression(): This is the machine learning model that will be evaluated using 
# cross-validation. In this case, it's a logistic regression classifier.


# ## Applying Feature Construction

# In[16]:


X['Family_size'] = X['SibSp'] + X['Parch'] + 1


# In[17]:


X.head()


# In[61]:


def myfunc(num):
    if num == 1:
        #alone
        return 0
    elif num >1 and num <=4:
        # small family
        return 1
    else:
        # large family
        return 2


# In[62]:


myfunc(4)


# In[95]:


X['Family_type'] = X['Family_size'].apply(myfunc)


# In[96]:


X.head()


# In[97]:


X.drop(columns=['SibSp','Parch','Family_size'],inplace=True)


# In[98]:


X.head()


# In[99]:


np.mean(cross_val_score(LogisticRegression(),X,y,scoring='accuracy',cv=20))


# ## Feature Splitting

# In[21]:


df = pd.read_csv('titanic_train.csv')


# In[22]:


df.head()


# In[23]:


df['Name']


# In[28]:


df['Name'].str.split(', ', expand=True)


# In[29]:


df['Name'].str.split(', ', expand=True)[1]


# In[30]:


df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)


# In[31]:


df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]


# In[32]:


df['Title'] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]


# In[33]:


df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]


# In[109]:


df[['Title','Name']]


# In[110]:


(df.groupby('Title').mean()['Survived']).sort_values(ascending=False)


# In[111]:


df['Is_Married'] = 0
df['Is_Married'].loc[df['Title'] == 'Mrs'] = 1


# In[112]:


df['Is_Married']


# In[ ]:




