#!/usr/bin/env python
# coding: utf-8

# ### Feature Transformation

# #### 1. Mathematical Transformation

# In[ ]:


ML labrary sklearn has three types of Mathematical Transformation.
1. Function Transformation.
2. Power Transformation.
3. Quantile Transformation.


# In[ ]:


1. Function Transformation.
A. Log Tranform.
B. Reciprocal Transform.
C. Power Transform (Squre, Sq root)

2. Power Transformation.
A. Box-Cox Tranform
B. Yeo Johnson Transform


# In[ ]:


We have the data and we apply some transformtation on the data.
we will check how the model improvemnt there.

Result is that the abnormal distribtion of the data changed into Normal Distribution.


# #### How to check the Data is Normally Distributed

# In[ ]:


1. sns.distplot in Seaborn Libraries.
2. df.skew()
3. QQ plot


# ### Log Transform

# In[ ]:


Log transform used on Right Skwed Data


# In[ ]:


when you have the data and it is Right Skwd distributed .
apply log tranform and now it is equely distributed on both sides equal scale.
all the data in properly distributed.

linear models like linear Regressiona and linear Logistic Regression perform well 
on linearly distributed data.


# In[ ]:


B. Reciprocal Transform. 1/x
C. Power Transform 
   (Squre x2, left skwed data
    , Sq root , root of x


# In[18]:


df = pd.read_csv("titanic_train.csv", usecols = ["Age","Fare", "Survived"])


# In[36]:


import numpy as np  # for numerical operations
import pandas as pd  # for data manipulation

import matplotlib.pyplot as plt  # for plotting
import seaborn as sns  # for enhanced data visualization

from sklearn.model_selection import train_test_split  # for splitting data into training and testing sets
from sklearn.metrics import accuracy_score  # for measuring classification accuracy

from sklearn.model_selection import cross_val_score  # for cross-validation
from sklearn.linear_model import LogisticRegression  # for logistic regression

from sklearn.tree import DecisionTreeClassifier  # for decision tree classification
from sklearn.preprocessing import FunctionTransformer  # for custom data transformation
from sklearn.compose import ColumnTransformer  # for feature-specific data transformations


# In[6]:


df.sample()


# In[19]:


df.isnull().sum()


# In[20]:


df["Age"].fillna(df["Age"].mean(), inplace= True)


# In[21]:


df.isnull().sum()


# In[22]:


x= df.iloc[:, 1:3]  # [rows, columns]
y = df.iloc[:,0]


# In[24]:


x_train,x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)


# In[25]:


# Check the Data is Normally Distributed or Not


# In[27]:


pip install scipy


# In[28]:


import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats  # Import the stats module

plt.figure(figsize=(14, 4))
plt.subplot(121)
sns.distplot(x_train["Age"])
plt.title("Age PDF")

plt.subplot(122)
stats.probplot(x_train["Age"], dist="norm", plot=plt)
plt.title("Age QQ Plot")

plt.show()


# In[29]:


plt.figure(figsize=(14, 4))
plt.subplot(121)
sns.distplot(x_train["Fare"])
plt.title("Fare PDF")

plt.subplot(122)
stats.probplot(x_train["Fare"], dist="norm", plot=plt)
plt.title("Fare QQ Plot")

plt.show()


# In[31]:


clf = LogisticRegression()
clf2 = DecisionTreeClassifier()


# In[32]:


clf.fit(x_train, y_train)
clf2.fit(x_train, y_train)

y_pred = clf.predict(x_test)
y_pred1 = clf2.predict(x_test)

print("Accuracy LR", accuracy_score(y_test, y_pred))
print("Accuracy DT", accuracy_score(y_test, y_pred1))


# In[ ]:


# There is no impact of distribution of Data on Decision Tree Classifier
# Linear Regression and Logistic Regression follow the distribution of the Data.


# In[33]:


# Apply the Log Transform on both


# In[46]:


from sklearn.preprocessing import FunctionTransformer
import numpy as np

trf = FunctionTransformer(func=np.log1p)


# In[47]:


# what is np.log and np.log1p


# In[48]:


x_train_tranformed = trf.fit_transform(x_train)
x_test_tranformed = trf.transform(x_test)


# In[51]:


# After transformation there is less chnages in DT# There is impovement# There is impovement

clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

clf.fit(x_train_tranformed, y_train)
clf2.fit(x_train_tranformed, y_train)

y_pred = clf.predict(x_test_tranformed)
y_pred1 = clf2.predict(x_test_tranformed)

print("Accuracy LR", accuracy_score(y_test, y_pred))
print("Accuracy DT", accuracy_score(y_test, y_pred1))


# In[ ]:





# In[55]:


# There is improvement

x_transformed = trf.fit_transform(x)

clf = LogisticRegression()
clgf2 = DecisionTreeClassifier()

print("LR", np.mean(cross_val_score(clf, x_transformed, y, scoring = 'accuracy', cv=10)))
print("DT", np.mean(cross_val_score(clf2, x_transformed, y, scoring = 'accuracy', cv=10)))


# In[58]:


plt.figure(figsize=(14, 4))
plt.subplot(121)
stats.probplot(x_train["Fare"], dist="norm", plot=plt)
plt.title("Fare before Log")

plt.subplot(122)
stats.probplot(x_train_tranformed["Fare"], dist="norm", plot=plt)
plt.title("Fare After Log")

plt.show()


# In[59]:


# The result is bad after performing t/s

plt.figure(figsize=(14, 4))
plt.subplot(121)
stats.probplot(x_train["Age"], dist="norm", plot=plt)
plt.title("Age before Log")

plt.subplot(122)
stats.probplot(x_train_tranformed["Age"], dist="norm", plot=plt)
plt.title("Age After Log")

plt.show()


# In[61]:


# Aplly the t/s only on 'Fare' rather than both


# In[69]:


trf2= ColumnTransformer([('log',FunctionTransformer(np.log1p),["Fare"])],remainder = "passthrough")

x_train_transformed2 = trf2.fit_transform(x_train)
x_test_transformed2 = trf2.fit_transform(x_test)


# In[71]:


clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

clf.fit(x_train_transformed2, y_train)
clf2.fit(x_train_transformed2, y_train)

y_pred = clf.predict(x_test_transformed2)
y_pred2 = clf2.predict(x_test_transformed2)

print("Accuracy LR", accuracy_score(y_test, y_pred))
print("Accuracy DT", accuracy_score(y_test, y_pred2))


# In[72]:


x_transformed2 = trf2.fit_transform(x)

clf = LogisticRegression()
clgf2 = DecisionTreeClassifier()

print("LR", np.mean(cross_val_score(clf, x_transformed2, y, scoring = 'accuracy', cv=10)))
print("DT", np.mean(cross_val_score(clf2, x_transformed2, y, scoring = 'accuracy', cv=10)))


# In[ ]:





# In[96]:


def apply_transform(transform):
    x= df.iloc[:,1:3]
    y = df.iloc[:,0]
    
    trf = ColumnTransformer([('log',FunctionTransformer(transform), ["Fare"])], remainder = 'passthrough')
    
    x_train = trf.fit_transform(x)
    clf = LogisticRegression()
    
    print("Accuracy", np.mean(cross_val_score(clf, x_train, y, scoring = 'accuracy', cv= 10)))
    
    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    stats.probplot(x["Fare"], dist="norm", plot=plt)
    plt.title("Fare Before Transform")

    plt.subplot(122)
    stats.probplot(x_train[:,0], dist ="norm", plot=plt)
    plt.title("Fare After Transform")

    plt.show()


# In[97]:


apply_transform(lambda x:x)


# In[98]:


apply_transform(lambda x:x**2)


# In[99]:


apply_transform(lambda x:x**1/2)  # SQRT


# In[102]:


apply_transform(lambda x:1/(x+0.000001)) 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




