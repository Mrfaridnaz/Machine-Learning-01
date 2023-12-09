#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np
import pandas as pd


# In[61]:


df=pd.read_csv("PlacementCGPA.csv")


# In[62]:


df


# In[63]:


df.shape


# In[64]:


df=df.iloc[:100,1:]


# In[65]:


df


# In[ ]:


Steps

1. Preprocess + EDA + Feature Selection.
2. Extract input and output columns.
3. Scale the values
4. Train Test Split or Cross Validation. 
5. Train The Model.
6. Evaluate the Model/Model Selection.
7. Deploy the Model.


# In[66]:


df.info()


# In[67]:


df.describe().T


# In[68]:


import matplotlib.pyplot as plt


# In[69]:


plt.scatter(df['CGPA'],df['IQ'],c=df['Placement'])


# In[73]:


# 2. Extract input and output columns.
# x is Input data or Independent Columns
# y is Output and Dependent Columns

x= df.iloc[:,:2]
y=df.iloc[:,2:]


# In[74]:


x


# In[75]:


y


# In[ ]:


# 4. Train Test Split or Cross Validation. 


# In[76]:


from sklearn.model_selection import train_test_split


# In[78]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)


# In[79]:


x_train


# In[80]:


x_test


# In[81]:


y_train


# In[82]:


y_test


# In[ ]:


# 3. Scale the values


# In[84]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[86]:


x_train = scaler.fit_transform(x_train)


# In[87]:


x_train


# In[88]:


x_test = scaler.transform(x_test)


# In[89]:


x_test


# In[90]:


# Model Train


# In[93]:


from sklearn.linear_model import LogisticRegression


# In[94]:


clf=LogisticRegression()


# In[96]:


clf.fit(x_train, y_train)


# In[97]:


# 6. Evaluate the Model/Model Selection.


# In[ ]:


# Find the accucary on the test data


# In[103]:


y_predic = clf.predict(x_test)


# In[100]:


y_test   # y_predic and y_test shape is the same for both


# In[104]:


from sklearn.metrics import accuracy_score


# In[105]:


accuracy_score(y_test,y_predic )  # (actual o/p, predicted o/p)


# In[ ]:


1.0 mean accuracy is 100%


# In[ ]:


# How to plot the decision boundry
# You can visulize wich pattern used by Ml to predict your model


# In[111]:


pip install mlxtend


# In[112]:


from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions


# In[114]:


y_train = y_train.values.ravel()
plot_decision_regions(x_train, y_train, clf=clf, legend=2)

# y_train.values changing y_values in array in 1-D


# In[ ]:


what is pickel


# In[ ]:




