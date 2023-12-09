#!/usr/bin/env python
# coding: utf-8

# In[201]:


import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression


# In[248]:


df = np.round(pd.read_csv('50_Startups.csv')[['R&D Spend','Administration','Marketing Spend','Profit']]/10000)
np.random.seed(9)
df = df.sample(5)
df


# In[224]:


df = df.iloc[:,0:-1]
df


# In[225]:


df.iloc[1,0] = np.NaN
df.iloc[3,1] = np.NaN
df.iloc[-1,-1] = np.NaN


# In[226]:


df.head()


# In[227]:


# Step 1 - Impute all missing values with mean of respective col

df0 = pd.DataFrame()

df0['R&D Spend'] = df['R&D Spend'].fillna(df['R&D Spend'].mean())
df0['Administration'] = df['Administration'].fillna(df['Administration'].mean())
df0['Marketing Spend'] = df['Marketing Spend'].fillna(df['Marketing Spend'].mean())


# In[228]:


# 0th Iteration
df0


# In[229]:


# Remove the col1 imputed value
df1 = df0.copy()

df1.iloc[1,0] = np.NaN

df1


# In[230]:


# Use first 3 rows to build a model and use the last for prediction

X = df1.iloc[[0,2,3,4],1:3]
X


# In[231]:


y = df1.iloc[[0,2,3,4],0]
y


# In[232]:


lr = LinearRegression()
lr.fit(X,y)
lr.predict(df1.iloc[1,1:].values.reshape(1,2))


# In[233]:


df1.iloc[1,0] = 23.14


# In[234]:


df1


# In[235]:


# Remove the col2 imputed value

df1.iloc[3,1] = np.NaN

df1


# In[236]:


# Use last 3 rows to build a model and use the first for prediction
X = df1.iloc[[0,1,2,4],[0,2]]
X


# In[237]:


y = df1.iloc[[0,1,2,4],1]
y


# In[238]:


lr = LinearRegression()
lr.fit(X,y)
lr.predict(df1.iloc[3,[0,2]].values.reshape(1,2))


# In[239]:


df1.iloc[3,1] = 11.06


# In[240]:


df1


# In[241]:


# Remove the col3 imputed value
df1.iloc[4,-1] = np.NaN

df1


# In[242]:


# Use last 3 rows to build a model and use the first for prediction
X = df1.iloc[0:4,0:2]
X


# In[243]:


y = df1.iloc[0:4,-1]
y


# In[244]:


lr = LinearRegression()
lr.fit(X,y)
lr.predict(df1.iloc[4,0:2].values.reshape(1,2))


# In[245]:


df1.iloc[4,-1] = 31.56


# In[246]:


# After 1st Iteration
df1


# In[247]:


# Subtract 0th iteration from 1st iteration

df1 - df0


# In[249]:


df2 = df1.copy()

df2.iloc[1,0] = np.NaN

df2


# In[250]:


X = df2.iloc[[0,2,3,4],1:3]
y = df2.iloc[[0,2,3,4],0]

lr = LinearRegression()
lr.fit(X,y)
lr.predict(df2.iloc[1,1:].values.reshape(1,2))


# In[252]:


df2.iloc[1,0] = 23.78


# In[253]:


df2.iloc[3,1] = np.NaN
X = df2.iloc[[0,1,2,4],[0,2]]
y = df2.iloc[[0,1,2,4],1]

lr = LinearRegression()
lr.fit(X,y)
lr.predict(df2.iloc[3,[0,2]].values.reshape(1,2))


# In[254]:


df2.iloc[3,1] = 11.22


# In[255]:


df2.iloc[4,-1] = np.NaN

X = df2.iloc[0:4,0:2]
y = df2.iloc[0:4,-1]

lr = LinearRegression()
lr.fit(X,y)
lr.predict(df2.iloc[4,0:2].values.reshape(1,2))


# In[256]:


df2.iloc[4,-1] = 31.56


# In[257]:


df2


# In[258]:


df2 - df1


# In[259]:


df3 = df2.copy()

df3.iloc[1,0] = np.NaN

df3


# In[260]:


X = df3.iloc[[0,2,3,4],1:3]
y = df3.iloc[[0,2,3,4],0]

lr = LinearRegression()
lr.fit(X,y)
lr.predict(df3.iloc[1,1:].values.reshape(1,2))


# In[261]:


df3.iloc[1,0] = 24.57


# In[262]:


df3.iloc[3,1] = np.NaN
X = df3.iloc[[0,1,2,4],[0,2]]
y = df3.iloc[[0,1,2,4],1]

lr = LinearRegression()
lr.fit(X,y)
lr.predict(df3.iloc[3,[0,2]].values.reshape(1,2))


# In[265]:


df3.iloc[3,1] = 11.37


# In[266]:


df3.iloc[4,-1] = np.NaN

X = df3.iloc[0:4,0:2]
y = df3.iloc[0:4,-1]

lr = LinearRegression()
lr.fit(X,y)
lr.predict(df3.iloc[4,0:2].values.reshape(1,2))


# In[272]:


df3.iloc[4,-1] = 45.53


# In[270]:


df2.iloc[3,1] = 11.22


# In[273]:


df3


# In[275]:


df3 - df2


# In[ ]:




