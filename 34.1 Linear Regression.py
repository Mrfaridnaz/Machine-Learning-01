#!/usr/bin/env python
# coding: utf-8

# In[43]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[44]:


df = pd.read_csv('placement.csv')


# In[45]:


df.head()


# In[11]:


plt.scatter(df['cgpa'],df['package'])
plt.xlabel('cgpa')
plt.ylabel('Package(in lpa)')


# In[12]:


X = df.iloc[:,0:1]
y = df.iloc[:,-1]


# In[13]:


y


# In[14]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[15]:


from sklearn.linear_model import LinearRegression


# In[16]:


lr = LinearRegression()


# In[17]:


lr.fit(X_train,y_train)


# In[18]:


X_test


# In[19]:


y_test


# In[20]:


lr.predict(X_test.iloc[0].values.reshape(1,1))


# In[21]:


plt.scatter(df['cgpa'],df['package'])
plt.plot(X_train,lr.predict(X_train),color='red')
plt.xlabel('CGPA')
plt.ylabel('Package(in lpa)')


# In[22]:


m = lr.coef_


# In[23]:


b = lr.intercept_


# In[24]:


# y = mx + b

m * 8.58 + b


# In[25]:


m * 9.5 + b


# In[103]:


m * 100 + b


# In[ ]:





# In[ ]:





# ## Crteate a class for Simple Linear Regression

# In[47]:


class MeraLR:
    
    def __init__(self): # Constructor
        self.m = None
        self.b = None
        
    def fit(self, x_train, y_train):    # fit and predict two method , fit is to train model with some code 
        
        num = 0
        den = 0
        for i in range(x_train.shape[0]): # The loop will run 160 times
            num = num + (x_train[i]-x_train.mean())*(y_train[i]-y_train.mean())
            den = den +  (x_train[i]-x_train.mean())*(x_train[i]-x_train.mean())
            
        self.m = num/den
        self.b = y_train.mean() -(self.m*x_train.mean())
        print(self.m)
        print(self.b)
    
    def predict(self, x_test):
        
        return self.m*x_test + self.b
    
    # main purpose to train the model mean calculate the value 'm' and 'b'
    
    


# In[48]:


import numpy as np
import pandas as pd


# In[49]:


df


# In[50]:


x= df.iloc[:,0].values # values for numpy array
y = df.iloc[:,1].values


# In[51]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 2)


# In[52]:


x_train.shape


# In[58]:


# craete an Object lr

lr = MeraLR()


# In[59]:


lr.fit(x_train, y_train)

# 'm' value and 'b' value


# In[60]:


print(lr.predict(x_test[0]))


# In[2]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[ ]:


placement (1).csv


# In[5]:


df = pd.read_csv('placement.csv')


# In[6]:


df


# In[7]:


df.head()
df.shape


# In[8]:


plt.scatter(df['cgpa'],df['package'])
plt.xlabel('CGPA')
plt.ylabel('Package(in lpa)')


# In[9]:


X = df.iloc[:,0:1]
y = df.iloc[:,-1]


# In[10]:


y


# In[11]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[12]:


from sklearn.linear_model import LinearRegression


# In[13]:


lr = LinearRegression()


# In[14]:


lr.fit(X_train,y_train)


# In[15]:


plt.scatter(df['cgpa'],df['package'])
plt.plot(X_train,lr.predict(X_train),color='red')
plt.xlabel('CGPA')
plt.ylabel('Package(in lpa)')


# In[16]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# In[17]:


y_pred = lr.predict(X_test)


# In[18]:


y_test.values


# In[19]:


print("MAE",mean_absolute_error(y_test,y_pred))


# In[20]:


print("MSE",mean_squared_error(y_test,y_pred))


# In[21]:


print("RMSE",np.sqrt(mean_squared_error(y_test,y_pred)))


# In[22]:


print("MSE",r2_score(y_test,y_pred))
r2 = r2_score(y_test,y_pred)


# In[23]:


# Adjusted R2 score
X_test.shape


# In[24]:


1 - ((1-r2)*(40-1)/(40-1-1))


# In[25]:


new_df1 = df.copy()
new_df1['random_feature'] = np.random.random(200)

new_df1 = new_df1[['cgpa','random_feature','package']]
new_df1.head()


# In[26]:


plt.scatter(new_df1['random_feature'],new_df1['package'])
plt.xlabel('random_feature')
plt.ylabel('Package(in lpa)')


# In[27]:


X = new_df1.iloc[:,0:2]
y = new_df1.iloc[:,-1]


# In[28]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[29]:


lr = LinearRegression()


# In[30]:


lr.fit(X_train,y_train)


# In[31]:


y_pred = lr.predict(X_test)


# In[32]:


print("R2 score",r2_score(y_test,y_pred))
r2 = r2_score(y_test,y_pred)


# In[33]:


1 - ((1-r2)*(40-1)/(40-1-2))


# In[34]:


new_df2 = df.copy()

new_df2['iq'] = new_df2['package'] + (np.random.randint(-12,12,200)/10)

new_df2 = new_df2[['cgpa','iq','package']]


# In[35]:


new_df2.sample(5)


# In[36]:


plt.scatter(new_df2['iq'],new_df2['package'])
plt.xlabel('iq')
plt.ylabel('Package(in lpa)')


# In[37]:


np.random.randint(-100,100)


# In[38]:


X = new_df2.iloc[:,0:2]
y = new_df2.iloc[:,-1]


# In[39]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[40]:


lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)


# In[41]:


print("R2 score",r2_score(y_test,y_pred))
r2 = r2_score(y_test,y_pred)


# In[42]:


1 - ((1-r2)*(40-1)/(40-1-2))


# In[ ]:





# In[ ]:




