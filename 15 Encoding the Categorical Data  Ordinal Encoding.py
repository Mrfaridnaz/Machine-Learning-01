#!/usr/bin/env python
# coding: utf-8

# ### Feature Engineering

# In[ ]:


1. Feature Transformation
   A. Feature Scaling
   B. Encoding Categorical Data


# #### B. Encoding Categorical Data

# In[ ]:


Categorical Data:

Nominal: Nominal data represents categories with no inherent order, such as colors, 
         types of fruits, or country names.
    
Ordinal: Ordinal data represents categories with a meaningful order, like education 
         levels (e.g., high school, college, graduate) or customer satisfaction ratings (e.g., 
         poor, fair, good).


# In[ ]:


Categorical data is in string format and the ML understands the numerical data
so you will have to convert it into numerical.


# In[ ]:


1. Ordinal Encoding.
    used in Ordinal data
2. One Hot Encoding.
    used in Nominal data


# #### Label Encoding

# In[ ]:





# #### 1. Ordinal Encoding.

# In[ ]:


Input x = If there is ordinal categorical data | use Ordinal Encoder
Output y = if the o/p is categorical | use label Encoding | dont use Ordinal Encoding


# In[ ]:


Education
HS   0
UG   1
PG   2

 There is Order in the data
    HS =0, UG=1, PG=2
    HS<UG<PG


# In[3]:


import numpy as np
import pandas as pd


# In[4]:


df = pd.read_csv("customer.csv")


# In[4]:


df.sample(5)


# CategoricalColumns 
      gender | review | education | purchased
      Convert all into Numerical  


# In[ ]:


Ordinal Encoding on | review | education |
Label Encoding | purchased |


# In[5]:


df=df.iloc[:,2:]


# In[6]:


df.head()


# In[22]:


from sklearn.model_selection import train_test_split

# x,y = train_test_split(x,y,test_size = 0.2)
x_train,x_test,y_train,y_test = train_test_split(df.iloc[:,0:2],df.iloc[:,-1],test_size=0.2)


# In[23]:


from sklearn.preprocessing import OrdinalEncoder


# In[24]:


# Define your categorical data
oe = OrdinalEncoder(categories = [['Poor','Average','Good'],['School','UG','PG']])


# In[10]:


oe


# In[25]:


oe.fit(x_train)

# It's used to train or fit a model to a dataset. 
# In this case, it's being applied to the "oe" object, 
# which suggests that you are training or fitting something using the training data provided.


# In[26]:


x_train.sample(2)  # DataFrame form


# In[27]:


x_test.sample(2)  # DataFrame form


# In[29]:


x_train = oe.transform(x_train)
x_test = oe.transform(x_test)


# In[20]:


x_train[1:4]  # Array form


# In[45]:


x_test[1:4]  # Array form


# In[33]:


y_test.sample(4) # It is as it is now apply label incoding for this


# In[ ]:


Label Encoding only for O/p column y | Dont use label encoding for input Columns

Encode target labels with value between 0 and n_classes-1.

This transformer should be used to encode target values, i.e. y, and not the input X.


# In[35]:


from sklearn.preprocessing import LabelEncoder


# In[36]:


le = LabelEncoder()


# In[37]:


le.fit(y_train)


# In[38]:


le.classes_


# In[39]:


y_train = le.transform(y_train)
y_test = le.transform(y_test)


# In[46]:


y_train[1:11]


# In[47]:


y_test[1:11]


# ### One Hot Encoding

# In[ ]:


Nomimal data is handled by One Hot Encoding


# In[ ]:


Multicolinearity | Relationship between input columns


# In[5]:


import numpy as np
import pandas as pd


# In[6]:


df=pd.read_csv("cars.csv")


# In[7]:


df.head()


# In[8]:


df["brand"].value_counts()


# In[9]:


df["brand"].unique()


# #### There are 32 brands.

# In[10]:


df["brand"].nunique()


# #### There are 4 types of Fuel

# In[11]:


df["fuel"].value_counts()


# In[31]:


df["fuel"].nunique()


# #### 1. One Hot Encoding using Pandas

# In[12]:


pd.get_dummies(df, columns = ['fuel','owner'])


# ### 2. k-1 One Hot Encoding using Pandas

# In[ ]:


There is a problem using pandas, becs pandas dont remember the columns position at which 
position it has mentioned the columns before.


# In[13]:


pd.get_dummies(df,columns = ['fuel','owner'], drop_first = True)

# It will drop the fist column from "Fuel" and "Owner"
# 10 columns will be Remain


# ### 3. One Hot Encoding using Sklearn

# In[14]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df.iloc[:,0:4],df.iloc[:,-1], test_size = 0.3,  
                                                 random_state = 0)
                                                          


# In[67]:





# In[ ]:


# The orignal DataFrame is as it is, as we started.


# In[15]:


df.head(5)


# In[54]:


x_train.head(5)


# In[16]:


x_test.head(5)


# In[17]:


from sklearn.preprocessing import OneHotEncoder


# In[ ]:


use "OneHotEncoder" and assign an Object for this


# In[18]:


ohe = OneHotEncoder()


# In[73]:


#seprate the columns "fuel" and "owner" from DataFrame and use OneHotEncoder only for them
#again add with these columns with the DataFrame, than only will get the complete input


# In[19]:


ohe.fit_transform(x_train[['fuel','owner']])

# It will produce the sparse matrix
# If most of the elements of the matrix have 0 value, then it is called a sparse matrix


# In[20]:


# Store in x_train_new

x_train_new= ohe.fit_transform(x_train[['fuel','owner']]).toarray()


# In[21]:


x_test_new= ohe.fit_transform(x_test[['fuel','owner']]).toarray()


# In[22]:


x_train_new


# In[23]:


x_train_new.shape


# In[24]:


x_train # we have x_train will all train cols


# In[47]:


df = x_train[['brand','km_driven']] 

# Extract the two columns [brand],[km_driven] from 'x_train' and append in 'x_train_new'
# Numpy array lets B


# In[48]:


df 


# In[51]:


x_train_new


# In[53]:


df1 = pd.DataFrame(x_train_new)


# In[54]:


df1


# In[66]:


# Horizontally Stack with each other "x_train_new" and "x_train"


# In[58]:


df2 = np.hstack((df , df1))


# In[59]:


df3 =  pd.DataFrame(df2)


# In[60]:


df3


# In[62]:


df3.columns = ['brand','km_driven','Diesel','LPG','Petrol','Fourth','Second','Test','Third']


# In[63]:


df3


# In[29]:


np.hstack((x_train[['brand','km_driven']].values , x_train_new)).shape


# In[30]:


np.hstack((x_train[['brand','km_driven']].values , x_train_new))


# In[31]:


x_train_new # It will print the same numbers of columns
            # that is presents


# In[32]:


ohe = OneHotEncoder(drop = 'first') # You can remove the fist column
                                 # to remove multicolinearilty


# In[33]:


x_train_new= ohe.fit_transform(x_train[['fuel','owner']]).toarray()


# In[34]:


x_test_new= ohe.fit_transform(x_test[['fuel','owner']]).toarray()


# In[35]:


x_train


# In[36]:


df1 = np.hstack((x_train[['brand','km_driven']].values , x_train_new))


# In[39]:


df2 = pd.DataFrame(df1)


# In[40]:


df2


# In[ ]:




