#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# There are two methods for changing Numerical data to categorical data.

1. Disceretization or Binning.
2. Binarization.


# In[ ]:


Disceretization is the process of transforming continuous variable into discrete variable by 
creating by creating a set of contiguous intervals that span the range of the variable's values.
Discretization is also called Binning, where bin is an alternative name for interval.


# In[ ]:


get_ipython().run_line_magic('pinfo', 'Disceretization')
1. To handle the Outliers.
2. To improve the Value spread.


# In[ ]:


Types of Disceretization.
   1. Unsuperwised Binning.
      A. Equal wirth Binning or Uniform Binning.
      B. Equal frequency binning
      C. kmeans Binning.
        
   2. Superwise Binning.
      A. Decision Tree Binning.
    
   3. Custom Binning.


# #### A. Equal wirdh Binning or Unifor Binning.

# In[ ]:


bins = 10,
max-min/bins

Histogram will be ploted based on bin size


# In[ ]:


1. You can easily handle the outliers.
2. No change in Spread of data.


# #### B. Equal frequency binning

# In[ ]:


Interval = 10
Each interval contains 10% of total observtions


# In[ ]:


0 - 16 : 10 Percntile
16- 20 : 20 Percentile
20 - 22 : 30 Percentile
22 -30 : 40 Percentile
   
   wirth is not same


# 1. welll working on Outliers.
# 2. Make uniform the values spread.

# #### K-Means Binning

# In[ ]:


It will use clustering Algorethem, when the data is in cluster form mean groups
Interval is called centroid.


# #### Encoding the Discretized Variable.

# In[ ]:


calsss in sklearn : KBinsDiscretizer
    1. bins ?
    2. Strategy : 
        Uniform, Quantile, Kmeans
    3. Encoding    
       Ordinal, One Hot Encoding.


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer


# In[4]:


df = pd.read_csv("titanic_train.csv", usecols = ["Age","Fare","Survived"])


# In[5]:


df.sample(2)


# In[6]:


# Drop the rows with missing values

df.dropna(inplace = True)


# In[7]:


df.shape


# In[8]:


df.head()


# In[ ]:


# Withiut applying any Binning Transformation


# In[9]:


x = df.iloc[:, 1:]
y = df.iloc[:,0]


# In[10]:


x_train, x_test, y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)


# In[11]:


x_train.head(2)


# In[12]:


clf = DecisionTreeClassifier()


# In[13]:


clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)


# In[14]:


accuracy_score(y_test, y_pred)


# In[15]:


np.mean(cross_val_score(DecisionTreeClassifier(),x,y,cv=10, scoring = 'accuracy'))


# In[39]:


kbin_age = KBinsDiscretizer(n_bins =15, encode = 'ordinal',strategy = 'quantile')
kbin_fare = KBinsDiscretizer(n_bins =15, encode = 'ordinal',strategy = 'quantile')


# In[40]:


from sklearn.compose import ColumnTransformer

# 0 and 1 for index 0 for age and 1 for fare in DateTable
trf = ColumnTransformer([
    ('first', kbin_age, [0]),
    ('second', kbin_fare, [1])
])


# In[41]:


x_train_trf = trf.fit_transform(x_train)
x_test_trf = trf.transform(x_test)


# In[42]:


trf.named_transformers_


# In[25]:


trf.named_transformers_['first'].n_bins_


# In[43]:


trf.named_transformers_['second'].n_bins_


# In[ ]:





# In[44]:


trf.named_transformers_['first']


# In[45]:


trf.named_transformers_['first'].bin_edges_


# In[46]:


trf.named_transformers_['second'].bin_edges_


# In[47]:


output = pd.DataFrame({
    'age': x_train['Age'],
    'age_trf' : x_train_trf[:,0],
    'fare_trf': x_train_trf[:,1]
})


# In[48]:


output['age_labels'] = pd.cut(x=x_train['Age'],
                             bins = trf.named_transformers_['first'].bin_edges_[0].tolist())

output['fare_labels'] = pd.cut(x=x_train['Fare'],
                             bins = trf.named_transformers_['second'].bin_edges_[0].tolist())


# In[49]:


output.sample(5)


# In[ ]:


28:00


# In[50]:


clf  = DecisionTreeClassifier()
clf.fit(x_train_trf, y_train)
y_pred2 = clf.predict(x_test_trf)


# In[51]:


accuracy_score(y_test, y_pred2)


# In[52]:


x_trf = trf.fit_transform(x)
np.mean(cross_val_score(DecisionTreeClassifier(),x,y, cv = 10, scoring = 'accuracy'))


# In[65]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Replace 'x' and 'y' with your actual dataset and target variable
# For example:
# x = your_features
# y = your_target_variable

def discretize(bins, strategy):
    kbin_age = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=strategy)
    kbin_fare = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=strategy)

    trf = ColumnTransformer([
        ('first', kbin_age, [0]),
        ('second', kbin_fare, [1])
    ])

    x_trf = trf.fit_transform(x)  # Fix the typo here
    print(np.mean(cross_val_score(DecisionTreeClassifier(), x, y, cv=10, scoring='accuracy')))

    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    plt.hist(x['Age'])  # Assuming Age is the first column
    plt.title("Before")

    plt.subplot(122)
    plt.hist(x_trf[:, 0], color='red')
    plt.title('After')

    plt.show()
    
    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    plt.hist(x['Fare'])  # Assuming Age is the first column
    plt.title("Before")

    plt.subplot(122)
    plt.hist(x_trf[:, 0], color='red')
    plt.title('After')

    plt.show()


# In[66]:


discretize(bins=5, strategy='uniform')  # Adjust 'bins' and 'strategy' as needed


# In[ ]:





# #### Custom/Domain Based Binning

# In[ ]:


If you have the knolodge of data than you can Binning acc to you.


# ### Binarization

# In[ ]:


We change the continuous value in Discrete value in Discritization
here, we will change the numerical values in Binanry(0,1)


# In[ ]:


Image processing, tex


# In[ ]:


1. Thresold
2. Copy


# In[86]:


df = pd.read_csv("titanic_train.csv")[['Age','Fare','SibSp','Parch','Survived']]


# In[87]:


df.dropna(inplace = True)


# In[88]:


df.sample(4)


# In[89]:


df['family'] = df['SibSp'] + df['Parch']


# In[90]:


df.sample(4)


# In[91]:


df.drop(columns = ['SibSp','Parch'], inplace = True)


# In[92]:


df.sample(4)


# In[94]:


x= df.drop(columns = ['Survived'])
y= df['Survived']


# In[95]:


x_train, x_test , y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)


# In[96]:


x_train.head()


# In[84]:





# In[97]:


# Without Binarization

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

accuracy_score(y_test, y_pred)


# In[98]:


np.mean(cross_val_score(DecisionTreeClassifier(),x,y, cv = 10, scoring = 'accuracy'))


# In[103]:


# Applying Binarization

from sklearn.preprocessing import Binarizer


# In[106]:


trf = ColumnTransformer([
    ('bin', Binarizer(copy = False),['family'])
], remainder = 'passthrough')


# In[107]:


x_train_trf = trf.fit_transform(x_train)
x_test_trf = trf.transform(x_test)


# In[108]:


pd.DataFrame(x_train_trf, columns = ['family','Age','Fare'])


# In[109]:


clf = DecisionTreeClassifier()
clf.fit(x_train_trf, y_train)

y_pred2 = clf.predict(x_test_trf)

accuracy_score(y_test, y_pred2)


# In[ ]:





# In[110]:


x_trf = trf.fit_transform(x)
np.mean(cross_val_score(DecisionTreeClassifier(),x_trf,y, cv = 10, scoring = 'accuracy'))


# In[ ]:




