#!/usr/bin/env python
# coding: utf-8

# #### Power Transformer | Box-Cox Transformer | Yeo - Johnson t/s

# In[ ]:


2. Power Transformer
   Box-Cox Transformer
   Yeo-Johnson Transformer


# In[ ]:


you can transform any distribution to Normal distribution with the help of these t/ms


# In[ ]:


# What is Box-Cox Transformer?
the values should be greater than 0 and positive.


# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats # For QQ plot


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from sklearn.preprocessing import PowerTransformer



# In[ ]:


The r2_score function from the sklearn.metrics module is used to evaluate the performance of a regression model in 
machine learning. Specifically, it calculates the R-squared (coefficient of determination) score, which is a 
statistical measure that represents the proportion of the variance in the dependent variable that is predictable from 
the independent variables.

In the context of regression models, R-squared is a common metric used to assess how well the model 
explains the variability in the target variable. The R-squared score ranges from 0 to 1, where:

0 indicates that the model does not explain any of the variability in the target variable.
1 indicates that the model explains all of the variability in the target variable.
Intermediate values between 0 and 1 indicate the proportion of variability explained by the model.


# In[3]:


df=pd.read_csv("concrete_data.csv")


# In[4]:


df.head()


# In[5]:


df.shape


# In[7]:


df.isnull().sum()


# In[8]:


df.describe()


# In[9]:


x=df.drop(columns = ["Strength"])
y = df.iloc[:, -1]


# In[42]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)


# In[43]:


# Applying Regression without any Transformation

lr = LinearRegression()

lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

r2_score(y_test, y_pred)


# In[44]:


# Cross checking with cross val score

lr = LinearRegression()
np.mean(cross_val_score(lr, x, y, scoring = 'r2'))


# In[45]:


# Potting the Distplots without any t/ms

for col in x_train.columns:
    plt.figure(figsize=(14, 4))
    plt.subplot(121)

    sns.histplot(x_train[col], kde=True)
    plt.title(col)

    plt.subplot(122)
    stats.probplot(x_train[col], dist="norm", plot=plt)
    plt.title(col)

    plt.show()



# ## Apply Box-Cox Transform

# In[46]:


from sklearn.preprocessing import PowerTransformer
import pandas as pd

pt = PowerTransformer(method='box-cox')

x_train_transformed = pt.fit_transform(x_train+0.00001)
x_test_transformed = pt.transform(x_test+0.00001)

pd.DataFrame({'cols': x_train.columns, 'box_cox_lambda': pt.lambdas_})


# In[48]:


# Applying linear Regression on Transformed Data

lr = LinearRegression()
lr.fit(x_train_transformed, y_train)

y_pred2 = lr.predict(x_test_transformed)

r2_score(y_test, y_pred2)


# In[ ]:


Cross-validation is a statistical technique used to assess how well a model generalizes to an independent dataset. 
The basic idea is to split the dataset into multiple subsets, train the model on some of these subsets, and then evaluate 
its performance on the remaining subset.


# In[50]:


# Using cross val score

from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np

pt = PowerTransformer(method='box-cox')
x_transformed = pt.fit_transform(x + 0.000001)

lr = LinearRegression()
r2_score = np.mean(cross_val_score(lr, x_transformed, y, scoring='r2'))


# In[51]:


r2_score


# In[55]:


# Before and after comparison for Box-Cox Plot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

x_train_transformed = pd.DataFrame(x_train_transformed, columns=x_train.columns)

for col in x_train_transformed.columns:
    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    sns.histplot(x_train[col], kde=True)  # Use sns.histplot instead of sns.distplot
    plt.title(col)

    plt.subplot(122)
    sns.histplot(x_train_transformed[col], kde=True)  # Use sns.histplot instead of sns.distplot
    plt.title(col)

    plt.show()

    
    
    


# In[62]:


# Apply Yeo-Johnson Tramsform


import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Assuming you have x_train, x_test, y_train, and y_test defined

# Fit and transform the training data using PowerTransformer
pt1 = PowerTransformer()
x_train_transformed2 = pt1.fit_transform(x_train)
x_test_transformed2 = pt1.transform(x_test)

# Fit a Linear Regression model
lr = LinearRegression()
lr.fit(x_train_transformed2, y_train)
y_pred3 = lr.predict(x_test_transformed2)

# Calculate the R-squared score
r2 = r2_score(y_test, y_pred3)
print("R-squared score:", r2)

# Access the lambda values and create a DataFrame
lambda_values = pd.DataFrame({'cols': x_train.columns, 'Yeo_Johnson_lambda': pt1.lambdas_})
print(lambda_values)


# In[64]:


# Apply cross val score

pt = PowerTransformer()
x_transformed2 = pt.fit_transform(x)

lr = LinearRegression()
np.mean(cross_val_score(lr, x, y, scoring = 'r2'))


# In[65]:


x_train_transformed2 = pd.DataFrame(x_train_transformed2, columns=x_train.columns)


# In[66]:


# Before and after comparison for Yeo-Johnson Plot

for col in x_train_transformed.columns:
    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    sns.histplot(x_train[col], kde=True)  # Use sns.histplot instead of sns.distplot
    plt.title(col)

    plt.subplot(122)
    sns.histplot(x_train_transformed2[col], kde=True)  # Use sns.histplot instead of sns.distplot
    plt.title(col)

    plt.show()


# In[67]:


# Side by side labmdas

pd.DataFrame({'cols': x_train.columns, 'box_cox_lambdas': pt.lambdas_, 'Yeo_johnson_lambdas':pt1.lambdas_})


# In[ ]:




