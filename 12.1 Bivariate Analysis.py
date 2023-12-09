#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


df = pd.read_csv("iris.csv")  


# In[4]:


df.sample(2)


# In[13]:


titan = pd.read_csv("titanic_train.csv")


# In[6]:


titan.sample(2)


# In[7]:


tips = pd.read_csv("tips.csv")


# In[8]:


tips.sample(2)


# In[9]:


flights = pd.read_csv("flights.csv")


# In[10]:


flights.sample(2)


# In[11]:


import seaborn as sns


# ### 1. Scatterplot (Numerial-Numerical)

# In[29]:


sns.scatterplot(data=tips, x="total_bill",y="tip", hue ="sex" )

# linear relationship between "total_bill" and "tip"


# In[34]:


sns.scatterplot(data=tips, x="total_bill",y="tip",style = "smoker", hue ="sex" )


# In[36]:


sns.scatterplot(data=tips, x="total_bill",y="tip",style = "smoker", hue ="sex" , size = "size")


# ### 2. Bar plot (Nuumerical-Categorical)

# In[ ]:


# Bar Plot is used when one is categorical and One is Numerical
# x axis categories and y-axs numerical


# In[38]:


titan.sample(2)


# In[40]:


sns.barplot(data= titan, x="Pclass",y="Age")

# Black line is showing the confidence interval
# Bar plots are particularly useful for visualizing categorical data, showing the relationship 
# between one categorical variable and another numerical variable.

# in this graph we can see that younger passengers in 3-Pclass


# In[ ]:


# What is the relationship between "Pclass" and "fair"


# In[43]:


sns.barplot(data= titan, x="Pclass",y="Fare")

# Class 1 has the heighest fare and class 3 has lowest fare


# In[44]:


sns.barplot(data= titan, x="Pclass",y="Fare", hue = "Sex")


# In[45]:


sns.barplot(data= titan, x="Pclass",y="Age", hue = "Sex")


# ### 3. Box Plot (Numerical - Categorical)

# In[54]:


sns.boxplot(data=titan, x="Age", y="Sex",orient="h")


# In[55]:


sns.boxplot(data=titan, x="Age", y="Sex",orient="h", hue = "Survived")


# ### 4. Distplot (Numerical - Categorical)

# In[57]:


sns.distplot(titan["Age"])


# In[12]:


sns.distplot(titan[titan["Survived"]==0]["Age"])

# It will represent the Probability density Fucntion for age for thoes who didnt "survived"


# In[60]:


sns.distplot(titan[titan["Survived"]==0]["Age"])
sns.distplot(titan[titan["Survived"]==1]["Age"])


# In[64]:


sns.distplot(titan[titan["Survived"]==0]["Age"], hist = False)
sns.distplot(titan[titan["Survived"]==1]["Age"], hist = False)

# Blue curve- Died || Orange curve-Alive


# ### 5. HeatMap (Categorical - categorical)

# In[65]:


titan.sample(2)


# In[66]:


# How many people died or alive in Pclass


# In[67]:


pd.crosstab(titan["Pclass"],titan["Survived"])

# This information between two categorical columns


# In[69]:


sns.heatmap(pd.crosstab(titan["Pclass"],titan["Survived"]))


# In[ ]:


# percentage of people survived in Pclass


# In[72]:


(titan.groupby("Pclass").mean()["Survived"]*100).plot(kind="bar")


# In[74]:


(titan.groupby("Sex").mean()["Survived"]*100).plot(kind="bar")

# Female "survived" percentage is more than male


# ### 6. cluster Map (Catogorical - Catogorical)

# In[76]:


pd.crosstab(titan["SibSp"],titan["Survived"])


# In[79]:


sns.clustermap(pd.crosstab(titan["Parch"],titan["Survived"]))


# ### 7. Pairplot

# In[83]:


df.sample(2)


# In[86]:


sns.pairplot(df)


# In[87]:


sns.pairplot(df, hue = "species")


# ### 8. Lineplot (Numrical - Numerical)

# In[89]:


# x axis should have time based data (Year, week, day)


# In[95]:


flights.sample(2)


# In[96]:


flights.groupby("year").sum()


# In[105]:


import matplotlib.pyplot as plt
plt.plot(fl["year"], fl["passengers"])


# In[ ]:




