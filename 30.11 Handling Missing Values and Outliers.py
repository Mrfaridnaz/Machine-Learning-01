#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Handling missing values


# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np

import plotly.express as px
import bokeh

import scipy.stats as stats
from sklearn import datasets, linear_model

import statsmodels.api as sm


# In[3]:


df=pd.read_csv("titanic_train.csv")


# In[11]:


df


# In[12]:


df=df[["Age","Fare","Survived","SibSp"]]


# In[14]:


df.head()


# In[15]:


df.info()


# In[16]:


df.isnull().sum()


# In[15]:


# Percentage bias missing Data


# In[17]:


df.isnull().sum()/891*100
#df.isnull().sum().mean()


# In[24]:


# Numerical Column ---> mean and mode
# categorical Column ---> Mode


# In[19]:


from sklearn.model_selection import train_test_split

x_train, x_test = train_test_split(df,test_size=0.25)


# In[20]:


x_train


# In[21]:


x_test


# In[29]:


# Handling Missing values in Age


# In[23]:


mean_age=x_train["Age"].mean()     # mean of the Age


# In[25]:


mean_age


# In[29]:


max_age=x_train["Age"].max() 


# In[30]:


max_age


# In[31]:


min_age=x_train["Age"].min() 


# In[32]:


min_age


# In[ ]:





# In[ ]:





# In[32]:


# Fill all mising values with this mean value


# In[35]:


x_train["Age"].fillna(mean_age)


# In[40]:


x_train["Age"].fillna(mean_age).isnull().sum()

# Check how many Null values are there


# In[41]:


# If there are outliers present in the data than fill the missing values by Median instead of mean


# In[42]:


median_age=x_train["Age"].median()


# In[43]:


median_age


# In[44]:


x_train["Age"].fillna(median_age)


# In[45]:


# Same operation you can do with the help of Sklearn as well
# Simpleimputer


# In[ ]:


from sklearn.impute import SimpleImputer

# Create a SimpleImputer instance with the "mean" strategy
imputer = SimpleImputer(strategy='mean')

# Fit the imputer to your data and transform it (fill missing values with the mean)
X_imputed = imputer.fit_transform(X)


# # categorical data or Columns

# In[46]:


# cabin is a categorial data


# In[38]:


df1=pd.read_csv("titanic_train.csv")


# In[40]:


df1.sample(5)


# In[39]:


df1["Cabin"]


# In[55]:


df1["Cabin"].mode()

# here the count for any value that is shown.


# In[56]:


# Fill the missing values with 1st one

df1["Cabin"].fillna(df1["Cabin"].mode()[0])


# # Outliers Handling and Detector

# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np

import plotly.express as px
import bokeh

import scipy.stats as stats
from sklearn import datasets, linear_model

import statsmodels.api as sm


# In[2]:


# What is outliers


# In[ ]:


1. Trimming Data
2. Capping


# In[ ]:


# How to handle the non-normal Data
# Five number of summary 
1. Lower fence = q1-(1.5*IQR)
Reject the data point if it is less than Lower fence
2. Upeer fence = q3+(1.5*IQR)
Reject the data points if it is greater than Upper Fence


# In[ ]:


# 2. Percentile technique
Define the percentage that at what percentage the data is workable
let's 2 to 98 percent of the data is workable
if you find the data less than 2% and greater than 98%, discard the data.


# In[41]:


#pd.read_scv("File location/file name.csv")
df=pd.read_csv("placement (1).csv")


# In[42]:


df


# In[43]:


df.sample(5)


# In[44]:


df.describe().T


# In[47]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(16, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['placement_exam_marks'],kde=True)

plt.subplot(1, 2, 2)
sns.histplot(df['cgpa'],kde=True)

plt.show()


# In[29]:


# using the sdt deviation method to handle the outliers
# Discard the data that is away from std deviation


# In[7]:


df["placement_exam_marks"]


# In[8]:


# check how much the % data is skwed

df["placement_exam_marks"].skew()*100


# In[9]:


df["placement_exam_marks"].mean()


# In[35]:


df["cgpa"].mean()


# In[10]:


# Std Deviation

df["cgpa"].std()


# In[12]:


df["cgpa"].min()


# In[13]:


df["cgpa"].max()


# In[14]:


# Range of this data


# In[15]:


# Upper value ===> mean+3 STD
df["cgpa"].mean()+3*df["cgpa"].std()


# In[16]:


# Lower value ===> mean-3 STD
df["cgpa"].mean()-3*df["cgpa"].std()


# In[ ]:


"3.042"----5.11----6.96----8.80----"9.12"


# In[18]:


df1=df[df["cgpa"]<5.11] # less than lower value 3 STD


# In[19]:


df2=df[df["cgpa"]>8.80]


# In[20]:


result = pd.concat([df1, df2])
# pd.concat([df[df["cgpa"]<5.11],df[df["cgpa"]>8.80]])
# df[(df["cgpa"]<5.11) | df(df["cgpa"]>8.80)]  # | mean OR


# In[21]:


result
# Outliers


# # Trimming

# In[22]:


df_new=df[(df["cgpa"]>5.11) & (df["cgpa"]<8.80)]

# Print the data that is between 5.11 to 8.80


# In[23]:


df_new


# In[24]:


df_new.head()


# In[25]:


sns.histplot(df_new['cgpa'],kde=True)


# # Capping

# In[51]:


exam=pd.read_csv("placement (1).csv")


# In[52]:


exam.head(5)


# In[53]:


exam["cgpa"].mean()


# In[54]:


exam["cgpa"].std()


# In[55]:


LowerL=exam["cgpa"].mean()-3*exam["cgpa"].std()
UpperL=limit1=exam["cgpa"].mean()+3*exam["cgpa"].std()

print(LowerL)
print(UpperL)


# In[56]:


exam["cgpa"]=np.where(exam["cgpa"]>UpperL,UpperL,exam["cgpa"])

# np.where it is a conditianal statement
# UpperL,UpperL if the value is greater than Upper value that id 8.80 than asign uppper value
# mean the data dont exceed than 8.80.


# In[32]:


exam["cgpa"]=np.where(exam["cgpa"]<LowerL,LowerL,exam["cgpa"])

# np.where it is a conditianal statement
# LowerL,LowerL if the value is less than lower value that id 5.11 than asign lower value
# mean the data dont go below than 5.11.


# In[57]:


exam["cgpa"].min()


# In[58]:


LowerL


# In[59]:


UpperL


# In[36]:


exam["cgpa"].min()


# In[60]:


exam["cgpa"].max()


# In[37]:


exam[(exam["cgpa"]<5.11)]


# In[61]:


df.shape


# In[38]:


sns.displot(exam['cgpa'],kde=True)


# In[256]:


#IQR Method
# Inter Quantile Method


# In[39]:


sns.boxplot(x=df["placement_exam_marks"], orient="h")


# In[40]:


percentile25=df['placement_exam_marks'].quantile(0.25)


# In[41]:


percentile75=df['placement_exam_marks'].quantile(0.75)


# In[42]:


percentile25


# In[43]:


percentile75


# In[44]:


IQR=percentile75-percentile25


# In[45]:


IQR


# In[46]:


lower_fence = percentile25-1.5*IQR
upper_fence = percentile75+1.5*IQR


# In[47]:


lower_fence


# In[48]:


upper_fence


# In[49]:


df[df["placement_exam_marks"]>upper_fence]


# In[50]:


df[df["placement_exam_marks"]<lower_fence]


# In[55]:


nw_df=df[df["placement_exam_marks"]<=upper_fence]


# In[57]:


nw_df


# In[58]:


nw_df.shape


# In[59]:


sns.boxplot(x=nw_df["placement_exam_marks"], orient="h")


# In[61]:


#pd.read_scv("File location/file name.csv")
df=pd.read_csv("placement (1).csv")


# In[65]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(16, 8))

plt.subplot(2, 2, 1)
sns.distplot(df['placement_exam_marks'],kde=True)

plt.subplot(2, 2, 2)
sns.boxplot(df['placement_exam_marks'],orient="h")

plt.subplot(2, 2, 3)
sns.distplot(nw_df['placement_exam_marks'],kde=True)

plt.subplot(2, 2, 4)
sns.boxplot(nw_df['placement_exam_marks'],orient="h")


plt.show()


# In[95]:


nw_df.shape


# In[ ]:


# Capping method on top col(placement_exam_marks)


# # Percentile Method

# In[66]:


# Apply the percentage according to you || Decide your own range


# In[84]:


percentile10=df['placement_exam_marks'].quantile(0.30)
percentile85=df['placement_exam_marks'].quantile(0.70)


# In[85]:


percentile10


# In[86]:


percentile85


# In[87]:


IQR1=percentile75-percentile10


# In[88]:


IQR1


# In[77]:


df['placement_exam_marks'].mean()


# In[89]:


df['placement_exam_marks'].max()


# In[ ]:





# In[90]:


lower_fence1 = percentile10-1.5*IQR1
upper_fence1 = percentile85+1.5*IQR1


# In[91]:


lower_fence1


# In[79]:


upper_fence1


# In[96]:


df3=df[df['placement_exam_marks']<lower_fence1]
# Rejected Data


# In[101]:


df3


# In[102]:


df4=df[df['placement_exam_marks']>upper_fence1]
#Rejected Data


# In[103]:


df4


# In[104]:


df4=df[df['placement_exam_marks']<upper_fence1]


# In[106]:


df4


# In[108]:


df4.shape


# In[109]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(16, 8))

plt.subplot(2, 2, 1)
sns.distplot(df['placement_exam_marks'],kde=True)

plt.subplot(2, 2, 2)
sns.boxplot(df4['placement_exam_marks'],orient="h")

plt.subplot(2, 2, 3)
sns.distplot(df['placement_exam_marks'],kde=True)

plt.subplot(2, 2, 4)
sns.boxplot(df4['placement_exam_marks'],orient="h")


plt.show()


# In[ ]:





# In[ ]:




