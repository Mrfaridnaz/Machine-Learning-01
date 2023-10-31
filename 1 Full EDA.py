#!/usr/bin/env python
# coding: utf-8

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


# In[1]:


# Find the Data from github Naik sir


# In[ ]:


we are going to develope any M/L application or 
Deep Learning application so that we have certain stagses
1. Data Collection or data gathering.
2. EDA means Analysis of the Data (Use python Script) Exploratory Data Analysis
3. Preprocess the Data (Feature Engineering.
4. Model Building.
5. Evaluate the Model (Accuracy of the Model or Performance the Model)
                        
These are core steps when we are solving any M/L problem.


# In[ ]:


When we have the Data
1. Basis profile from Data
2. Stats based Analysis
   a. By writing a python code.
   b. Visulize the Data
3. Python logic or Python Code for extracting diff diff pattern
4. Pandas Libarary


# In[4]:


import pandas as pd
df=pd.read_csv('titanic_train.csv')


# # 1 Check the size of the Data

# In[5]:


df.shape


# In[6]:


df.size  # col*raw
# Return the memory usage of each column in bytes
# Multiplication of Rows and Columns


# In[7]:


df.memory_usage()


# In[9]:


df.memory_usage(deep=True)
# Memory accupied in bytes


# # 2. How the Data Looks likes

# In[10]:


df.head()


# In[11]:


df.tail() # By defalt it prints 5 Rows


# In[15]:


df.tail(2)


# In[13]:


df.sample() # Random sample from the Data


# In[14]:


df.sample(5) # 5 Random sample from the Data


# # 3. Data Type of the Column

# In[17]:


df.dtypes


# In[18]:


df.info


# In[19]:


df.info()

# How many null values in the data


# # 4. How the data looks like mathematically

# In[20]:


df.describe() 
# It will print only the Numwerical Data


# In[21]:


df.describe().T


# # 5. check missing Values

# In[22]:


df.isnull()

# Where there is null value it will show True


# In[23]:


# Check the Total number of Null values in each column

df.isnull().sum()


# In[24]:


df.isnull().sum().sum()

# Total number of missing values


# In[ ]:


# if you have 20-30% missing data inside your Data,There is a criateria 
#It will check the missing data about the perticular column or the entire data


# # 6. Duplicate values

# In[ ]:


1:00:00 min


# In[26]:


df.duplicated()

# If there is any duplicate it will print True


# In[27]:


# find the sum or how many the Duplicate values are there.
df.duplicated().sum()


# In[29]:


df[df.duplicated()]

# Check how many the Duplicate values are there in form of dataFrame


# In[ ]:


# There is no Duplicate values


# In[ ]:


# Check the Unique Values


# In[30]:


df.nunique()


# In[31]:


# Correlation between the columns

df.corr()


# In[32]:


corr_mat = df.corr()


# In[33]:


import seaborn as sns
sns.heatmap(corr_mat)

# Correlation between the Column
# Ligjht color = Highly correlleted
# dark one = less correleted


# In[34]:


sns.heatmap(corr_mat, annot = True)

# showing the pearsion Correletion


# # Univariate Analysis || Bivariate Analysis || multivariate Analysis

# In[ ]:


#in univarite analysis independent analysis regarding the Column


# In[36]:


df.head()


# In[38]:


df.columns
# Print all the Column name


# In[ ]:


# In univariate you can take any one column and analys that column


# In[ ]:


# Basis understanding regarding the Coulumn or variable
# Categorical var || Numeric var || 


# In[39]:


1:16:00


# In[40]:


# Column name insite the list


# In[48]:


cat_feature = [columns for columns in df.columns if df[columns].dtype == "O"]

# Print the Column that are object data type, O for Object Capital O
# Print the table with objective data type


# In[50]:


df[cat_feature]


# In[ ]:


# Print the Column that are Numeric data type, non-objective


# In[49]:


Num_feature =[columns for columns in df.columns if df[columns].dtype!= "O"]


# In[51]:


df[Num_feature]


# In[52]:


# Performing the univariant analysis
# Two types of the Columns here (Cat_Column and Num_column)


# # Categorical Column

# #### 1. count plot

# In[60]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=df, x="Embarked")
plt.show()


# In[64]:


sns.countplot(data=df, x="Sex")


# In[65]:


sns.countplot(data=df, x="Pclass")


# In[68]:


df["Sex"]


# In[69]:


df["Sex"].value_counts()

# It will print how many male and female are there.


# In[71]:


df["Sex"].value_counts().plot(kind='bar')


# # 2. Pie Chart

# In[72]:


df["Sex"].value_counts().plot(kind='pie')


# In[73]:


df["Sex"].value_counts().plot(kind='pie', autopct='%.2f')

# define the data percentage bias


# In[75]:


df["Pclass"].value_counts().plot(kind='pie', autopct='%.2f')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Numerical Column

# # 1. Histogram

# In[76]:


plt.hist(df["Age"])

# distribution regarding the age
# A histogram is a graph showing frequency distributions.

# It is a graph showing the number of observations within each given interval.


# # Distplot

# In[77]:


sns.distplot(df["Age"])

# Bins are the interval


# In[78]:


# We can see from the graph, the probability of the people whose age between 20 to 40
# Most of the people 20 to 40
# There is interval by 10 year


# In[80]:


# create an another graph with 5 year of interval

plt.hist(df["Age"], bins = 5)


# In[81]:


# Divide the data into 20 Equal Groups or Intervals

plt.hist(df["Age"], bins = 20)


# # Boxplot

# In[86]:


sns.boxplot(df["Age"])

# It represent the disperssion of the data in terms of quantile


# In[84]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is your DataFrame
sns.boxplot(x=df["Age"], orient="h")  # Set orient to "h" for horizontal
plt.show()


# In[ ]:


# Lower fence || Q1 || Mean ||| Q3 || Upper fence || IQR || Outliers
# Box plot is imp for Univariant Analysis


# In[ ]:


1:48:00


# In[87]:


df["Age"].min()


# In[88]:


df["Age"].max()


# In[89]:


df["Age"].mean()


# In[90]:


df["Age"].median()


# In[ ]:


# Skweness of the Data


# In[91]:


df["Age"].skew()*100
# 38.9 % of the Data is skwed


# In[94]:


(100-38.910778230082705)
#61.08 % of the Data is Normal Distributed


# In[ ]:


This is all about the Univariant 


# In[ ]:


1:52:00


# # Bivariant Analysis

# In[95]:


# Categorical var || Numerical Var


# In[96]:


x--> categorical y--> categorical
x--> Numerical y--> Numerical
x--> categorical y--> Numerical


# # Multivariant Analysis

# In[ ]:


# More than two columns will be involved


# In[174]:


tips = sns.load_dataset("tips")


# In[99]:


flight= sns.load_dataset("flights")


# In[101]:


iris = sns.load_dataset("iris")


# # 1. scatter plot is used when the x and y are numeric column

# In[175]:


tips


# In[176]:


import seaborn as sns

sns.scatterplot(x=tips["total_bill"], y=tips["tip"])

# This is bivariant where we are analysing total bill with respect to tip


# In[177]:


# We can see from the scatter plot
as the bill is increasing the tip increasing too


# In[178]:


sns.scatterplot(x=tips["total_bill"], y=tips["tip"], hue = tips["sex"])
# Multivariant analysis || 


# In[108]:


sns.scatterplot(x=tips["total_bill"], y=tips["tip"], hue = tips["sex"], style= tips["smoker"])


# In[111]:


sns.scatterplot(x=tips["total_bill"], y=tips["tip"], hue = tips["sex"], style= tips["smoker"], size = tips["size"])


# In[113]:


# circle size represent the size of the family
# Scatter plot is applied only for numerical column


# # 2. Bar plot(x is numerical and y is categorical)

# In[125]:


import pandas as pd
df=pd.read_csv('titanic_train.csv')


# In[126]:


df.head()


# In[128]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.barplot(x=df["Pclass"], y=df["Age"])

# Age distribution wrt diff diff classes


# In[129]:


sns.barplot(x=df["Pclass"], y=df["Age"], hue = df["Sex"])


# # 3. Box plot(x is numerical and y is categorical)

# In[132]:


sns.boxplot(x=df["Sex"], y=df["Age"])

# You can add more parameter, can check by pressing Shift+Tab


# In[133]:


sns.boxplot(x=df["Sex"], y=df["Age"], hue = df["Survived"])


# In[134]:


df


# In[137]:


df["Survived"]

# 0 who has not Survived and 1 for who survived


# In[139]:


df["Survived"]==1

# True is for who survived
# False is for who didnt survived


# In[141]:


df[df["Survived"]==1]["Age"]

# It all the data who survived
# 342 people


# In[144]:


df[df["Survived"]==1]["Age"].max()

# Max age of the person who survived


# In[145]:


df[df["Survived"]==1]["Age"].min()

# Min age of the person who survived


# In[146]:


df[df["Survived"]==0]["Age"].min()

# Min age of the person who didn't survived


# In[152]:


sns.displot(df[df["Survived"]==1]["Age"],kde = True)


# In[159]:


sns.distplot(df[df["Survived"]==1]["Age"], hist=False)
sns.distplot(df[df["Survived"]==0]["Age"], hist=False)


# There is comparison b/w for those who survived or not survived


# # 4. heatmap plot(x is categorical and y is categorical)

# In[160]:


df.head()


# In[161]:


df["Pclass"]


# In[163]:


df["Survived"]


# In[166]:


pd.crosstab(df["Pclass"],df["Survived"])

# 0 mean Non-survived
# 1 mean Survived

# 1, 2 and 3 are the categories


# In[167]:


sns.heatmap(pd.crosstab(df["Pclass"],df["Survived"]))

# light color show 
# dark color show


# In[168]:


sns.heatmap(pd.crosstab(df["Pclass"],df["Survived"]), annot = True)


# # Pair Plot

# In[170]:


iris.head()


# In[171]:


sns.pairplot(iris)


# In[172]:


sns.pairplot(iris,hue="species")

# Three categories are here.


# # Line Plot

# In[182]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.lineplot(x="total_bill", y="tip", data=tips)
plt.show()


# In[183]:


sns.scatterplot(x="total_bill", y="tip", data=tips)
plt.show()

#Connect all the points, you will get the line plot


# In[ ]:


# When you are plotting a line plot than try to take time related data


# In[186]:


flight.groupby("year").sum()


# In[187]:


flight.groupby("year").sum().reset_index()


# In[191]:


#sns.lineplot(flight["month"],flight["passengers"])

import seaborn as sns
import matplotlib.pyplot as plt  # Importing matplotlib for plotting

flight_new = sns.lineplot(x=flight["month"], y=flight["passengers"])  # Assuming flight is your DataFrame
plt.show()  # Display the plot


# In[195]:





# In[198]:


import seaborn as sns  # Import the Seaborn library if not already imported

# Assuming you have a DataFrame named flight_new with columns "year" and "passengers"
sns.lineplot(x=flight_new["year"], y=flight_new["passengers"])


# In[199]:


flight.pivot_table(values="passengers",index="month")


# In[202]:


flight.pivot_table(values="passengers",index="month", columns="year")


# In[203]:


sns.heatmap(flight.pivot_table(values="passengers",index="month", columns="year"))


# In[205]:


sns.heatmap(flight.pivot_table(values="passengers",index="month", columns="year"), annot = True)


# In[206]:


#https://pypi.org/project/pandas-profiling/
#Pandas profiling


# In[23]:


import pandas as pd
df=pd.read_csv('titanic_train.csv')


# In[24]:


from pandas_profiling import ProfileReport
profile = ProfileReport(df, title = "Pandas Profiling Report")


# In[ ]:





# In[ ]:




