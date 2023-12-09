#!/usr/bin/env python
# coding: utf-8

# ### 1. What are outliers?
# 
# Outliers are data points that significantly deviate from the majority of the dataset, 
# exhibiting values or characteristics that are rare or unexpected. Outliers can distort the result
# of statistical analyses and predictive models, leading to inaccurate or biased outcomes.

# ### 2. Efect of Outliers on ML algorrithem.

# In[ ]:


1. Linear regression.
2. Logistics regression.
3. Deep Learning.
4. Adaboost.


# ### 3. How to treat outliers

# In[ ]:


1. Trimming.
2. Capping.
3. Assuming outliers as missing value.
4. Distretization: 1-10, 10-20


# In[ ]:


1. Normal Distribution.

m-3s > outliers > m+3s


# In[ ]:


2. Skewed Distribution.
 Boxplot
    
Q1-1.5IQR > Outliers < Q1+1.5IQR 


# In[ ]:


3. Percentile based approach
2.5% > Outliers > 97.5%


# In[ ]:


Techniques for Outliers Detection and Removal
1. z-score treatment.
2. IQR based filtering.
3. Percentile.
4. Winserization.


# ### Normal Distributed Data

# #### 1. Z-Score Methos

# In[ ]:


Assumptions-
1. Column should be normally distributed.

m-s and m+s = 68.2%
m-2s and m+2s = 95.4%
m-3s and m+3s = 99.7%

Z-Score xi' = xi-m/s


# In[51]:


li = [12, 13, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 27]


# In[53]:


sum = 0
for i in li:
    sum = sum+i  
print(sum)    


# In[55]:


mean = sum/ len(li)


# In[57]:


mean = 19


# In[ ]:





# In[ ]:





# ### Working on Data

# In[114]:


import pandas as pd
df = pd.read_csv('placement (3).csv')


# In[115]:


df.shape


# In[116]:


df.sample(6)


# In[117]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (16,5))
plt.subplot(1,2,1)
sns.distplot(df['cgpa'])

plt.subplot(1,2,2)
sns.distplot(df['placement_exam_marks'])

plt.show()


# In[118]:


df['cgpa'].skew()


# In[ ]:


# We can apply Z-Score only for CGPA becs it is Normally Distributed.


# In[119]:


print("mean value of cgpa", df["cgpa"].mean())
print("std value of cgpa", df["cgpa"].std())
print("min value of cgpa", df["cgpa"].min())
print("max value of cgpa", df["cgpa"].max())


# In[120]:


# Finding the Boundary Value

print("Highest allowed", df["cgpa"].mean()+3*df['cgpa'].std())
print("lowest allowed", df['cgpa'].mean()-3*df['cgpa'].std())


# In[91]:


# Finding the Outliers

Outliers= df[  (df['cgpa']>8.80) | (df['cgpa']<5.11)  ]


# In[92]:


Outliers


# In[ ]:





# In[98]:


# Finding the Outliers

df1= df[  (df['cgpa']<8.80) & (df['cgpa']>5.11)  ]


# In[99]:


df1


# ### Appproach 2

# In[100]:


# Calculating the Z-Score

df['cgpa_zscore'] = (df['cgpa']-df['cgpa'].mean())/df['cgpa'].std()


# In[102]:


df.head()


# In[103]:


# Place the student whose z-score in the range of -3 t0 +3.

df[(df['cgpa_zscore']>3) | (df['cgpa_zscore']< -3)]


# In[105]:


df[(df['cgpa_zscore']<3) & (df['cgpa_zscore']> -3)]


# ## Capping

# In[ ]:


Dont let the data go out of the Upper Limit and lower limit
without deleting the Data


# In[106]:


upper_limit = df['cgpa'].mean() + 3*df['cgpa'].std()
lower_limit = df['cgpa'].mean() - 3*df['cgpa'].std()


# In[108]:


import numpy as np
df['cgpa'] = np.where(
   df['cgpa']>upper_limit,
   upper_limit,
    np.where(
        df['cgpa']<lower_limit,
         lower_limit,
          df['cgpa']
    )
)


# ### Skewed Distributed Data

# In[ ]:


1. boxplot | IQR


# In[ ]:


Box-Plot-and-Whisker-Plot-1.png


# In[109]:


from PIL import Image
from IPython.display import display  # for Jupyter Notebook or IPython

# Open an image file
image = Image.open("Box-Plot-and-Whisker-Plot-1.png")  # Replace 'your_image.jpg' with the actual image file path

# Display the image (in Jupyter Notebook or IPython)
display(image)


# In[ ]:


100 percentile mean max value in the data
0 percentile mean min value in the data


# In[121]:


df1 = pd.read_csv('placement (3).csv')


# In[122]:


df1 


# In[123]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (16,5))
plt.subplot(1,2,1)
sns.distplot(df['cgpa'])

plt.subplot(1,2,2)
sns.distplot(df['placement_exam_marks'])

plt.show()


# In[124]:


df1['placement_exam_marks'].describe()


# In[126]:


import seaborn as sns

sns.boxplot(df['placement_exam_marks'],orient = 'h')


# In[ ]:


# we hvae outliers in this cols at max side,
#we have to find out the outlier and solve it.
1. trimming 2. Capping.


# In[127]:


# Finding the IQR
percentile25 = df['placement_exam_marks'].quantile(0.25)
percentile75 = df['placement_exam_marks'].quantile(0.75)


# In[128]:


percentile25 


# In[129]:


IQR = percentile75-percentile25


# In[130]:


IQR


# In[135]:


Upper_limit = percentile75 + 1.5*IQR
Lower_limit = percentile25 - 1.5*IQR


# In[136]:


Upper_limit 


# In[137]:


Lower_limit 


# ### Finding outliers

# In[138]:


df[df['placement_exam_marks']>Upper_limit]


# In[139]:


df[df['placement_exam_marks']<Lower_limit]


# ### Trimming

# In[142]:


new_df = df[df['placement_exam_marks']<Upper_limit]


# In[143]:


new_df.shape


# In[147]:


# Comapre

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (16,8))
plt.subplot(2,2,1)
sns.distplot(df['placement_exam_marks'])

plt.subplot(2,2,2)
sns.boxplot(df['placement_exam_marks'], orient = 'h')

plt.subplot(2,2,3)
sns.distplot(new_df['placement_exam_marks'])

plt.subplot(2,2,4)
sns.boxplot(new_df['placement_exam_marks'], orient = 'h')

plt.show()


# In[160]:


upper_limit


# ### Capping

# In[161]:


df = pd.read_csv('placement (3).csv')


# In[166]:


upper_limit


# In[167]:


lower_limit


# In[169]:


new_df_cap = df.copy()


# In[170]:


new_df_cap


# In[177]:


import numpy as np


new_df_cap = df.copy()

new_df_cap['placement_exam_marks'] = np.where(
    new_df_cap['placement_exam_marks'] > upper_limit,
    upper_limit,
    np.where(
        new_df_cap['placement_exam_marks'] < lower_limit,
        lower_limit,
        new_df_cap['placement_exam_marks']
    )
)

# np.where(condition, True, False)


# In[163]:


new_df_cap


# In[171]:


new_df_cap.shape


# In[178]:


# Comapre

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (16,8))
plt.subplot(2,2,1)
sns.distplot(df['placement_exam_marks'])

plt.subplot(2,2,2)
sns.boxplot(df['placement_exam_marks'], orient = 'h')

plt.subplot(2,2,3)
sns.distplot(new_df_cap['placement_exam_marks'])

plt.subplot(2,2,4)
sns.boxplot(new_df_cap['placement_exam_marks'], orient = 'h')

plt.show()


# In[ ]:





# In[ ]:




