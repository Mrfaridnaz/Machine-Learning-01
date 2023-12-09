#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[32]:


df = pd.read_csv('placement.csv')


# In[33]:


df.head()


# In[34]:


plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df['cgpa'])

plt.subplot(1,2,2)
sns.distplot(df['placement_exam_marks'])

plt.show()


# In[35]:


df['placement_exam_marks'].describe()


# In[36]:


sns.boxplot(df['placement_exam_marks'])


# In[37]:


# Finding the IQR
percentile25 = df['placement_exam_marks'].quantile(0.25)
percentile75 = df['placement_exam_marks'].quantile(0.75)


# In[39]:


percentile75


# In[40]:


iqr = percentile75 - percentile25


# In[41]:


iqr


# In[42]:


upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr


# In[43]:


print("Upper limit",upper_limit)
print("Lower limit",lower_limit)


# ## Finding Outliers

# In[44]:


df[df['placement_exam_marks'] > upper_limit]


# In[45]:


df[df['placement_exam_marks'] < lower_limit]


# ## Trimming

# In[46]:


new_df = df[df['placement_exam_marks'] < upper_limit]


# In[47]:


new_df.shape


# In[48]:


# Comparing

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['placement_exam_marks'])

plt.subplot(2,2,2)
sns.boxplot(df['placement_exam_marks'])

plt.subplot(2,2,3)
sns.distplot(new_df['placement_exam_marks'])

plt.subplot(2,2,4)
sns.boxplot(new_df['placement_exam_marks'])

plt.show()


# ## Capping

# In[49]:


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


# In[ ]:


np.where(condtion,true,false)


# In[50]:


new_df_cap.shape


# In[51]:


# Comparing

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['placement_exam_marks'])

plt.subplot(2,2,2)
sns.boxplot(df['placement_exam_marks'])

plt.subplot(2,2,3)
sns.distplot(new_df_cap['placement_exam_marks'])

plt.subplot(2,2,4)
sns.boxplot(new_df_cap['placement_exam_marks'])

plt.show()


# In[ ]:




