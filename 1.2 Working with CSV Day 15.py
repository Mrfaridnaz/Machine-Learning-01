#!/usr/bin/env python
# coding: utf-8

# #### 1. Importing Pandas

# In[2]:


import pandas as pd


# #### 2. Opening a Local CSV file

# In[ ]:


df=pd.read_csv('file name.csv')

# This is workable if you have your file in machine/Jupyter


# In[3]:


df=pd.read_csv('aug_train.csv')

# This is workable if you have your file in machine/Jupyter


# In[4]:


df


# #### 3 Opening a csv file from URL

# In[ ]:


import requests
import csv
from io import StringIO

# Replace the URL with the URL of the CSV file you want to open
csv_url = "https://example.com/yourfile.csv"
headers = {}
req = requests.get(csv_url,headers=headers)
data = StringIO(req.text)

pd.read_csv(data)


# #### 4. Sep Parameter

# In[ ]:


pd.read_csv('file_name.tsv',sep='\t')  # If the Data is separated by Tab


# In[ ]:


when you dont have the column name


# In[ ]:


pd.read_csv('file_name.tsv',sep='\t', names=['A','B','C','D'])


# #### 5. Index Col parameter

# In[ ]:


Convert any column into Index


# In[ ]:


df=pd.read_csv('file name.csv',index_col = 'A')


# #### 6. Header Parameter

# In[ ]:


Use fist row as column name of the data


# In[ ]:


df=pd.read_csv('file name.csv',header =1)


# #### 7. Use Cols Parameter

# In[ ]:


When you want only the specific columns in the data set with you want to work
instead of uploading all the data


# In[ ]:


pd.read_csv('file_name.csv', usecols = ['A','B','D','E'])


# #### 8. Squeeze Parameter

# In[ ]:


It will provide only 1 col in the form  of Series not in data_frame


# In[ ]:


pd.read_csv('file_name.csv', usecols = ['A'],squeeze = True)


# #### 9.SkipRows/nrows Parameter

# In[ ]:


You can skip the particular row


# In[ ]:


df=pd.read_csv('file name.csv',skiprows = [0,5])


# In[ ]:


# lambda use to remove the row


# In[ ]:


df=pd.read_csv('file name.csv',nrows = 100) #It will print only 100 rows


# #### 10. Encoding Parameter

# In[ ]:


pd..read('zonato.csv')


# In[8]:


# It will work when you face the error like UnicodeDecodeError , now use latin-1


# In[ ]:


pd..read('zonato.csv', 'latin-1')


# ### 11. Skip bad lines

# In[ ]:


pd.read_csv(BX-Books.csv, sep = ';', encoding = 'latin-1')


# In[ ]:


This will work when you there are some rows with 6 cols instead of 5
all the with 5 cols but 1 or more cols have 6 cols that is the Error.

You will see ParserError
Expected 8 fields in line 542, saw 9
use, error_bad_lines = False


# In[ ]:


pd.read_csv(BX-Books.csv, sep = ';', encoding = 'latin-1', error_bad_lines = False)


# ### 12. dtypes Parameter

# In[ ]:


Change the Data Type of any object


# In[38]:


pd.read_csv('aug_train.csv', dtype = {'Col_name':int or float})


# ### 13. Handiling Date

# In[ ]:


Change the column from Object type to DateTime


# In[58]:


pd.read_csv('IPL Matches 2008-2020.csv')


# In[57]:


pd.read_csv('IPL Matches 2008-2020.csv',parse_dates=['date']).info()


# ## 14. Convertors

# In[59]:


def rename(name):
       if name == "Royal Challengers Bangalore":
           return "RCB"
       else:
           return name


# In[60]:


rename("Royal Challengers Bangalore")


# In[61]:


pd.read_csv("IPL Matches 2008-2020.csv")


# In[63]:


pd.read_csv("IPL Matches 2008-2020.csv", converters = {'team1':rename})


# ### 15. na_values_parameter

# In[ ]:


16. 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[16]:





# In[ ]:




