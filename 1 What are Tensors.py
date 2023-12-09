#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[ ]:


#reating a scaler with the help of numpy


# ### SCALERS

# In[2]:


a=np.array(4)


# In[3]:


a  # 4 is scaler or 0 dimentional || 0-D tensor


# ### 1D Tensor/ Matrices

# In[ ]:


# Tensor and n-dimnention are the same thing.
# 1-Dim array mean Array
# 2-Dim array mean Array in array
# 3-Dim mean array in array in array


# In[ ]:


# 1-D Tensor
# Vector
# 1-D array axis=1 or rank or dimention is 1 


# In[4]:


# Creating 1-D array or 1-D Tensor for numpy

arr = np.array([1,2,3,4])


# In[5]:


# Also called the vector
# dimention for Vector 
# It is 1-D tensor and it is a Vector and it has 4 Dimention becs it has 4 numbers means it has 4-D
# 1-D tensor always a Vector
arr 


# In[6]:


arr.ndim


# In[ ]:


[1,2]

# 1-D tensor and vector
# vector has 2 Dimention

[1,2,3]

# 1-D tensor and vector
# vector has 3 Dimention


# ### 2D Tensor/Matrices

# In[8]:


# Matrices are the collection of Vectors


# In[9]:


import numpy as np


# In[10]:


arr=np.array([[1,2,3],[4,5,6],[7,8,9]])


# In[11]:


arr  # Matrices 2D Row and Column


# In[12]:


arr.ndim


# ### N-D Tensors

# In[13]:


# Cube has 3 axises x,y,z
# 3-D Tensor


# ### Rank, Axes and Shape

# In[ ]:


Rank=no. of Axis = No. of Dimentions
Shape => Rows and Columns (2.3), (3,3)
Size => The total number of Columns in matrice (3,3)=3x3=9


# #### 1D Tensor/Vector

# In[ ]:


Studens (10000)

CGPA | IQ | State | Placement

8.1     91    WB        1


# #### 2D Tensor

# In[ ]:


If you have the data of 10000 studens and each student data is vector
like this:
    [CGPA, IQ, State, Board,College, Placement]
collection of 10000 Vectors whould be Matrice
You can store all the stident data and it is called matrices (2D)


# #### 3D Tensor

# In[ ]:


Practical example is NLP (Natiral Language Processing)


# In[ ]:


If you are working with Textual Data there is 3D Tensons


# In[ ]:


Eg:
    Hello World
    Hello Machile Learning
    Good morning

Machine Learning Algorithem is purely mathematics and It doesnt understant the test and Strig
You will have to convert this test into numbers or Vectors its called vectarization Technique.


# In[ ]:


eg:
    Time Series Data
Highest price | Lowest Price



# #### 4D Tensor

# In[ ]:


Eg: Images , Images are collection of pixels and every pixel has a numerical value
    R | G | B
    Red | Green | Blue
    


# # 5D Tensor

# In[ ]:


Eg:
    Vedios
    fps - Frame per Second
    
60 sec of vedio
30 fps (shoot)
   1800 images in 60 sec
480x720 pixels 
3 Channel

for a single image = (480x720x3)

this is a vedio = (1800x480x720x3)  4D Tensor

if you have 4 vedios (collection of Vedios) 5D

this is a 5D = 4x(1800x480x720x3)  5D Tensor


# In[ ]:





# In[ ]:





# In[ ]:




