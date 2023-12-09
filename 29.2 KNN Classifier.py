#!/usr/bin/env python
# coding: utf-8

# ### 1. Decision Boundary for KNN

# #### 1. What is Decision Boundary
# 
# In a classification problem with two or more classes. a decision boundary or decision surface is a 
# hypersurface that partitions the underlying vector space into two or more sets.One for each class.
# The classifier will classify all the points on one side of the decision boundary as belonging to one
# class and all those on the other side as belonging to the other class.

# In[ ]:





# In[5]:


from IPython.display import Image
Image(filename='Image 1.png')


# ### Imoport points
# 
# 1. we can draw decision boundary for all the classification algorithems including Neural Networks.
# 2. Decision boundary can be both linear(as in the case of SVM) or non-linear(as in the case of 
#    Decision tree classifier or KNN).
# 3. Decision boundary are not always clear cut. That is the transition from one class in the feature
# space to another is not discontinuous, but gradual. This effect is common in fuzzy logic 
# based classification
# algorithems. Where membership in one class or another is ambiguous.
# 4. for higher dimension problems the decision boundary acts as a hyperplane (for linear ones).
# 
# SVM support vector Machine

# ### 2. Vornoi diagram

# In mathematics, a Voronol diagram is partitioning of a plane into regions based on distance to points
# in a speceific subset of the plane.

# In[7]:


from IPython.display import Image
Image(filename='Image 2.png')


# #### 3. Steps to plot Decision Boundary for KNN(assuming 2 in in input cols)

# 1. Train the classifier on the training set.
# 2. Create a uniform grid(with the help of numpy Meshgrid) of points that densely cover the region
# of input space containing the training set.
# 3. Classify each poit on the grid. Store the results in an array A. where Aij contain the predicted
#    class for the point at row i. column j on the grid.
# 4. Plot the array as an image. where each pixel corresponds to a grid point and its color represents
# the predicted class. The decision boundary can be seen as contours where the image changes color.
# 5. Finally print out training data with their respeective color on the same contour.

# ### Meshgrids

# In[9]:


from IPython.display import Image
Image(filename='Image 3.png')


# In[10]:


from IPython.display import Image
Image(filename='Image 4.png')


# ### Creating a sample meshgrid

# In[1]:


import numpy as np
x=np.array([1,2,3])
y=np.array([4,5,6,7])

xx,yy= np.meshgrid(x,y)


# In[2]:


xx.shape


# In[3]:


yy.shape


# ### Purpose of Meshgrid

# Meshgrid is very useful to evaluate fucntion on a grid. We can apply any function to the points of a meshgrid to plot a fucntion

# #### Ploting a fucntion using Meshgrid

# In[4]:


x= np.linspace(-40,40,1000)
y=np.linspace(-50,50,900)
xx,yy = np.meshgrid(x,y)


# In[5]:


xx.shape


# In[6]:


yy.shape


# In[7]:


z= np.random.random((900,1000))


# In[10]:


plt.contourf(xx,yy,z)


# In[9]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns


# In[11]:


df = pd.read_csv("placement (1).csv")


# In[12]:


df


# In[13]:


df.isnull().sum()


# In[14]:


# Your data and labels
X, y = df.drop("placed", axis=1), df["placed"]


# Split the data into a training set and a test set (e.g., 80% training and 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[19]:


X_train.shape


# In[16]:


y_train.shape


# In[21]:


# 1st method to find out the k
# sqrt of number of rows in training dataset
np.sqrt(X_train.shape[0])


# In[22]:


k = 28


# In[24]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = k)


# In[25]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[26]:


# Fit the scaler to your data and transform the data point
x_train_scale = scaler.fit_transform(X_train)

# The standardized_data_point array now contains your standardized data point
print(x_train_scale)


# ### Step 2. Creating a Meshgrid

# In[27]:


x_train_scale.shape


# In[28]:


x_test_scale = scaler.transform(X_test)


# In[29]:


x_test_scale.shape


# In[30]:


print(x_train_scale.min())
print(x_train_scale.max())


# In[31]:


a = np.arange(start = x_train_scale[:,0].min()-1, stop = x_train_scale[:,0].max()+1 , step = 0.01)
b = np.arange(start = x_train_scale[:,1].min()-1, stop = x_train_scale[:,1].max()+1 , step = 0.01)


# In[32]:


a.min()


# In[33]:


a.max()


# In[34]:


b.shape, a.shape


# In[35]:


shape = 697*744
print(shape)


# In[36]:


xx,yy = np.meshgrid(a,b)


# In[37]:


xx.shape, yy.shape


# ### Step 3. Classifying every point on the meshgrid

# In[38]:


Description by vedio


# In[39]:


print(xx[0][[0]])
print(yy[0][[0]])


# In[41]:


knn.predict(np.array([-4.3872027,-2.66916138]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




