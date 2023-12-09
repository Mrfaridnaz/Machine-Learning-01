#!/usr/bin/env python
# coding: utf-8

# # K Nearest neighbors

# In[ ]:


Topics
1. Intoduction and Geometric intuition.
2. Working with a real world data-set.
3. Building a KNN classifier from scracth.
4. Building and Deploying the ML model.
5. Advance topics related to KNN.


# In[ ]:


Assumptions
1. KNN assumes data is in metrics space and there is a notion of distance.
2. Each of the training data consist of a label data associated with it, either + or - 
   although KNN also supports multiclass classification.
3. We are also  given a sinlge number "K". This number decides how many neighbors influence the
   classification. This is usually a ODD number.


# ### Geometric intuition

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("Social_Network_Ads.csv")


# In[3]:


data.head()


# In[4]:


data.iloc[:, 2:4] 


# In[5]:


# Convert it into numpy Array

x= data.iloc[:, 2:4].values


# In[6]:


x.shape


# In[7]:


y = data.iloc[:,-1].values


# In[12]:


y.shape


# In[8]:


from sklearn.model_selection import train_test_split
x_train , x_test , y_train, y_test = train_test_split(x,y, test_size = 0.20)


# In[9]:


x_train.shape


# In[10]:


x_test.shape


# In[ ]:


"Age" and "EstimatedSalary" out of range to each other
diff to find out the distance between the points.


# In[11]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[12]:


x_train = scaler.fit_transform(x_train)
x_train

# now both the cols in the range


# In[13]:


x_test = scaler.transform(x_test)
x_test


# In[14]:


# 1st method to find out the k
# sqrt of number of rows in training dataset
np.sqrt(x_train.shape[0])


# In[15]:


k = 17


# In[17]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = k)


# In[18]:


# Train our model
knn.fit(x_train, y_train)


# In[20]:


y_pred = knn.predict(x_test)


# In[21]:


y_pred


# In[35]:


y_pred.shape


# In[36]:


y_test.shape


# In[22]:


y_test


# In[26]:


# compare our y_pred with y_test to check how the algorethem was accurate

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[27]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[28]:


# 2nd method

accuracy = []

for i in range(1,26):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(x_train, y_train)
    accuracy.append(accuracy_score(y_test, knn.predict(x_test)))


# In[29]:


len(accuracy)


# In[30]:


plt.plot(range(1,26),accuracy)


# In[31]:


knn = KNeighborsClassifier(n_neighbors = 11)
knn.fit(x_train, y_train)


# In[32]:


y_pred = knn.predict(x_test)


# In[33]:


accuracy_score(y_test, y_pred)


# In[34]:


# we have created the classifier and checked our accurac score


# In[35]:


# Create a function that show the output when you provide the input

def predict_output():
    age = int(input("Entyer the age"))
    salary = int(input("Enter the salary"))
    
    return np.array([[age], [salary]])


# In[ ]:


predict_output()


# In[ ]:


def predict_output():
    age = int(input("Entyer the age"))
    salary = int(input("Enter the salary"))
    
    x_new = np.array([[age], [salary]]).reshape(1,2)
    x_new = scaler.transform(x_new)
    
    return knn.predict(x_new)


# In[ ]:


predict_output()


# In[58]:


def predict_output():
    age = int(input("Entyer the age"))
    salary = int(input("Enter the salary"))
    
    x_new = np.array([[age], [salary]]).reshape(1,2)
    x_new = scaler.transform(x_new)
    
    return knn.predict(x_new)[0]


# In[59]:


predict_output()


# In[60]:


def predict_output():
    age = int(input("Entyer the age"))
    salary = int(input("Enter the salary"))
    
    x_new = np.array([[age], [salary]]).reshape(1,2)
    x_new = scaler.transform(x_new)
    
    if knn.predict(x_new)[0] == 0:
        return "Will not purchase"
    else:
        return "Will Purchase"


# In[61]:


predict_output()


# ### Few Observations

# In[ ]:


# 1. Hyperparameter

# 2. Method to choose k
     method 1 k=17
     method 2 k=11
        there is no set rule to use k value, use both methods and check in which case the 
        algorithem is performimng better and then decide accordingly.


# In[ ]:





# In[ ]:





# In[2]:


### 1. Decision Boundary for KNN


# In[3]:


#### 1. What is Decision Boundary

In a classification problem with two or more classes. a decision boundary or decision surface is a 
hypersurface that partitions the underlying vector space into two or more sets.One for each class.
The classifier will classify all the points on one side of the decision boundary as belonging to one
class and all those on the other side as belonging to the other class.


# In[1]:


from IPython.display import Image
Image(filename='Image 1.png')


# In[ ]:


### Imoport points

1. we can draw decision boundary for all the classification algorithems including Neural Networks.
2. Decision boundary can be both linear(as in the case of SVM) or non-linear(as in the case of 
   Decision tree classifier or KNN).
3. Decision boundary are not always clear cut. That is the transition from one class in the feature
space to another is not discontinuous, but gradual. This effect is common in fuzzy logic 
based classification
algorithems. Where membership in one class or another is ambiguous.
4. for higher dimension problems the decision boundary acts as a hyperplane (for linear ones).

SVM support vector Machine


# In[ ]:


### 2. Vornoi diagram


# In[ ]:


In mathematics, a Voronol diagram is partitioning of a plane into regions based on distance to points
in a speceific subset of the plane.


# In[4]:


from IPython.display import Image
Image(filename='Image 2.png')


# In[ ]:


#### 3. Steps to plot Decision Boundary for KNN(assuming 2 in in input cols)


# In[ ]:


1. Train the classifier on the training set.
2. Create a uniform grid(with the help of numpy Meshgrid) of points that densely cover the region
of input space containing the training set.
3. Classify each poit on the grid. Store the results in an array A. where Aij contain the predicted
   class for the point at row i. column j on the grid.
4. Plot the array as an image. where each pixel corresponds to a grid point and its color represents
the predicted class. The decision boundary can be seen as contours where the image changes color.
5. Finally print out training data with their respeective color on the same contour.


# In[ ]:


### Meshgrids


# In[5]:


from IPython.display import Image
Image(filename='Image 3.png')


# In[6]:


from IPython.display import Image
Image(filename='Image 4.png')


# In[ ]:


### Creating a sample meshgrid


# In[7]:


import numpy as np
x=np.array([1,2,3])
y=np.array([4,5,6,7])

xx,yy= np.meshgrid(x,y)


# In[8]:


xx.shape


# In[ ]:


yy.shape


# In[ ]:


### Purpose of Meshgrid


# In[ ]:


Meshgrid is very useful to evaluate fucntion on a grid. We can apply any function to the points of a meshgrid to plot a fucntion


# In[ ]:


#### Ploting a fucntion using Meshgrid


# In[ ]:


x= np.linspace(-40,40,1000)
y=np.linspace(-50,50,900)
xx,yy = np.meshgrid(x,y)


# In[ ]:


xx.shape


# In[ ]:


yy.shape


# In[ ]:


z= np.random.random((900,1000))


# In[ ]:


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


# In[ ]:


df = pd.read_csv("placement (1).csv")


# In[ ]:


df


# In[ ]:


df.isnull().sum()


# In[ ]:


# Your data and labels
X, y = df.drop("placed", axis=1), df["placed"]


# Split the data into a training set and a test set (e.g., 80% training and 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


X_train.shape


# In[ ]:


y_train.shape


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[ ]:


# Fit the scaler to your data and transform the data point
x_train_scale = scaler.fit_transform(X_train)

# The standardized_data_point array now contains your standardized data point
print(x_train_scale)


# In[ ]:


### Step 2. Creating a Meshgrid


# In[ ]:


x_train_scale.shape


# In[ ]:


x_test_scale = scaler.transform(X_test)


# In[ ]:


x_test_scale.shape


# In[ ]:


print(x_train_scale.min())
print(x_train_scale.max())


# In[ ]:


a = np.arange(start = x_train_scale[:,0].min()-1, stop = x_train_scale[:,0].max()+1 , step = 0.01)
b = np.arange(start = x_train_scale[:,1].min()-1, stop = x_train_scale[:,1].max()+1 , step = 0.01)


# In[ ]:


a.min()


# In[ ]:


a.max()


# In[ ]:


b.shape, a.shape


# In[ ]:


shape = 697*744
print(shape)


# In[ ]:


xx,yy = np.meshgrid(a,b)


# In[ ]:


xx.shape, yy.shape


# In[ ]:


### Step 3. Classifying every point on the meshgrid


# In[ ]:


Description by vedio


# In[ ]:


print(xx[0][[0]])
print(yy[0][[0]])


# In[ ]:


knn.predict(np.array([-3.28732165,-2.46211756]))


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




