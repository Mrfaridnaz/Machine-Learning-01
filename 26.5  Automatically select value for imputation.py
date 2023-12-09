#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression


# In[4]:


df = pd.read_csv("titanic_train.csv")


# In[5]:


df.head()


# In[6]:


df.drop(columns = ['PassengerId','Name','Ticket','Cabin'], inplace = True)


# In[7]:


df.head()


# In[8]:


x= df.drop(columns = ['Survived'])
y = df["Survived"]


# In[9]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 2)


# In[10]:


x_train.head()


# In[ ]:


'Age' and 'Fare' numerical cols
apply imputation for missing values and than apply StandardScaling


# In[ ]:


'Sex' and 'Embarked' both are categorical cols
apply imputationn and than apply One Hot Encoding

after that apply the Logisticregression


# In[18]:


numerical_features = ['Age', 'Fare']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['Embarked', 'Sex']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])




# In[19]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


# In[21]:


clf = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])


# In[22]:


from sklearn import set_config

set_config(display = 'diagram')
clf


# In[42]:


from sklearn.model_selection import GridSearchCV

# Define your parameter grid
param_grid = {
    'preprocessor_num_imputer_strategy': ['mean', 'median'],
    'preprocessor_cat_imputer_strategy': ['most_frequent', 'constant'],
    'classifier_C': [0.1, 1.0, 10, 100]
}

# Create a GridSearchCV object
grid_search = GridSearchCV(clf, param_grid, cv=10)

# Fit the GridSearchCV object to your data
grid_search.fit(x_train, y_train)

print(f"Best params:")
print(grid_search.best_params_)


# In[43]:


print(f"Internal CV score : {grid_search.best_score_:3f}")


# In[44]:


import pandas as pd


# In[ ]:


Incomplete


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




