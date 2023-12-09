#!/usr/bin/env python
# coding: utf-8

# In[ ]:


1. MAE (Mean Absolute Error)
2. MSE (Mean Squared Error)
3. RMSE (Root Mean Square Error)
These are called loss funtion or Error Function.
4. R2 Score
5. Adjustd R2 Score


# ### 1. MAE (Mean Absolute Error)

# In[ ]:


If you are working in Simple linear regression, where one input and one output.


# In[ ]:


Total absolute error = |y1-y1>| + |y2-y2>| + |y3-y3>| + ... + |yn-yn>|


# In[ ]:


MAE =(|y1-y1>| + |y2-y2>| + |y3-y3>| + ... + |yn-yn>|)/n


# In[ ]:


MAE = SUM|yi-yi>|/n


# In[ ]:


Advantages
1. Same unit
2. Robust to outliers mean handle the outliers.

Disadvantages:
1. Modulous graph is not diffrenciable.


# ### 2. MSE (Mean Squared Error)

# In[ ]:


MSE = SUM(yi-yi>)*2/n


# In[ ]:


Advantages:
1. Diffrencialble

Disadvantage:
1. Not robust to outliers    


# ### 3. RMSE (Root Mean Square Error)

# In[ ]:


RMSE = root of MSE SUM(yi-yi>)*2/n


# In[ ]:


Advantages:
1. Diffrencialble

Disadvantage:
1. Not robust to outliers 


# ### 4. R2 Score

# In[ ]:


It will tell how well the model is performing.


# In[ ]:


R2 Score also known as Coeficient of Determination or Goodness of fit

formuls = 1 - SSR/SSm

SSR = Sum of square Error in the Regression line
SSm = Sum of squre Error in the mean line


# In[ ]:





# In[ ]:





# In[ ]:




