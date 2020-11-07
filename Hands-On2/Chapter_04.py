#!/usr/bin/env python
# coding: utf-8

# # Chapter 4: Training models and Mathematics
# The iris database

from sklearn import datasets
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import numpy as np


iris = datasets.load_iris()

list(iris.keys())

X = (iris["data"][:,3:]) # Petal width
y = (iris["target"] == 2).astype(np.int) # 1 if iris verginica, else 0

# A description of the iris dataset  is stored in this magic variable.
print (iris.DESCR)

# Create a logistic regression model, and fit it.
log_reg = LogisticRegression()
# Note that I am fitting the full dataset, I haven't set aside a test
# set.  This is a bad idea
log_reg.fit(X, y)

print (iris.data.shape)



# Let's plot the predicted values

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
plt.plot(X_new, y_proba[:, 1], 'r-', label="Iris virginica")
plt.plot(X_new, y_proba[:, 0], 'b--', label="Not Iris virginica")
plt.legend()
plt.grid()


# Let's try to fit a regression model with two variables

X = (iris["data"][:,2:]) # Petal length and width
y = (iris["target"] == 2).astype(np.int) # 1 if iris verginica, else 0

log_reg.fit(X, y)


# In[68]:


log_reg.predict([[4.5, 1.7]])


# That predicted this is not iris virginica (0)
# 

# # Softmax regression
# 
# Logistic regression is capable of predicting multiple classes by calculating a score for each class, and a corresponding probability for each class.
# 
# The cost function has a penalty for not predicting the target class.

# In[69]:


# This is how you do a softmax regression
X = iris["data"][:, (2,3)] # petal length, petal width
y = iris["target"]

# The C parameter is the opposite of the alpha parameter for regularization.
# Low C means more regularization, high C means less regularization.
# TODO: find its range
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
softmax_reg.fit(X,y)


# In[70]:


softmax_reg.predict([[5,2]])


softmax_reg.predict_proba([[5,2]])




