#!/usr/bin/env python
# coding: utf-8

# # Chapter 5: Support Vector machines
# 
# Support Vector is an interesting beast. It relies on the fact that to separate two areas, most of the decision is made by a small set of values, these are called the support vectors. These are the boundary values. Much like the boundary between India and China is not governed by the location of Bombay, but instead on the specifics of Arunachal Pradesh and Kashmir.
# 
# There are a few ways of training Support Vector classifiers. SGD will do it, and has the advantage of looking at a subset of the values at a time.  LinearSVC seems to use all of the data at the same time, rather than the SGD mini-batch approach. Finally SVC will do this, and can apply a polynomial or string kernel to generalize the model to polynomial or strings, or other specialized attributes.

# In[15]:


import numpy as np

from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# In[16]:


iris = datasets.load_iris()
X = iris["data"][:, (2,3)] # petal length, petal width
y = (iris["target"] == 2).astype(np.float64) # Iris Virginica

svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('linear_svc', LinearSVC(C=1, loss="hinge")),
])

svm_clf.fit(X, y)


# In[17]:
print ("Stopping early, still work left to do")
sys.exit(0)



svm_clf.predict([[5.5, 1.7]])


# # The moons dataset, and the "kernel" trick

# In[18]:


from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# Create a random dataset with "moons" that you have to separate
X, y = make_moons(n_samples=1000, noise=0.15)

polynomial_svm_clf = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3)),
    ('scaler', StandardScaler()),
    ('svm_clf', LinearSVC(C=10, loss="hinge", max_iter=10000))
])

polynomial_svm_clf.fit(X, y)


# In[19]:


print(X[:10])
print(y[:10])


# In[20]:


index=(y==1)
print(index[0:10])
print (X[index][:10, 0:1])


# In[21]:


X[:, 0]


# In[22]:


# Plot this data
import matplotlib.pyplot as plt

index_c0 = (y == 0)
index_c1 = (y == 1)

plt.figure(figsize=(15, 10))

plt.plot(X[index_c0][:,0], X[index_c0][:,1], 'go', label='Class 1')
plt.plot(X[index_c1][:,0], X[index_c1][:,1], 'rv', label='Class 0')
plt.legend()


# In[25]:


# Using the notebook from homl.info

def plot_dataset(X, y, axes):
    "Plot the Moons dataset with two classes: 0 and 1"
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)


def plot_predictions(clf, axes):
    "Visualize what the classifier will choose on this dataset"
    x0s = np.linspace(axes[0], axes[1], 1000)
    x1s = np.linspace(axes[2], axes[3], 1000)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)

# Plot the data
plt.figure(figsize=(15, 10))
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.show()

# Plot the predictions and the classifier
plt.figure(figsize=(15, 10))
plot_predictions(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.show()


# In[28]:


# Try a different classifier

p1 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3)),
    ('scaler', StandardScaler()),
    ('svm_clf', LinearSVC(C=1, loss="hinge", max_iter=10000))
])

p1.fit(X, y)


# In[29]:


# Plot the predictions and the classifier
plt.figure(figsize=(15, 10))
plot_predictions(p1, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.show()


# In[34]:


# Which one has lesser ROC scores?
from sklearn.metrics import roc_curve, roc_auc_score

def roc(clf, X, y, label):
    "Plot ROC curve for this classifier"
    
    y_pred = clf.predict(X)
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    print("Area Under Curve for ", label, " is ", roc_auc_score(y, y_pred))
    
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.show()
    
roc(p1, X, y, "C=1, SVM polynomial")


# In[35]:


roc(polynomial_svm_clf, X, y, "C=10, SVM polynomial")


# In[32]:


get_ipython().run_line_magic('pinfo', 'SVC')


# In[33]:


from sklearn.svm import SVC
get_ipython().run_line_magic('pinfo', 'SVC')


# In[36]:


# Let's use the kernel trick to create a SVC with Radial Basis Function (RBF) kernel

rbf_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_classifier', SVC(kernel="rbf", gamma=5, C=0.001))
])

rbf_svm.fit(X, y)


# In[37]:



roc(rbf_svm, X, y, "C=0.001, SVM with RBF kernel")


# Interesting, it is worse than with polynomial (3rd degree) features. Let's try with a polynomial kernel instead
# 

# In[38]:


get_ipython().run_line_magic('pinfo', 'SVC')


# In[43]:


# Polynomial kernel

poly_kernel_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_classifier', SVC(kernel="poly", degree=3, gamma=50, C=1))
])

poly_kernel_svm.fit(X, y)


# In[44]:



roc(poly_kernel_svm, X, y, "C=0.001, SVM with Polynomial (3rd degree) kernel")


# In[ ]:




