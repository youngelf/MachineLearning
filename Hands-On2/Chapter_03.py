#!/usr/bin/env python
# coding: utf-8

# # Chapter 3: Classification
# 
# Getting the MNIST dataset. Sigh. I'm tired of this dataset.
# 
# 

# In[41]:


from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

# All imports and creating the dataset

# Download the MNIST data if it isn't available. Cached in ~/scikit_learn_data
mnist = fetch_openml('mnist_784', version=1, cache=True)


# In[42]:


# The format is quirky: there is the actual dataframe, but also some metadata.
mnist.keys()

type(mnist)


# In[43]:


print ("Stopping early, still work left to do")
sys.exit(0)

# Assign attributes and labels using standard terminology
X, y = mnist["data"], mnist["target"]

print (X.shape)
print (y.shape)


# # Visually inspecting the data 
# 
# These are digits, and you can use matplotlib to see the data

# In[44]:


# You can find what the data is like by reading the description
mnist['DESCR']


# In[45]:


import matplotlib as mpl
import matplotlib.pyplot as plt

def showImage(digit):
    # The description tells you that these are 28x28 pixel boxes
    digit_image = digit.reshape(28,28)
    plt.imshow(digit_image, cmap='binary')
    plt.axis("off")
    plt.show()

showImage(X[42])


# In[46]:


# And what does that value correspond to? Let's look at the label
print(y[42])
print(type(y[42]))
print (y[42] == 7)


# In[47]:


# But y itself is strings, so we need to recast them as integers
import numpy as np
y = y.astype(np.uint8)


# In[48]:


print(type(y))


# In[49]:


print(type(y[42]))
print (y[42] == 7)


def showImageAndLabel(index):
    showImage(X[index])
    print ("The label for the image above: ", y[index])

showImageAndLabel(32)
# This is one quirky 5!
showImageAndLabel(1032)


# In[ ]:





# # Training and test data split
# 
# The very first step should always be splitting training and test data. So we do that. Luckily the mnist data is
# already split as 60000 values of training data, and 10000 values of training data in a stratified way (by digits)
# 
# Let's use this splitting.
# 
# And let's start out using a binary classifier, that detects a single label: '5' to start out.

# In[50]:


from sklearn.linear_model import SGDClassifier

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Creates boolean arrays that have true when 5 and false otherwise.
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_classifier = SGDClassifier(random_state=42)
sgd_classifier.fit(X_train, y_train_5)


# In[51]:


# Let's give this our difficult 5
print(sgd_classifier.predict([X[1032]]))

# The first element is a much simpler 5. Let's test that too
print(sgd_classifier.predict([X[0]]))


# # Long computations
# 
# When we have time, let's run these long computations to see if we can improve the quality of the SGD classification

# In[52]:


sgd_big = SGDClassifier(random_state=42, loss='perceptron')
sgd_big.fit(X_train, y_train_5)

estimator=sgd_big
print("SGD with perceptron")
# Let's give this our difficult 5
print(estimator.predict([X[1032]]))

# The first element is a much simpler 5. Let's test that too
print(estimator.predict([X[0]]))


sgd_a = SGDClassifier(random_state=42, n_iter_no_change=40)
sgd_a.fit(X_train, y_train_5)

estimator=sgd_a
print("SGD with big iterations without change")
# Let's give this our difficult 5
print(estimator.predict([X[1032]]))

# The first element is a much simpler 5. Let's test that too
print(estimator.predict([X[0]]))


# In[53]:


sgd_b = SGDClassifier(random_state=42, learning_rate='adaptive', eta0=4)
sgd_b.fit(X_train, y_train_5)

estimator=sgd_b
print("SGD with adaptive learning")
# Let's give this our difficult 5
print(estimator.predict([X[1032]]))

# The first element is a much simpler 5. Let's test that too
print(estimator.predict([X[0]]))

sgd_c = SGDClassifier(random_state=42, tol=1e-5)
sgd_c.fit(X_train, y_train_5)

estimator=sgd_c
print("SGD with much lower tolerance")
# Let's give this our difficult 5
print(estimator.predict([X[1032]]))

# The first element is a much simpler 5. Let's test that too
print(estimator.predict([X[0]]))

# And we'll keep these different estimators ready for later comparisons of f1 score, precision/recall.


# # Precision / Recall of classification
# 
# It got the difficult 5 incorrect, but got the easy 5 correct. Even Stochastic Gradient Descend seems to work well. How well? Let's look at the ROC curve, recall and precision

# In[54]:


from sklearn.model_selection import cross_val_score

cross_val_score(sgd_classifier, X_train, y_train_5, cv=3, scoring='accuracy')

# But these numbers are misleading because only 10% values are the digit '5'. So we need better metrics for quality
# of our classifier


# In[55]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(sgd_classifier, X_train, y_train_5, cv=3)

print(confusion_matrix(y_train_5, y_train_pred))


# ## Definitions
# 
# Precision:
# $$ \frac{TP}{TP+FP} $$
# Recall:
# 
# $$ \frac{TP}{TP+FN} $$

# In[56]:


from sklearn.metrics import precision_score, recall_score

cm = confusion_matrix(y_train_5, y_train_pred)

precision = (cm[1][1]/(cm[1][1]+cm[0][1]))
recall = (cm[1][1]/(cm[1][1]+cm[1][0]))

print(precision)
print(recall)


# In[57]:


from sklearn.metrics import precision_score, recall_score

p=precision_score(y_train_5, y_train_pred)
r=recall_score(y_train_5, y_train_pred)
print(p)
print(r)


# In[58]:


from sklearn.metrics import f1_score

# The f1 score is the harmonic mean of precision and recall
f1_score(y_train_5, y_train_pred)


# In[59]:


# Let's verify that it is what we think it is
f1 = 2/((1/p)+(1/r))

print(f1)


# # Precision-Recall tradeoff
# 
# You can't have it all. High precision means that you label fewer things as '5', thus hurting recall as some elements that are truly 5 get left behind.

# In[60]:


from sklearn.metrics import precision_recall_curve

y_scores = cross_val_predict(sgd_classifier, X_train, y_train_5, cv=3,
                            method='decision_function')

precision_vals, recall_vals, threshold_vals = precision_recall_curve(y_train_5, y_scores)

def plot_precision_recall(p, r, t):
    plt.plot(t, p[:-1], "b--", label="Precision")
    plt.plot(t, r[:-1], "g-", label="Recall")
    
plot_precision_recall(precision_vals, recall_vals, threshold_vals)
plt.show()


# Let's evaluate the other models as well and see what their performance looks like

# In[61]:


def evaluate_model(classifier):
    y_scores = cross_val_predict(classifier, X_train, y_train_5, cv=3,
                                method='decision_function')
    pv, rv, tv = precision_recall_curve(y_train_5, y_scores)
    plot_precision_recall(pv, rv, tv)
    plt.show()

evaluate_model(sgd_big)

evaluate_model(sgd_a)

evaluate_model(sgd_b)

evaluate_model(sgd_c)


# In[62]:


# Plot precision versus recall graph

def plot_precision_vs_recall(p, r):
    fig,ax = plt.subplots()
    ax.plot(r[:-1], p[:-1], "b--")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision versus Recall")
    ax.grid()
    # To save the figure, do this
    fig.savefig("test.png")
    plt.show()
    
plot_precision_vs_recall(precision_vals, recall_vals)


# # ROC curve
# 
# Receiver Operating Characteristic (ROC) curve lists out true-positive rate (recall) versus false-positive rate. I need to get a comfortable feeling about this, since it is a central concept.

# In[63]:


from sklearn.metrics import roc_curve

# fpr: False Positive Rate, tpr: True Positive Rate
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, 'b--', label=label)
    plt.ylabel('True Positive Rate(Recall)')
    plt.xlabel('False Positive Rate')
    plt.suptitle('ROC curve')
    plt.grid()
    
    plt.legend(loc='lower right')
    plt.show()

plot_roc_curve(fpr, tpr, "SGD")


# In[64]:


# The area under the curve is very useful to evaluate this graph. Larger is better. 1 is the best-possible

from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)


# # Random forest classifier
# 
# These do much better on this dataset.
# 

# In[65]:


from sklearn.ensemble import RandomForestClassifier

forest_classifier = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_classifier, X_train, y_train_5, cv=3,
                                   method='predict_proba')

# score = proba of positive class
y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)


plot_roc_curve(fpr_forest, tpr_forest, 'RandomForest')


# In[66]:


plt.plot(fpr, tpr, 'r:', label='SGD')
plot_roc_curve(fpr_forest, tpr_forest, label='Random Forest')


# In[67]:


# How good is this? Let's try the difficult case again
forest_classifier.fit(X_train, y_train_5)
out_label = forest_classifier.predict([X[1032]])

print(out_label)


# In[ ]:





# # Multiclass classification
# 
# 

# In[68]:


from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)

print(svm_classifier.predict([X[1032]]))

# And look at the scores for that input
print(svm_classifier.decision_function([X[1032]]))


# In[69]:


# And train a one versus rest classifier here.

ovr_classifier = OneVsRestClassifier(SVC())
ovr_classifier.fit(X_train, y_train)


# In[70]:


ovr_classifier


# In[71]:


# Let's see what the svm_classifier chooses for the easy case:
print(svm_classifier.decision_function([X[32]]))

showImageAndLabel(32)


# In[72]:


print(svm_classifier.decision_function([X[1032]]))
showImageAndLabel(1032)


# In[73]:


ovr_classifier.predict([X[1032]])


# In[74]:


# Let's train an SGD classifier on scaled input

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

sgd_number = SGDClassifier(random_state=42)
# Now we do a full fitting, not just _5, so we get the actual value
sgd_number.fit(X_train_scaled, y_train)


# In[75]:



print(sgd_number.predict([X[1032]]))
print (sgd_number.predict([X[32]]))


# # Error Analysis
# Let's look at the confusion matrix with the true values removed.
# 
# 

# In[76]:


# Get the errors for the ovr classifier, and we can get SGD with scaling later

y_train_pred = cross_val_predict(ovr_classifier, X_train, y_train, cv=3)
confusion_mx = confusion_matrix(y_train, y_train_pred)


# In[77]:


# Print this out when it is computed in the morning.
plt.matshow(confusion_mx, cmap=plt.cm.gray)
plt.show()


# In[78]:


# Let's look just at the errors

# axis=1 is column sum. So we are choosing to normalize them by predicted classes.
row_sums = confusion_mx.sum(axis=1, keepdims=True)
norm_sum_mx = confusion_mx / row_sums

# Delete the diagonal to focus on the errors
np.fill_diagonal(norm_sum_mx, 0)


# In[79]:


# In morning, run this

# And now plot just the errors
plt.matshow(norm_sum_mx, cmap=plt.cm.gray)
plt.show()


# In[80]:


p=norm_sum_mx*10000
p.astype(np.int32)


# # Exercises
# 
# Most important to do the exercises, especially when waiting for the kernel in the next chapter to finish.
# 
# This is where the real learning happens.
# 

# In[81]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn import decomposition
from sklearn import datasets

np.random.seed(5)

centers = [[1, 1], [-1, -1], [1, -1]]
iris = datasets.load_iris()
X = iris.data
y = iris.target

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)

for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(X[y == label, 0].mean(),
              X[y == label, 1].mean() + 1.5,
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
           edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()


# In[ ]:




