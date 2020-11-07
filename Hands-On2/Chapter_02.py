#!/usr/bin/env python
# coding: utf-8

# In[32]:


# %reset 

# Import all the libraries here
import os, tarfile, urllib
import numpy as np
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt

print ("Loading constants")

# Constants are defined here

# This is where we'll get it.
HOUSING_PATH = os.path.join("datasets", "housing")


# In[ ]:


print ("Stopping early, still work left to do")
sys.exit(0)




# # Downloading the data
# 
# By convention, $m$ is the number of data instances
# 
# This following method downloads it from the internet. We need to just do this once.

# In[33]:


# Get the housing data
import os, tarfile, urllib

DOWNLOAD_ROOT = 'https://github.com/ageron/handson-ml/raw/master/'
HOUSING_URL = DOWNLOAD_ROOT + 'datasets/housing/housing.tgz'


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
fetch_housing_data()


# This runs the code to download the datasets and puts it in the current directory.
# 
# 

# In[34]:


fetch_housing_data()


# # Reading the data
# 
# Load the housing data like this:
# 

# In[35]:


import pandas as pd

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# # Data examination
# 
# Now let's examine the data a little bit
# 
# 

# In[36]:


housing = load_housing_data()
housing.head()


# In[37]:


housing.info()


# In[38]:


housing["ocean_proximity"].value_counts()


# In[39]:


housing.describe()


# In[40]:


housing.columns


# In[41]:


# Print out the occurrances of different values
housing["ocean_proximity"].value_counts()


# In[42]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()


# This following method shows how to see what the histograms of the individual values looks like.
# 

# In[43]:


housing["income_cat"] = pd.cut(housing["median_income"],
                              bins = [0., 1.5, 3.0, 4.5, 6., np.inf],
                              labels=[1, 2, 3, 4, 5])
housing["income_cat"].hist()

# And then let's drop the "income_cat" attribute to keep the data pristine
housing.drop("income_cat", axis=1, inplace=True)


# Let's examine the housing data to ensure that it has the correct tables
# 

# In[44]:


print (housing.info())
housing.describe()


# # Sampling
# 
# Sampling a test set and keeping it aside, so that it never gets involved in training

# In[45]:


import numpy as np

def split_training(data, test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(test_ratio * len(data))
    
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    # Since we provided a random seed, these values will
    # be the same for every subsequent run
    # print(test_indices[:20])
    
    return data.iloc[train_indices], data.iloc[test_indices]


# Keep 20% (0.2) of the data aside for testing later
train_set, test_set = split_training(housing, 0.2)
print("Training data has %d elements" % len(train_set))
print("Testing data has %d elements" % len(test_set))

# This can also be done with train_test_split like this
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print("Training data has %d elements" % len(train_set))
print("Testing data has %d elements" % len(test_set))


# That was purely random sampling. Sometimes you want to stratify your sampling to
# allow for matching the population by their characteristics: ensure enough males/females
# ensure enough lat/long spread, etc.
# 
# So here we do stratified sampling by income category instead.

# In[48]:


import pylab as pl

housing["income_cat"] = pd.cut(housing["median_income"],
                              bins = [0., 1.5, 3.0, 4.5, 6., np.inf],
                              labels=[1, 2, 3, 4, 5])
housing["income_cat"].hist()

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print(train_index[:10])
print (strat_test_set["income_cat"].value_counts() / len(strat_test_set))
test = pd.cut(strat_test_set["median_income"],
                              bins = [0., 1.5, 3.0, 4.5, 6., np.inf],
                              labels=[1, 2, 3, 4, 5])
test.hist()
pl.suptitle("Orange: Test set, Blue: Population")
        
# And then let's drop the "income_cat" attribute to keep the data pristine
housing.drop("income_cat", axis=1, inplace=True)


# In[49]:


# Now we copy the data, so that we only work with the training set and never see the test set

housing = strat_train_set.copy()

housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10, 7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
            )
plt.legend()


# Showing correlations between the variables. If you get the seaborn library, you can see the correlations between the variables and the "median_housing_value" which is the attribute we are trying to predict.

# In[50]:


plt.matshow(housing.corr())
plt.show()

import seaborn as sns

plt.figure(figsize=(15, 10))
sns.heatmap(housing.corr(), annot=True)


# Try to look at the quantile-quantile plot for a single variable to see how close it is to the normal distribution.

# In[51]:


import scipy, matplotlib

scipy.stats.probplot(housing["median_income"], dist="norm", plot=matplotlib.pyplot)


# In[52]:


from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age", "population"]
_=scatter_matrix(housing[attributes] ,figsize=(16, 12))


# In[53]:


# Create new variables that might be better correlated

housing["rooms_per_household"] = housing["total_rooms"] / housing ["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing ["total_rooms"]
housing["population_per_household"] = housing["population"] / housing ["households"]

# And let's see how well the new variables work
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# # Preparing the data
# 
# Got to handle the missing values and other quirks
# 
# 

# In[54]:


# The input attributes and the labels (known quantities we will be predicting)

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# Check which values are 'na' (not available) and hence have to be filled in. 
na_indices = np.where(pd.isna(housing["ocean_proximity"]) == True)

na_indices

# We could drop the na values in one of these ways

# Drop only the na rows
# housing.dropna(subset=["total_bedrooms"])

# Drop the entire attribute!
# housing.drop("total_bedrooms", axis=1)

# Fill it in with the median
# median = housing["total_bedrooms"].median()
# housing["total_bedrooms"].fillna(median, inplace=True)


# If we have any 'na' values, we would use an Imputer to find the means
# and then fill them in.
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
# The "ocean_proximity" attribute is categorical. The rest are numerical, and thus
# their medians can be calculated
housing_num = housing.drop("ocean_proximity", axis=1)

# Calculate the medians here
imputer.fit(housing_num)

print(imputer.statistics_)
print (housing_num.median().values)

# Now we can use this trained imputer to transform the dataset to fill in the
# missing values with medians across the board
X = imputer.transform(housing_num)

# A normal Pandas dataframe can be created out of this array like this
housing_transformed = pd.DataFrame(X, columns=housing_num.columns,
                                    index=housing_num.index)


# The housing_cat variable is the only categorical variable, so the book examines it and then converts it into
# a on hot encoding of various choices. I think it would be better to make it odinal

# In[55]:


housing[["ocean_proximity"]].head(10)


# In[56]:


from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()

housing_cat_encoded = ordinal_encoder.fit_transform(housing[["ocean_proximity"]])


# In[57]:


housing_cat_encoded[:10]


# In[58]:


ordinal_encoder.categories_


# There's no easy way to change the categories. You can renumber these values after the encoder has run.
# 
# The book really wants us to use a one-hot array with individual values for each of the categories.

# In[59]:


from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()

housing_cat_1hot = cat_encoder.fit_transform(housing[["ocean_proximity"]])
housing_cat_1hot


# Creating your own custom transformer. You do this to pre-process the data in the way
# that you want, and so it can be included in a data pipeline
# 

# In[60]:


from sklearn.base import BaseEstimator, TransformerMixin



# Indices of the various columns in X
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args, or **kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self, X, y=None):
        " Nothing done to fit "
        return self; # Nothing to fit
    
    def transform(self, X):
        "Adds the all derived attributes as columns"
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X,
                         rooms_per_household, 
                         population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, 
                         rooms_per_household,
                         population_per_household]
        
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# In[61]:


housing_extra_attribs


# # Feature scaling and Transformation Pipelines
# 
# We need to scale the features (values, not the predictions) so that convergence of
# most methods is faster

# In[62]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Add this later to clear variables
# %reset

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

# This is the full pipeline, including stratified sampling.
# Load the data (no need to fetch it again)
housing = load_housing_data()

# First, generate income categories by binning median_income into five bins.
housing["income_category"] = pd.cut(housing["median_income"],
                              bins = [0., 1.5, 3.0, 4.5, 6., np.inf],
                              labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_category"]):
    # Assign test and training sets
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# And then let's drop the "income_category" attribute to keep the data pristine
housing.drop("income_category", axis=1, inplace=True)
strat_train_set.drop("income_category", axis=1, inplace=True)
strat_test_set.drop("income_category", axis=1, inplace=True)

# Now we copy the data, so that we only work with the training set and never see the test set
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# If we have any 'na' values, we would use an Imputer to find the means
# and then fill them in.
imputer = SimpleImputer(strategy="median")

# The "ocean_proximity" attribute is categorical. The rest are numerical, and thus
# their medians can be calculated
housing_num = housing.drop("ocean_proximity", axis=1)

# Calculate the medians here
imputer.fit(housing_num)


# Indices of the various columns in X
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args, or **kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self, X, y=None):
        " Nothing done to fit "
        return self; # Nothing to fit
    
    def transform(self, X):
        "Adds the all derived attributes as columns"
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X,
                         rooms_per_household, 
                         population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, 
                         rooms_per_household,
                         population_per_household]

# The ordinal values (numbers)
number_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

# The categorical values (categories)
number_attribs = list(housing_num)
# Just one, which is ocean_proximity
categorical_attribs = ["ocean_proximity"]

data_pipe = ColumnTransformer([
    ('number', number_pipe, number_attribs),
    ('categorical', OneHotEncoder(), categorical_attribs)
])

housing_prepared = data_pipe.fit_transform(housing)


# # Training a model
# 
# Having prepared the model thus, we can proceed to training the model. We try different systems: Linear Regression
# Decision Trees, Decision Forests, and an ensemble method. The goal is to reduce mean square of errors since all values
# are alike, and we want a model that is close enough to the house prices to be able to predict them in tne future.

# In[63]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# In[64]:


small_X = housing.iloc[:5]
small_y = housing_labels.iloc[:5]

X_prep = data_pipe.transform(small_X)
print ("Predictions: ", lin_reg.predict(X_prep))
print ("Observed: ", list(small_y))


# In[65]:


# Let's calculate the errors
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)

linear_rMSE = np.sqrt(mean_squared_error(housing_labels, housing_predictions))

print("Linear regression: Root Mean Squared error: ", linear_rMSE)


# Let's compare this to the values that are in the labeled data. looks like the range is [14k, 500k] and is capped at 500k, so the error of 68k is significant.

# In[66]:


housing_labels.describe()
housing_labels.std()


# Let's define a function that will print the errors versus the range so we can see how big the error is in this case
# 

# In[67]:


def showRmsError(estimator, X, y, label="Estimator"):
    y_predicted = estimator.predict(X)
    rMSE = np.sqrt(mean_squared_error(y, y_predicted))
    true_range = y.max() - y.min()
    percent = (rMSE * 1.0) / true_range
    print (label, " has a squared root mean square error of ", rMSE)
    print ("This is roughly %.2f%% of the range" % percent)
    print ("Here are the mean (%f) and standard deviation(%f) of the true range." % (y.mean(), y.std()))
    print ("The RMS error is %.2f%% of the standard deviation " % (rMSE*1.0/y.std()))


# In[68]:


showRmsError(lin_reg, housing_prepared, housing_labels, "Linear estimator")


# In[69]:


# Let's try a decision trees
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
showRmsError(tree_reg, housing_prepared, housing_labels, "Decision Tree Estimator")


# # Cross validation
# 
# Since the Decision Tree Estimator is over-fitting, then adjust by doing cross validation
# 

# In[70]:


from sklearn.model_selection import cross_val_score

def display_scores(label, scores):
    print ("------------ RMS Error for %s -----------" % label)
    print ("Scores: ", scores)
    print ("Mean: ", scores.mean())
    print ("Standard deviation: ", scores.std())

tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                        scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)    
display_scores("Decision Tree", tree_rmse_scores)


# Try it out with Linear Regression
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                        scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores("Linear Regression", lin_rmse_scores)


# In[71]:


# Let's try that a hundred times to see what we find

tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                        scoring="neg_mean_squared_error", cv=100)
tree_rmse_scores = np.sqrt(-tree_scores)    
display_scores("Decision Tree", tree_rmse_scores)


# Try it out with Linear Regression
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                        scoring="neg_mean_squared_error", cv=100)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores("Linear Regression", lin_rmse_scores)


# In[72]:


# What if we only did this five times?

tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                        scoring="neg_mean_squared_error", cv=5)
tree_rmse_scores = np.sqrt(-tree_scores)    
display_scores("Decision Tree", tree_rmse_scores)


# Try it out with Linear Regression
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                        scoring="neg_mean_squared_error", cv=5)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores("Linear Regression", lin_rmse_scores)


# Looks like the $\sigma$ increases as we have more cross validation folks. Why is that?
# 
# Perhaps we need to divide by the number of steps?

# In[73]:


# Why does the cross validation need a trained model? If I understand correctly,
# it is trained on n-1 folds, and 1 fold is kept for validation. If that is true,
# then we should be able to take an untrained estimator, and get similar values (modulo
# choice of folds). Let's try that.

tree_reg_untrained = DecisionTreeRegressor()
tree_scores = cross_val_score(tree_reg_untrained, housing_prepared, housing_labels,
                        scoring="neg_mean_squared_error", cv=5)
tree_rmse_scores = np.sqrt(-tree_scores)    
display_scores("Decision Tree", tree_rmse_scores)


# Try it out with Linear Regression
lin_reg_untrained = LinearRegression()
lin_scores = cross_val_score(lin_reg_untrained, housing_prepared, housing_labels,
                        scoring="neg_mean_squared_error", cv=5)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores("Linear Regression", lin_rmse_scores)


# And that's true. You don't need a trained estimator. You just need an estimator created.
# This is great. Learning things!

# In[74]:


# A random forest creates a few decision trees and averages their output.
# For housing, it would be a lot better to create decision tree models that are location/zip
# code specific, and then use those in an ensemble method.

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                        scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)    
display_scores("Decision Tree", tree_rmse_scores)


# In[75]:


# Train it on the full data, and see if it overfits on the full data.
forest_reg.fit(housing_prepared, housing_labels)

showRmsError(forest_reg, housing_prepared, housing_labels, "Random Forest Estimator")


# Ah, so you run the error on the fully labeled, prepared data and then run cross-validation to see how well the system *will* perform on unseen data. That's the point.
# 
# So in the above example, the Random Forest have impressively small error: 18k on the data,
# but when we use cross validation, we realize that on the test set, the errors are actually
# quite high: 71k or so. So the Random Forest is overfitting the training data, and will
# perform much worse on information it hasn't seen.
# 
# However, if we are to see the regression as a way of *capturing* the data from the data,
# then the regressor is a good estimator for data seen already.

# In[76]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    {'bootstrap': [True], 'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]}
]

# We want to search for best parameters for a Random Forest regressor
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                          scoring='neg_mean_squared_error',
                          return_train_score=True)

# Now train the model with all the choices
grid_search.fit(housing_prepared, housing_labels)

# grid_search.fit(housing_prepared, housing_labels)
showRmsError(grid_search, housing_prepared, housing_labels, "Grid Search Estimator")


# Let's get the best estimator here and examine what we found

# In[77]:


selected_estimator = grid_search.best_estimator_

grid_search.best_params_


# In[78]:


# The cross_validation results:
cvres = grid_search.cv_results_
for mean_score, params in sorted(zip(cvres["mean_test_score"], cvres["params"])):
    print(np.sqrt(-mean_score), params)


# Let's see what the important attributes were and do a final evaluation of the system

# In[79]:


feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# In[80]:


# Let's see them next to attribute names
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = data_pipe.named_transformers_["categorical"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])

# All the names of attributes
attributes_all = number_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes_all), reverse=True)


# # Final validation of the system against the test set

# In[81]:


final_model = grid_search.best_estimator_

# Separate out input variables and actual labels
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

# Important to call transform and not fit_transform! We should not be fitting to the
# test set.
X_test_prepared = data_pipe.transform(X_test)

showRmsError(final_model, X_test_prepared, y_test, "Final model")


# # Exercises
# 

# In[82]:


# Trying our RandomizedSearchCV

from sklearn.model_selection import RandomizedSearchCV

# All the values to try
param_list = [
    {'bootstrap': [True, False]},
    {'n_estimators': [i for i in range(3, 30)]},
    {'max_features': [i for i in range(2, 8)]},
]

# We want to search for best parameters for a Random Forest regressor
forest_reg = RandomForestRegressor()
random_search = RandomizedSearchCV(forest_reg, param_list,
                           cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True,
                           n_iter = 5, # Test this out, and then increase when ready.
                          )

# Now train the model with all the choices
random_search.fit(housing_prepared, housing_labels)

# grid_search.fit(housing_prepared, housing_labels)
showRmsError(random_search, housing_prepared, housing_labels, "Randomized Search Estimator")


# In[83]:


random_search.best_params_


# In[84]:


# Exercise 4. Create a single pipeline that prepares the data and does the prediction

# We have two things to do: prepare the data using the existing data pipeline,
# And then predicting using the final model.
predict_pipe = Pipeline([
    ('preparation', data_pipe),
    ('prediction', final_model)
])

y_prediction = predict_pipe.predict(X_test)
rMSE = np.sqrt(mean_squared_error(y_test, y_prediction))
print(rMSE)

# But if you apply the steps separately, you get a different answer!
X_test_prep = data_pipe.transform(X_test)
y_pred2 = final_model.predict(X_test_prep)
rMSE = np.sqrt(mean_squared_error(y_test, y_pred2))
print(rMSE)


# In[85]:


# Let's try Ridge regression to see if it produces better estimates
# Ridge regression with the closed-form Normal equation

from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(housing_prepared, housing_predictions)

showRmsError(ridge_reg, housing_prepared, housing_labels, "Normal Equation, Ridge Regression Estimator")


# In[86]:


print(housing_prepared.shape)
print(housing_labels.shape)


# In[87]:


housing_prepared.shape

