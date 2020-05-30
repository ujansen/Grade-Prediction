# This file takes a look at the data and pre-processes it to make it ready to be used for ML algorithms

# Import required libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import seaborn as sns

# Import libaries to transform our features 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

# Read the data into csv files and see if there are null values within the data
dataset = pd.read_csv('train.csv', index_col = 'id')
test_dataset = pd.read_csv('test.csv', index_col = 'id')
dataset.info()

# See descriptive statistics of the data
dataset.describe()

# Look at the distribution of features and their correlations with respect to the target value
%matplotlib inline
dataset.hist(bins = 20 , figsize = (20,15))
plt.show()

# Heat map of correlation of features
correlation_matrix = dataset.corr()
fig = plt.figure(figsize = (12,9))
sns.heatmap(correlation_matrix, vmax = 0.8, square = True)
plt.show()

# Making the pipeline to fit and transform our data
# The numerical data is being scaled and the categorical data is being one-hot encoded
numerical_features = list(dataset.select_dtypes(exclude = object))
numerical_features.remove("grade")
categorial_features = list(dataset.select_dtypes(include = object))
numeric_pipline = make_pipeline(StandardScaler())
full_pipeline = make_column_transformer((numeric_pipline, numerical_features),(OneHotEncoder(), categorial_features))

# Pre-processing the data by scaling the numerical data and one-hot encoding the categorical data
X = dataset.drop(columns = "grade")
X = full_pipeline.fit_transform(X)
y = dataset.grade
