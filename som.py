import numpy as np
import matplotlib as plt
import pandas as pd


"""
SOM on credit_card applications dataset to identify fraud 
"""

# get the dataset

dataset = pd.read_csv("Self_Organizing_Maps/Credit_Card_Applications.csv")

# the rows are customers, columns are attributes of each customer
# let's locate the customers who were approved, since they have higher priority for discovering fraud for
# we split up the dataset into two subsets, X containing the information and Y containing the last column,
# which tells us which customers have been approved.

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# when we train the actual network, we will only use X

# let's do some feature scaling
# normalization (all features between 0 and 1

# use sklearn's minMaxScaler to normalize X
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range= (0, 1))
X_normal = sc.fit_transform(X)

# let's train our SOM now
# use miniSOM from http://pydoc.net/MiniSom/1.1/minisom/

from minisom import MiniSom
# parameters: X, Y dimensions, input length (number of columns in X),
# sigma (radius of different neighborhoods in the grid, which we keep as default)
# learning rate (how much the weights are updated at each step, keep as default)
# decay parameter (keep as default as well)

som = MiniSom(10, 10, len(X[0]))

# initialize weights randomly
som.random_weights_init(X)

# now let's begin training on data X
som.train_random(X, num_iteration=100)


