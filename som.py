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
X = sc.fit_transform(X)

# let's train our SOM now
# use miniSOM from http://pydoc.net/MiniSom/1.1/minisom/

from minisom import MiniSom
# parameters: X, Y dimensions, input length (number of columns in X),
# sigma (radius of different neighborhoods in the grid, which we keep as default)
# learning rate (how much the weights are updated at each step, keep as default)
# decay parameter (keep as default as well)

som = MiniSom(x=10, y=10, input_len=len(X[0]), sigma=1.0, learning_rate=0.5)

# initialize weights randomly
som.random_weights_init(X)

# now let's begin training on data X
som.train_random(X, num_iteration=100)

# the way we are detecting fraud here is using the som to find outliers, we reason that the outliers are the
# customers committing fraud. The outliers will be identified using the MID (mean inter-neuron distance inside
# radius sigma). The higher the MID, the more a neuron is an outlier.

# the larger the MID, the closer to white the color of the neuron will be
from pylab import bone, pcolor, colorbar, plot, show

# first initialize the figure on which we will display the map
bone()

# add the MID information of each neuron onto our map, and have it correspond to a color
pcolor(som.distance_map().T) # take the transpose of the MID matrix returned by the distance map method
colorbar()
markers = ["o", "s"]
colors = ["r", "g"]

for i, x in enumerate(X): # i is the indicies of the customer, x is the vector of customer information
    # get the winning node of customer x
    w = som.winner(x)

    # add marker onto the winning node depending on whether the customer got approval
    plot(w[0] + 0.5, w[1] + 0.5, markers[Y[i]], markeredgecolor= colors[Y[i]], markerfacecolor= "None",
         markersize= 10, markeredgewidth= 2)


# display the plot
show()

# ok let's find the actual customers who are the frauds
# get the mapping from winning nodes to customers from minisom

# get the mapping from data to winning nodes (mapping is a dictionary)
mappings = som.win_map(X)

# when doing this in jupyter notebook, we read the map and then get the coordinates that are white
# for example, if the white coords are (8, 1) (6, 8)
frauds = np.concatenate((mappings[(8, 1)], mappings[(6, 8)]), axis=0)

print(frauds)