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

