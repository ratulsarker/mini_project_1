'''
Project: Mini Project 1 
Authors: Everyone write your names, Jacob Harper
Group ID: 15

'''

import numpy as np
import pandas as pd
import requests
from io import StringIO

# URL's of UCI Datasets

uci_url_A = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
uci_url_Io = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
uci_url_Ir = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
uci_url_W = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"


# Downloading datasets
resp_A = requests.get(uci_url_A)
resp_Io = requests.get(uci_url_Io)
resp_Ir = requests.get(uci_url_Ir)
resp_W = requests.get(uci_url_W)

data_A = resp_A.text
data_Io = resp_Io.text
data_Ir = resp_Ir.text
data_W = resp_W.text

# Transfer data into Pandas DataFrames

columns_A = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex']
columns_Io = ['Attribute1', 'Attribute2', 'Attribute3', 'Attribute4', 'Attribute5', 'Attribute6', 'Attribute7', 'Attribute8', 'Attribute9', 'Attribute10']
columns_Ir = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
columns_W = ['class', 'Alcohol', 'Malicacid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins']


adult = pd.read_csv(StringIO(data_A), header=None, names=columns_A)
ionosphere = pd.read_csv(StringIO(data_Io), header=None, names=columns_Io)
iris = pd.read_csv(StringIO(data_Ir), header=None, names=columns_Ir)
wine = pd.read_csv(StringIO(data_W), header=None, names=columns_W)

# Extract features and labels
'''
X = iris.drop('class', axis=1).values  # Features
y = iris['class'].values  # Labels


'''


