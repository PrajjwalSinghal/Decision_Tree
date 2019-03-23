#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 17:02:53 2019

@author: prajjwalsinghal
"""

import numpy as np
import pandas as pd

#Importing the dataset

dataset = pd.read_csv('Decision_Tree_Data.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# Encodeing the categorical data
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
LE = preprocessing.LabelEncoder()
for temp in range(0,3):
    X[:, temp] = LE.fit_transform(X[:,temp])
print(X)

# Fitting the data
classifier = DecisionTreeClassifier(criterion = 'entropy',random_state = 0)
classifier.fit(X, Y)

#Predicting the new predictions
y_pred = classifier.predict(X)

print(y_pred)


# Code for Visualizing the graph
# https://pythonprogramminglanguage.com/decision-tree-visual-example/

# Visualize data
import pydotplus
import collections
from sklearn import tree

data_feature_names = [ 'Windy?', 'Air Quality Good?', 'HOT?' ]
dot_data = tree.export_graphviz(classifier,
                                feature_names=data_feature_names,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('tree.png')


