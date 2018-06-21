#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 13:57:37 2018

@author: hm1234
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import pandas as pd


#data = pd.DataFrame.from_dict({
#    'x': np.random.randint(low=1, high=10, size=5),
#    'y': np.random.randint(low=-1, high=1, size=5),
#    #'z': np.random.randint(low=-2, high=5, size=5),
#})
#
#p = PolynomialFeatures(degree=3).fit(data)
##print(p.get_feature_names(data.columns))
#
#features = pd.DataFrame(p.transform(data), 
#                        columns=p.get_feature_names(data.columns))
#print(features)


x = np.array([0., 4., 9., 12., 16., 20., 24., 27.])
y = np.array([2.9,4.3,66.7,91.4,109.2,114.8,135.5,134.2])

x_plot = np.linspace(0, max(x), 100)
# create matrix versions of these arrays
X = x[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]

plt.scatter(x, y, label="training points")

for degree in [1,2,3,4,5,6]:#np.arange(3, 6, 1):
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(X, y)
    y_plot = model.predict(X_plot)
    plt.plot(x_plot, y_plot, label="degree %d" % degree)

plt.legend()

plt.show()

ridge = model.named_steps['ridge']
print(ridge.coef_)









