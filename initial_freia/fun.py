#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 11:06:10 2018

@author: hm1234
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

rng = np.linspace(0.01, np.sqrt(2), 500)
rng2 = np.arange(10,500)
max_rng = np.amax(rng2)
hi = []
for j in rng2:
    n = j
    tmp = []
    for i in rng:
        a = i
        b = '**a'
        c = str(a) + n*b
        tmp.append(eval(c))
    if j % 23 == 0:
        print('{0:1.0f}'.format(j/max_rng *100), '%')
    hi.append(tmp)
print('Done!')
X_rng, Y_rng = np.meshgrid(rng2, rng)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_rng, Y_rng, np.array(hi).T)



