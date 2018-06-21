#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 16:32:56 2018

@author: hm1234
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate


with open('psi2.pickle', 'rb') as handle:
    psi = pickle.load(handle)
with open('ayc_r.pickle', 'rb') as handle:
    ayc_r = pickle.load(handle)
with open('ayc_te.pickle', 'rb') as handle:
    ayc_te = pickle.load(handle)
with open('ayc_ne.pickle', 'rb') as handle:
    ayc_ne = pickle.load(handle)
with open('psi.pickle', 'rb') as handle:
    psi2 = pickle.load(handle)
    

#psi_0_loc = np.where(psi['data'][1] == np.amin(psi['data'][1]))[0][0]
#psi_0 = psi['data'][1][psi_0_loc]
#
#
plt.figure(1)
plt.contour(psi['x'],psi['time'],psi['data'], 50)
plt.colorbar()
plt.ylabel('time (s)')
plt.xlabel('radial position (normalised?)')
plt.title('psi at z=0')
#
#for i in range(0, np.shape(psi['data'])[0]):
#    plt.figure(2)
#    plt.plot(psi['x'], psi['data'][i])
#    plt.xlabel('$\Psi_{N}$', rotation=0)
#    plt.ylabel('$\Psi$', rotation=0)
#plt.show()


min_t = ayc_r['time'][0]
max_t = ayc_r['time'][-1]
len_t = len(ayc_r['time'])
min_x = ayc_r['x'][0]
max_x = ayc_r['x'][-1]
len_x = len(ayc_r['x'])

min_psi_x = psi['x'][0]
min_psi_t = psi['time'][0]
max_psi_x = psi['x'][-1]
max_psi_t = psi['time'][-1]
len_psi_x = len(psi['x'])
len_psi_t = len(psi['time'])

X = np.array(psi['x'])
T = np.array(psi['time'])

x,t = np.meshgrid(X,T)

test = psi['data']
print(test)

print('hi')
f = interpolate.interp2d(x, t, test, kind='linear')
print('ho')

Xnew = np.array(ayc_r['data'][20])
Tnew = np.array(ayc_r['time'])

test_new = f(Xnew,Tnew)

print(test_new)

plt.figure(3)
plt.contour(ayc_r['x'],ayc_r['time'],test_new, 50)
plt.colorbar()
plt.ylabel('time (s)')
plt.xlabel('radial position (normalised?)')
plt.title('interpolated psi :\'(')

plt.show()












