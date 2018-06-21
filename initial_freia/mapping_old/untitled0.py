#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 16:32:56 2018

@author: hm1234
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

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
    
def getshape(d):
    if isinstance(d, dict):
        return {k:getshape(d[k]) for k in d}
    else:
        # Replace all non-dict values with None.
        return None
  
#plt.figure(1)
#plt.contour(psi['data'], 50)
#plt.colorbar()
#plt.xlabel('radial position (array index)')
#plt.title('psi at z=0')

#plt.figure(6)     
#plt.contour(psi2['data'][20,:,:], 50)
#plt.colorbar()

plt.figure(5)
plt.contour(psi['x'],psi['time'],psi['data'], 50)
plt.colorbar()
plt.ylabel('time (s)')
plt.xlabel('radial position (normalised?)')
plt.title('psi at z=0')

plt.figure(2)
plt.subplot(121)
plt.contourf(ayc_te['time'],ayc_te['x'],ayc_te['data'].T, 50)
plt.colorbar()
plt.xlabel('time (s)')
plt.ylabel('channel number')
plt.title('T_e vs channel number')
plt.subplot(122)
plt.contourf(ayc_ne['time'],ayc_ne['x'],ayc_ne['data'].T, 50)
plt.colorbar()
plt.title('n_e vs channel number')
plt.suptitle('electron stuff vs channel number')


plt.figure(3)
plt.contour(ayc_r['time'],ayc_r['x'],ayc_r['data'].T, 30)
plt.colorbar()
plt.title('R vs channel number and time')

plt.figure(4)
plt.plot(ayc_r['x'],ayc_r['data'].T)
plt.xlabel('radial index (channel number)')
plt.ylabel('Major Radius (m)')

psi_0_loc = np.where(psi['data'][1] == np.amin(psi['data'][1]))[0][0]
psi_0 = psi['data'][1][psi_0_loc]


for i in range(0, np.shape(psi['data'])[0]):
    plt.figure(7)
    plt.plot(psi['x'], psi['data'][i])
    plt.xlabel('s')
    plt.ylabel('$\Psi$', rotation=0)

plt.show()











