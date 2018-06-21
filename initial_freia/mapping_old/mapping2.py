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
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.gaussian_process import GaussianProcessRegressor
    
###############################################################################
#####    load the pickles

with open('psi_r_28819.pickle', 'rb') as handle:
    psi_r_28819 = pickle.load(handle)
with open('psi_r_30417.pickle', 'rb') as handle:
    psi_r_30417 = pickle.load(handle)
with open('psi_rz_28819.pickle', 'rb') as handle:
    psi_rz_28819 = pickle.load(handle)
with open('psi_rz_30417.pickle', 'rb') as handle:
    psi_rz_30417 = pickle.load(handle)

with open('ayc_r_28819.pickle', 'rb') as handle:
    ayc_r_28819 = pickle.load(handle)
with open('ayc_ne_28819.pickle', 'rb') as handle:
    ayc_ne_28819 = pickle.load(handle)
with open('ayc_te_28819.pickle', 'rb') as handle:
    ayc_te_28819 = pickle.load(handle)
with open('ayc_r_30417.pickle', 'rb') as handle:
    ayc_r_30417 = pickle.load(handle)
with open('ayc_ne_30417.pickle', 'rb') as handle:
    ayc_ne_30417 = pickle.load(handle)
with open('ayc_te_30417.pickle', 'rb') as handle:
    ayc_te_30417 = pickle.load(handle)

with open('efm_grid_r_28819.pickle', 'rb') as handle:
    efm_grid_r_28819 = pickle.load(handle)
with open('efm_grid_z_28819.pickle', 'rb') as handle:
    efm_grid_z_28819 = pickle.load(handle)
with open('efm_grid_r_30417.pickle', 'rb') as handle:
    efm_grid_r_30417 = pickle.load(handle)
with open('efm_grid_z_30417.pickle', 'rb') as handle:
    efm_grid_z_30417 = pickle.load(handle)
    
###############################################################################
#####    let's declare some arrays

# 2-dimensional (r,z) at different times
# psi as function of radius in m
psi_x = psi_rz_28819['x']
# psi as function of time in s
psi_t = psi_rz_28819['time']
# value of psi at specific radius, time, and z
psi_dat = psi_rz_28819['data']

# psi grid values
# same as psi_x
efm_grid_r = efm_grid_r_28819['data'].squeeze()
# would be psi_z i.e. z location of flux
efm_grid_z = efm_grid_z_28819['data'].squeeze()

# channel number vs radius
# x is the channel number
ayc_r_x = ayc_r_28819['x']
# t is the time
ayc_r_t = ayc_r_28819['time']
# dat is the radius correesponding to specific channel number at some time t
ayc_r_dat = ayc_r_28819['data']

# electron temperature data given as a function of time and channel number
# channel number
ayc_te_x = ayc_te_28819['x']
# time
ayc_te_t = ayc_te_28819['time']
# T_e at channel number and time
ayc_te_dat = ayc_te_28819['data']

# arbitrary value to check slices of data
chk = 33
# marker size for plotting
mrk = 2

###############################################################################
#####    do some stuff

# find the z=0 coordinate
z0_axis = np.where(efm_grid_z == 0)[0][0]
# define psi only along the z0 axis
psi_dat_z0 = psi_dat[:,z0_axis,:]

#print('psi_dat_z0 has shape:', psi_dat_z0.shape, 'but want:', ayc_r_dat.shape)

# used to see how good the interpolation is in the time axis
interp_test = interpolate.interp1d(psi_t, psi_dat_z0[:,chk],
                                   kind='cubic', fill_value='extrapolate')
test = interp_test(ayc_r_t)

# used to see how good the interpolation is in the radial direction
interp_test = interpolate.interp1d(psi_x, psi_dat_z0[chk],
                                   kind='cubic', fill_value='extrapolate')
test2 = interp_test(ayc_r_dat[chk])

#psi_t_interp = []
#
#for i in range(0, len(psi_x)):
#    interp_test = interpolate.interp1d(psi_t, psi_dat_z0[:,i], kind='cubic',
#                                       fill_value='extrapolate')
#    psi_t_interp.append(interp_test(ayc_r_t))
#    
#psi_t_interp = np.array(psi_t_interp)
#
#print('psi_t_interp has shape:', psi_t_interp.shape)
#
#psi_x_interp = []
#
#for i in range(0, len(psi_t)):
#    interp_test = interpolate.interp1d(psi_x, psi_dat_z0[i], kind='cubic',
#                                       fill_value='extrapolate')
#    psi_x_interp.append(interp_test(ayc_r_dat[i]))
#    
#psi_x_interp = np.array(psi_x_interp)
#
#print('psi_x_interp has shape:', psi_x_interp.shape)


###############################################################################
#####    interpolation

# interpolation of psi data so that it corresponds to the same channel number
# and time as the electron temperature data
# perform 2 seperate 1d interpolations. Not ideal but was struggling with 2d
# interpolation. Have some fun trying scikit learn and knn approach :D

'''
for j in range(len(ayc_r['data'][0,:])):
  R_channel_av = ayc_r['data'][:,j].mean()
  for i in range(len(psi['time'])):
    psi_channel_t[j,i] = InterpolatedUnivariateSpline(psi['x'],
                 psi['data'][i,32,:])(R_channel_av)
'''

psi_t_interp = []

for i in range(0, len(psi_x)):
    interp_test = interpolate.interp1d(psi_t, psi_dat_z0[:,i], kind='cubic',
                                       fill_value='extrapolate')
    psi_t_interp.append(interp_test(ayc_r_t))
    
psi_t_interp = np.array(psi_t_interp).T
# psi_t_interp is psi but with same time values as T_e data

#print('psi_t_interp has shape:', psi_t_interp.shape)

psi_x_interp = []

for i in range(0, psi_t_interp.shape[0]):
    interp_test = interpolate.interp1d(psi_x, psi_t_interp[i], kind='cubic',
                                       fill_value='extrapolate')
    psi_x_interp.append(interp_test(ayc_r_dat[i]))
    
psi_x_interp = np.array(psi_x_interp)
# psi_t_interp is psi but with same channel number values as T_e data
# since the time data is also the same the outputted array should be the
# correct shape

psi_dat_z0_new = psi_x_interp

#print('psi_x_interp has shape:', psi_x_interp.shape)

###############################################################################
#####    some mapping?

# after interpolation we have psi and T_e at the same channel numbers and time
# just gotta map channel number to psi. easy right? :'( 

tme = ayc_te_t
psi_ch = np.linspace(1, len(ayc_te_x), len(ayc_te_x))

#a = np.ndarray.flatten(ayc_te_dat)
#b = np.ndarray.flatten(psi_dat_z0_new)

psi_sort = np.sort(psi_dat_z0_new[chk,:])
psi_sort_ind = np.argsort(psi_dat_z0_new[chk,:])
T_e_sort = ayc_te_dat[chk,:][psi_sort_ind]

a = []
b = []

for i in range(len(tme)):
    a.append(np.sort(psi_dat_z0_new[i,:]))
    tmp = np.argsort(psi_dat_z0_new[i,:])
    b.append(ayc_te_dat[i,:][psi_sort_ind])
    
psi_sorted = np.array(a)
Te_sorted = np.array(b)

###############################################################################
#####    Experimental smoothing

a = np.where(np.isnan(T_e_sort) == False)[0]

x = psi_sort[a]
y = T_e_sort[a]
x_pred = np.linspace(np.amin(psi_sorted), np.amax(psi_sorted), len(ayc_te_x))

gp = GaussianProcessRegressor()
gp.fit(np.atleast_2d(x).T, y)
y_pred = gp.predict(np.atleast_2d(x_pred).T)

###############################################################################
#####    plotting

#plt.figure()
##plt.plot(ayc_r_t, ayc_r_dat[:,20])
#plt.contourf(ayc_r_dat.T, 11)
#plt.colorbar()
#plt.ylabel('channel number')
#plt.xlabel('time (index value)')
#plt.title('Radius (m)')

plt.figure()
plt.contourf(tme, psi_ch, ayc_te_dat.T, 33)
plt.colorbar()
plt.xlabel('time (s)')
plt.ylabel('channel number')
plt.title('$T_{e}$')

plt.figure()
plt.contourf(tme, psi_ch, psi_dat_z0_new.T, 33)
plt.colorbar()
plt.xlabel('time (s)')
plt.ylabel('channel number')
plt.title('psi_new at z=0')

#plt.figure()
#plt.contourf(psi_t, psi_x, psi_dat_z0.T, 33)
#plt.colorbar()
#plt.xlabel('time (s)')
#plt.ylabel('radial position (m)')
#plt.title('psi at z=0')

#plt.figure()
#plt.plot(psi_ch, psi_dat_z0_new[chk])
#plt.xlabel('channel number')
#plt.ylabel('$\Psi$', rotation=0)
#
#plt.figure()
#plt.plot(psi_ch, ayc_te_dat[chk])
#plt.xlabel('channel number')
#plt.ylabel('$T_{e}$', rotation=0)

plt.figure()
plt.plot(psi_sort, T_e_sort)
plt.plot(x_pred, y_pred, 'r')
plt.xlabel('$\Psi$')
plt.ylabel('$T_{e}$', rotation=0)

#plt.figure()
#plt.contour(efm_grid_r, efm_grid_z, psi_dat[chk,:,:], 33)
#plt.axis('equal')
#plt.colorbar()
#plt.ylabel('z (m)')
#plt.xlabel('radial position (m)')
#plt.title('$\Psi (r,z)$')
#
#plt.figure()
#plt.contour(psi_dat_z0, 33)
#
#plt.figure()
#plt.contour(psi_dat_z0_new, 33)
#
#plt.figure()
#plt.plot(ayc_r_x, ayc_r_dat[chk])
#plt.xlabel('radial index (channel number)')
#plt.ylabel('radial position (m)')
#
#plt.figure()
#plt.plot(psi_t, psi_dat_z0[:,chk], 'bx', ms=mrk)
#plt.plot(ayc_r_t, test, 'ro', ms=mrk)
#plt.xlabel('time (s)')
#plt.ylabel('$\Psi$', rotation=0)
#
#plt.figure()
#plt.plot(psi_x, psi_dat_z0[chk], 'bx', ms=mrk)
#plt.plot(ayc_r_dat[chk], test2, 'ro', ms=mrk)
#plt.xlabel('radial position (m)')
#plt.ylabel('$\Psi$', rotation=0)


###############################################################################
#####    Fancy animations

#for i in range(0, psi_rz_28819['time'].shape[0]):
#    fig = plt.figure()
#    plt.contour(efm_grid_r.squeeze(), efm_grid_z.squeeze(),
#                psi_dat[i,:,:], 50)
#    plt.colorbar()
#    plt.ylabel('z (m)')
#    plt.xlabel('radial position (m)')
#    plt.title('$\Psi (r,z)$')
#    print(i)
#    plt.savefig(str(i).zfill(4) +'.png')
#    plt.close(fig)

#for i in range(0, len(tme)):
#    fig = plt.figure()
#    plt.plot(psi_dat_z0_new[i], ayc_te_dat[i])
#    plt.xlabel('$\Psi$')
#    plt.ylabel('$T_{e}$', rotation=0)
#    plt.title('$T_{e}$ vs $\Psi$')
#    print(i)
#    plt.savefig(str(i).zfill(4) +'.png')
#    plt.close(fig)

plt.show()












