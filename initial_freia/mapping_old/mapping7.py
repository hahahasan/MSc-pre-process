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
from scipy.signal import argrelextrema
from mpl_toolkits.mplot3d import Axes3D
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
chk_t = 44
chk_x = 44
# marker size for plotting
mrk = 2

###############################################################################
#####    useful functions

# finds array index corresponding to array value closest to some value
def find_closest(data, v):
	return (np.abs(data-v)).argmin()

###############################################################################
#####    do some stuff

# find the z=0 coordinate
z0_axis = np.where(efm_grid_z == 0)[0][0]
# define psi only along the z0 axis
psi_dat_z0 = psi_dat[:,z0_axis,:]

#print('psi_dat_z0 has shape:', psi_dat_z0.shape, 'but want:', ayc_r_dat.shape)

# used to see how good the interpolation is in the time axis
interp_test = interpolate.interp1d(psi_t, psi_dat_z0[:,chk_x],
                                   kind='cubic', fill_value='extrapolate')
test = interp_test(ayc_r_t)

# used to see how good the interpolation is in the radial direction
interp_test = interpolate.interp1d(psi_x, psi_dat_z0[chk_t],
                                   kind='cubic', fill_value='extrapolate')
test2 = interp_test(ayc_r_dat[chk_t])

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

psi_sort = np.sort(psi_dat_z0_new[chk_t,:])
psi_sort_ind = np.argsort(psi_dat_z0_new[chk_t,:])
T_e_sort = ayc_te_dat[chk_t,:][psi_sort_ind]

a = []
b = []

for i in range(len(tme)):
    a.append(np.sort(psi_dat_z0_new[i,:]))
    tmp = np.argsort(psi_dat_z0_new[i,:])
    b.append(ayc_te_dat[i,:][psi_sort_ind])
    
psi_sorted = np.array(a)
Te_sorted = np.array(b)

psi_rng = np.linspace(np.amin(psi_sorted), np.amax(psi_sorted), len(ayc_te_x))

te_2d = []

for i in range(len(tme)):
    te_2d.append(Te_sorted)

psi_dat_z0_new[chk_t,:], ayc_te_dat[chk_t,:]

te_2d_2 = []

for i in range(len(tme)):
    te_2d_2.append(ayc_te_dat[i,:])

te_2d_2 = np.array(te_2d_2)

###############################################################################
#####    Experimental normalisation

max_psi = []
mag_ax = []
norm_ind = []
for i in range(len(tme)):
    a = np.where(psi_dat_z0_new[i] == np.amax(psi_dat_z0_new[i]))[0][0]
    max_psi.append(psi_dat_z0_new[i][a])
    mag_ax.append(ayc_r_dat[i][a])
    norm_ind.append(a)
max_psi = np.array(max_psi)
mag_ax = np.array(mag_ax)
norm_ind = np.array(norm_ind)

psi_norm = []
for i in range(len(tme)):
    tt = psi_dat_z0_new[i] - max_psi[i]
    tt[norm_ind[i]:,] = abs(tt[norm_ind[i]:,])
    psi_norm.append(tt)
psi_norm = np.array(psi_norm)

###############################################################################
#####    get rid of NaNs

Te = []
psi_fin = []

for i in range(len(tme)):
    nan_drop = np.where(np.isnan(ayc_te_dat[i]) == False)[0]
    Te.append(ayc_te_dat[i][nan_drop])
    psi_fin.append(psi_norm[i][nan_drop])
    
Te = np.array(Te)
psi_fin = np.array(psi_fin)

###############################################################################
#####    get rid of anomolies

####    find peaks

Te_peaks = []
psi_peaks = []

for i in range(len(tme)):
    peaks = argrelextrema(Te[i], np.greater, order=4)
    Te_peaks.append(Te[i][peaks])
    psi_peaks.append(psi_fin[i][peaks])

Te_peaks = np.array(Te_peaks)
psi_peaks = np.array(psi_peaks)

#### moving average????

#mylist = Te[25]
#N = 3
#cumsum, moving_aves = [0], []
#for i, x in enumerate(mylist, 1):
#    cumsum.append(cumsum[i-1] + x)
#    if i>=N:
#        moving_ave = (cumsum[i] - cumsum[i-N])/N
#        #can do stuff with moving_ave here
#        moving_aves.append(moving_ave)
#        
#def chunk(seq, num):
#    avg = len(seq) / float(num)
#    out = []
#    last = 0.0
#
#    while last < len(seq):
#        out.append(seq[int(last):int(last + avg)])
#        last += avg
#
#    return np.array(out)

n_slice = 5

for hi in range(0,4):
    for k in range(len(tme)):
        tmp = np.array_split(Te[k], len(Te[k])/n_slice)
        for j in range(len(tmp)):
            m = np.mean(tmp[j])
            s = np.std(tmp[j])
            for n, i in enumerate(tmp[j]):
                if int(i) > m + s:
                    ia = np.indices(tmp[j].shape)
                    not_ind = np.setxor1d(ia, np.where(tmp[j]) == int(i))
                    m2 = np.mean(tmp[j][not_ind])
                    tmp[j][n] = m2
                
#for k in range(len(tme)):
#    tmp = np.array_split(Te[k], len(Te[k])/n_slice)
#    for j in range(len(tmp)):
#        m = np.mean(tmp[j])
#        s = np.std(tmp[j])
#        for n, i in enumerate(tmp[j]):
#            if int(i) > m + s:
#                ia = np.indices(tmp[j].shape)
#                not_ind = np.setxor1d(ia, np.where(tmp[j]) == int(i))
#                m2 = np.mean(tmp[j][not_ind])
#                tmp[j][n] = m2
        
#for n, i in enumerate(tst):
#    if i > m + s:
#        ia = np.indices(tst.shape)
#        not_ind = np.setxor1d(ia, np.where(tst == i))
#        m2 = np.mean(tst[not_ind])
#        tst[n] = m2

###############################################################################
#####    Experimental smoothing

#psi_norm[i,:], ayc_te_dat[i,:]
#
#a = np.where(np.isnan(ayc_te_dat[chk_t]) == False)[0]
#
#x = psi_norm[chk_t][a]
#y = ayc_te_dat[chk_t][a]
#x_pred = np.linspace(np.amin(psi_sorted), np.amax(psi_sorted),
#                     10*len(ayc_te_x))
#
#gp = GaussianProcessRegressor()
#gp.fit(np.atleast_2d(x).T, y)
#y_pred = gp.predict(np.atleast_2d(x_pred).T)
#
#a = find_closest(x_pred, np.amin(x))
#b = find_closest(x_pred, np.amax(x))
#
#x_pred = x_pred[range(a, b)]
#y_pred = y_pred[range(a, b)]

#psi_fin = []
#Te_fin = []
#
#def find_closest(data, v):
#    return (np.abs(data-v)).argmin()
#
#for i in range(len(tme)):
#    a = np.where(np.isnan(Te_sorted[i]) == False)[0]
#    
#    x = psi_sorted[i][a]
#    y = Te_sorted[i][a]
#    x_pred = np.linspace(np.amin(psi_sorted), np.amax(psi_sorted),
#                         40*len(ayc_te_x))
#    
#    gp = GaussianProcessRegressor()
#    gp.fit(np.atleast_2d(x).T, y)
#    y_pred = gp.predict(np.atleast_2d(x_pred).T)
#    
#    a = find_closest(x_pred, np.amin(x))
#    b = find_closest(x_pred, np.amax(x))
#    
#    psi_fin.append(x_pred[range(a, b)])
#    Te_fin.append(y_pred[range(a, b)])
#
#psi_fin = np.array(psi_fin)
#Te_fin = np.array(Te_fin)

###############################################################################
#####    interpolation round 2

'''
fun stuff will happen here maybe if gp doesn't work
'''

###############################################################################
#####    plotting

def R_ch():
    plt.figure()
    #plt.plot(ayc_r_t, ayc_r_dat[:,20])
    plt.contourf(ayc_r_dat.T, 11)
    plt.colorbar()
    plt.ylabel('channel number')
    plt.xlabel('time (index value)')
    plt.title('Radius (m)')

#plt.figure()
#plt.contourf(tme, psi_rng, Te_sorted.T, 33)
#plt.colorbar()
#plt.xlabel('time (s)')
#plt.ylabel('$\Psi$', rotation=0)
#plt.title('$T_{e}$')

def te_channel():
    plt.figure()
    plt.contourf(tme, psi_ch, ayc_te_dat.T, 33)
    plt.colorbar()
    plt.xlabel('time (s)')
    plt.ylabel('channel number')
    plt.title('$T_{e}$')

def psi_channel():
    plt.figure()
    plt.contourf(tme, psi_ch, psi_dat_z0_new.T, 33)
    plt.colorbar()
    plt.xlabel('time (s)')
    plt.ylabel('channel number')
    plt.title('psi_new at z=0')
    #
    #plt.figure()
    #plt.contourf(psi_t, psi_x, psi_dat_z0.T, 33)
    #plt.colorbar()
    #plt.xlabel('time (s)')
    #plt.ylabel('radial position (m)')
    #plt.title('psi at z=0')
    
    plt.figure()
    plt.plot(psi_ch, psi_dat_z0_new[chk_t])
    plt.xlabel('channel number')
    plt.ylabel('$\Psi$', rotation=0)
    plt.title('for time index {}'.format(chk_t))

def te_x():
#    plt.figure()
#    plt.plot(psi_ch, ayc_te_dat[chk_t])
#    plt.xlabel('channel number')
#    plt.ylabel('$T_{e}$', rotation=0)
#    plt.title('for time index {}'.format(chk_t))
    
    plt.figure()
    plt.plot(ayc_r_dat[chk_t], ayc_te_dat[chk_t])
    plt.xlabel('radial position (m)')
    plt.ylabel('$T_{e}$', rotation=0)
    plt.title('for time index {}'.format(chk_t))
    
#    plt.figure()
#    plt.plot(ayc_r_x, ayc_r_dat[chk_t])
#    plt.xlabel('radial index (channel number)')
#    plt.ylabel('radial position (m)')

def te_psi():
    plt.figure()
    #plt.plot(psi_sort, T_e_sort)
    #plt.plot(psi_fin[chk], Te_fin[chk], 'r')
    plt.plot(psi_dat_z0_new[chk_t,:], ayc_te_dat[chk_t,:], 'g')
    plt.xlabel('$\Psi$')
    plt.ylabel('$T_{e}$', rotation=0)
    plt.title('for time index {}'.format(chk_t))

def te_multi_psi(strt=0 , stp=tme.shape[0]):
    plt.figure()
    plt.xlabel('$\Psi$')
    plt.ylabel('$T_{e}$', rotation=0)
    if strt == stp-1:
        plt.title('time step: {}'.format(strt))
    else:
        plt.title('time steps: {} to {}'.format(strt, stp-1))
    for i in range(strt, stp):
        plt.plot(psi_fin[i], Te[i])
        #plt.plot(psi_peaks[i], Te_peaks[i], 'go')

def psi_rz():
    plt.figure()
    plt.contour(efm_grid_r, efm_grid_z, psi_dat[chk_t,:,:], 33)
    plt.axis('equal')
    plt.colorbar()
    plt.ylabel('z (m)')
    plt.xlabel('radial position (m)')
    plt.title('$\Psi (r,z)$')

def psi_interp_multi():
    plt.figure()
    plt.contourf(psi_dat_z0, 33)
    plt.colorbar()
    plt.xlabel('channel number')
    plt.ylabel('time index')
    plt.title('$\Psi (z=0)$')
    
    plt.figure()
    plt.contourf(psi_dat_z0_new, 33)
    plt.colorbar()
    plt.xlabel('channel number')
    plt.ylabel('time index')
    plt.title('$\Psi (z=0)$ interpolated')

def psi_interp_test():
#    plt.figure()
#    plt.plot(psi_t, psi_dat_z0[:,chk], 'bx', ms=mrk)
#    plt.plot(ayc_r_t, test, 'ro', ms=mrk)
#    plt.xlabel('time (s)')
#    plt.ylabel('$\Psi$', rotation=0)
#    plt.title('for time index {}'.format(chk))
#    
    plt.figure()
    plt.plot(psi_x, psi_dat_z0[chk_t], 'bx', ms=mrk)
    plt.plot(ayc_r_dat[chk_t], test2, 'ro', ms=mrk)
    plt.xlabel('radial position (m)')
    plt.ylabel('$\Psi$', rotation=0)
    plt.title('for time index {}'.format(chk_t))

te_multi_psi()
te_multi_psi(25,26)

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

#for i in range(0, len(tme)):
#    fig = plt.figure()
#    plt.plot(psi_fin[i], Te[i])
#    plt.plot(psi_peaks[i], Te_peaks[i], 'go')
#    plt.xlabel('$\Psi$')
#    plt.ylabel('$T_{e}$', rotation=0)
#    plt.title('time step {}'.format(i))
#    print(i)
#    plt.savefig(str(i).zfill(4) +'.png')
#    plt.close(fig)

#plt.show()


###############################################################################
#####    slice plot

#from matplotlib.collections import PolyCollection
#from matplotlib.colors import colorConverter
#import matplotlib.colors
#
#def cc(arg):
#    return colorConverter.to_rgba(arg, alpha=0.6)
#def dd(arg):
#    return colorConverter.to_rgba(arg, alpha=0.1)
#
##matplotlib.rcParams.update({'font.size': 22})
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#
#cmap = plt.cm.Wistia
#norm = matplotlib.colors.Normalize(vmin=0, vmax=8000)
#
#verts = []
#for i in range(len(tme)):
#    a = list(zip(psi_norm[i,:], ayc_te_dat[i,:]))
#    verts.append(a)
#    
#zs = tme
#
#poly = PolyCollection(verts, edgecolors = dd('cornflowerblue'),
#                      facecolors = cmap(norm(np.sum(psi_rng, axis = 0))))
#poly.set_alpha(0.9)
#ax.add_collection3d(poly, zs=zs, zdir='z')
#
#ax.set_xlabel('$\Psi$')
#ax.set_xlim3d(np.amin(psi_sorted), np.amax(psi_sorted))
#ax.set_ylabel('$T_{e}$')
#ax.set_ylim3d(np.nanmin(Te_sorted), np.nanmax(Te_sorted))
#ax.set_zlabel('time (s)')
#ax.set_zlim3d(np.amin(tme), np.amax(tme))
#ax.grid('off')
#ax.set_title("Electron Temperature (eV)")

plt.show()












