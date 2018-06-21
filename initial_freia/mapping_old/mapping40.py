#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 16:32:56 2018

@author: hm1234
"""

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
    
###############################################################################
#####    load the pickles

os.chdir('./pickles')

with open('vt_ss_27873.pickle', 'rb') as handle:
    vt_ss_27873 = pickle.load(handle)
with open('ti_ss_27873.pickle', 'rb') as handle:
    ti_ss_27873 = pickle.load(handle)

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
    
with open('efm_psi_axis_28819.pickle', 'rb') as handle:
    efm_psi_axis_28819 = pickle.load(handle)
with open('efm_psi_boundary_28819.pickle', 'rb') as handle:
    efm_psi_boundary_28819 = pickle.load(handle)
with open('efm_psi_axis_30417.pickle', 'rb') as handle:
    efm_psi_axis_30417 = pickle.load(handle)
with open('efm_psi_boundary_30417.pickle', 'rb') as handle:
    efm_psi_boundary_30417 = pickle.load(handle) 
    
os.chdir('../')
    
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
te_x = ayc_te_28819['x']
# time
te_t = ayc_te_28819['time']
# T_e at channel number and time
te_dat = ayc_te_28819['data']

# electron density data given as a function of time and channel number
# channel number
ne_x = ayc_ne_28819['x']
# time
ne_t = ayc_ne_28819['time']
# n_e at channel number and time
ne_dat = ayc_ne_28819['data']

# ion temperature data given as a function of time and major radius
# majpr radius (m)
ti_x = ti_ss_27873['x']
# time
ti_t = ti_ss_27873['time']
# T_i at radius and time
ti_dat = ti_ss_27873['data']

# velocity data given as a function of time and major radius
# majr radius (m)
vt_x = vt_ss_27873['x']
# time (s)
vt_t = vt_ss_27873['time']
# Vt at radius and time
vt_dat = vt_ss_27873['data']

# magnetic axis and psi boundary
# magnetic axis
mag_axis = efm_psi_axis_28819['data']
mag_axis_tme = efm_psi_axis_28819['time']
# psi boundary
psi_bound = efm_psi_boundary_28819['data']
psi_bound_tme = efm_psi_boundary_28819['time']


# arbitrary value to check slices of data
chk_t = 44
chk_x = 44
# marker size for plotting
mrk = 2

tme = te_t

# there is wobble in the ayc_r_dat that means the channel number as a fnction
# of the radial positon changes with time by a very small amount
# defining psi_rng basically ignores these tiny perturbations
psi_rng = np.linspace(np.amin(ayc_r_dat), np.amax(ayc_r_dat), 
                      ayc_r_dat.shape[1])

psi_N_rng = np.linspace(-1, 1, 200)

###############################################################################
#####    useful functions

# finds array index corresponding to array value closest to some value
def find_closest(data, v):
	return (np.abs(data-v)).argmin()

def nan_finder(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def nan_interp(arr):
    '''
    insert a 2d array with length len(tme)
    '''
    y = np.copy(arr)
    for i in range(len(tme)):
       nans, x = nan_finder(y[i])
       y[i][nans]= np.interp(x(nans), x(~nans), y[i][~nans])
    return y

###############################################################################
#####    do some stuff

# find the z=0 coordinate
z0_axis = np.where(efm_grid_z == 0)[0][0]
# define psi only along the z0 axis
psi_dat_z0 = psi_dat[:,z0_axis,:]

###############################################################################
#####    interpolation

# interpolation of psi data so that it corresponds to the same channel number
# and time as the electron temperature data
# perform 2 seperate 1d interpolations. Not ideal but was struggling with 2d
# interpolation. Have some fun trying scikit learn and knn approach :D


# time axis for psi_boundary data needs to be interpolated to coincide with tme
bound_interp = interpolate.interp1d(psi_bound_tme, psi_bound, kind='cubic',
                                    fill_value='extrapolate')
psi_boundary = bound_interp(tme)

# mag axis data interpolated
mag_axis_interp_tmp = interpolate.interp1d(mag_axis_tme, mag_axis,
                                           kind='cubic',
                                           fill_value='extrapolate')
mag_axis_interp = mag_axis_interp_tmp(tme)

def interp_1d():
    psi_t_interp = []
    for i in range(0, len(psi_x)):
        interp_test = interpolate.interp1d(psi_t, psi_dat_z0[:,i],
                                           kind='cubic',
                                           fill_value='extrapolate')
        psi_t_interp.append(interp_test(ayc_r_t))   
    psi_t_interp = np.array(psi_t_interp).T
    # psi_t_interp is psi but with same time values as T_e data
    
    psi_x_interp = []
    for i in range(0, psi_t_interp.shape[0]):
        interp_test = interpolate.interp1d(psi_x, psi_t_interp[i],
                                           kind='cubic',
                                           fill_value='extrapolate')
        psi_x_interp.append(interp_test(ayc_r_dat[i]))  
    psi_x_interp = np.array(psi_x_interp)
    return psi_x_interp

# psi_t_interp is psi but with same channel number values as T_e data
# since the time data is also the same the outputted array should be the
# correct shape
psi_dat_z0_new = interp_1d()

def interp_2d():
    f = interpolate.interp2d(psi_x, psi_t, psi_dat_z0, kind='cubic',
                             fill_value='extrapolate')
    f_interp = f(psi_rng, tme)
    return f_interp

psi_dat_z0_new2 = interp_2d()

###############################################################################
#####    Normalisation

mag_ax_psi = []
mag_ax = []
norm_ind = []
for i in range(len(tme)):
    a = np.where(psi_dat_z0_new2[i] == np.amax(psi_dat_z0_new2[i]))[0][0]
    mag_ax_psi.append(psi_dat_z0_new2[i][a])
    mag_ax.append(ayc_r_dat[i][a])
    norm_ind.append(a)
mag_ax_psi = np.array(mag_ax_psi)
mag_ax_r = np.array(mag_ax)
norm_ind = np.array(norm_ind)

psi_N = []
for i in range(len(tme)):
    psi_N_temp = ( (psi_dat_z0_new2[i] - mag_ax_psi[i]) /
                  (psi_boundary[i] - mag_ax_psi[i]) )
    psi_N_temp[norm_ind[i]:,] = -psi_N_temp[norm_ind[i]:,]
    psi_N.append(-psi_N_temp)
psi_N = np.array(psi_N)

###############################################################################
#####    get the same psi values for all Te


def same_psi():   
    y2 = []
    y = nan_interp(te_dat)
    for i in range(len(tme)):
        f = interpolate.interp1d(psi_N[i], y[i], kind='nearest',
                                 fill_value='extrapolate')
        y2.append(f(psi_N_rng))
    return y2
Te_interp = same_psi()

###############################################################################
#####    suface fitting

x = psi_N_rng
y = tme
X1, Y1 = np.meshgrid(x, y, copy=False)
Z1 = np.array(Te_interp)

###############################################################################
#####    polynomial features

def bivar_polyfit(deg=5):
    # psi and time coordinates of the data
    x = psi_N_rng
    y = tme
    # gotta meshgrid it so that it forms a nice grid for Te data to sit
    X1, Y1 = np.meshgrid(x, y, copy=False)
    # the Te data flattened so that it linear regression can be performed on it
    B = np.array(Te_interp).flatten()
    # gotta flatten the grid too so that coords and Te match up
    X = X1.flatten()
    Y = Y1.flatten()
    # zips the coords together for input into polynoial features function
    aa = np.array(list(zip(X, Y)))

    # generates polynomial features of any degree
    poly = PolynomialFeatures(deg)
    # 
    X_fit = poly.fit_transform(aa)
    clf = linear_model.LinearRegression()
    clf.fit(X_fit, B)
    
    predict_x = np.concatenate((X1.reshape(-1,1), Y1.reshape(-1,1)), axis=1)
    pred_x = poly.fit_transform(predict_x)
    pred_y = clf.predict(pred_x)
    
    return pred_y.reshape(X1.shape)

deg = 6
te_fit = abs(bivar_polyfit(deg))

diff = []
for i in range(len(tme)):
    for j in range(len(psi_N_rng)):
        diff.append(abs(te_fit[i][j] - Te_interp[i][j]))
mean_diff = np.mean(diff)
print('average of raw data=', np.mean(Te_interp))
print('average of surface fit=', np.mean(te_fit))
print('average difference between', mean_diff)

def compare_contour():
    fig = plt.figure(figsize=(11,7))
    c = 33
    
    ax1 = fig.add_subplot(121)
    cont1 = ax1.contourf(X1, Y1, te_fit, c)
    fig.colorbar(cont1, ax=ax1)
    ax1.set_title('fitted temperature (eV)')
    ax1.set_xlabel('$\Psi_{N}$')
    ax1.set_ylabel('time (s)')
    
    ax2 = fig.add_subplot(122)
    cont2 = ax2.contourf(psi_N_rng, tme, Te_interp, c)
    fig.colorbar(cont2, ax=ax2)
    ax2.set_title('raw temperature data (eV)')
    ax2.set_xlabel('$\Psi_{N}$')
compare_contour()

def fit_compare():
    fig = plt.figure(figsize=(11, 7))
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X1, Y1, te_fit, cmap="jet", lw=0.5,
                             rstride=1, cstride=1, alpha=0.7)
    fig.colorbar(surf1, ax=ax1)
    ax2 = fig.add_subplot(122, projection='3d')
    cs = ax2.plot_surface(X1, Y1, Z1, cmap="jet", lw=0.5, rstride=1, cstride=1)
    fig.colorbar(cs, ax=ax2)
    plt.show()

def fit_compare2():
    fig = plt.figure(figsize=(11, 7))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_surface(X1, Y1, te_fit, cmap="winter", lw=0.5,
                     rstride=1, cstride=1, alpha=0.8)
    ax1.plot_surface(X1, Y1, Te_interp, cmap="autumn", lw=0.5,
                     rstride=1, cstride=1, alpha=0.4)
    ax1.set_xlabel('$\Psi_{N}$')
    ax1.set_ylabel('time (s)')
    ax1.set_zlabel('$T_{e}$ (eV)')
    ax1.grid(False)
    plt.show()
    
def plotly_test():
    import plotly.plotly as py
    import plotly.graph_objs as go
    
    z1 = te_fit
    z2 = Te_interp
    
    data = [
        go.Surface(z=z1, x=psi_N_rng, y=tme, colorscale='Jet'),
        go.Surface(z=z2, x=psi_N_rng, y=tme, showscale=False, 
                   opacity=0.7, colorscale='autumn'),
    ]
    
    
    layout = go.Layout(
        width=800,
        height=700,
        autosize=False,
        title='f fitted with {}th order bivariate polynomial'.format(deg),
        scene=dict(
            xaxis=dict(
                title = 'Psi_N',
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                title = 'time(s)',
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            zaxis=dict(
                title = 'T_e (eV)',
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            aspectratio = dict( x=1, y=1, z=0.7 ),
            aspectmode = 'manual'
        )
    )
    #py.plot(data)
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig)

###############################################################################
#####    plotting

def Te_vs_psiN():
    plt.figure()
    plt.contourf(psi_N_rng, tme, Te_interp, 33)
    plt.colorbar()
    plt.xlabel('$\Psi_{N}$')
    plt.ylabel('t (s)')
    plt.title('$T_{e} (eV)$')
    plt.show()
    

def psi_plot(x=chk_x):
    plt.figure()
    plt.plot(tme, psi_dat_z0_new2[:,x], label='$\Psi$ at pos_index {}'
             .format(x))
    plt.plot(tme, mag_axis_interp, label='mag-axis from freia')
    plt.plot(tme, mag_ax_psi, label='mag-axis from code')
    plt.plot(tme, psi_boundary, label='boundary from freia')
    plt.fill_between(tme, mag_ax_psi, psi_boundary, alpha=0.3)
    plt.xlabel('time (s)')
    plt.ylabel('$\Psi$', rotation=0)
    plt.legend()
    plt.show()

def te_psi():
    plt.figure()
    #plt.plot(psi_sort, T_e_sort)
    #plt.plot(psi_fin[chk], Te_fin[chk], 'r')
    plt.plot(psi_dat_z0_new2[chk_t,:], te_dat[chk_t,:], 'g')
    plt.xlabel('$\Psi$')
    plt.ylabel('$T_{e}$', rotation=0)
    plt.title('for time index {}'.format(chk_t))
    plt.show()

def te_multi_psi(strt=0 , stp=len(tme)-1):     
    plt.figure()
    plt.xlabel('$\Psi$')
    plt.ylabel('$T_{e}$', rotation=0)
    tmp1 = strt
    tmp2 = stp
    plt.title('time: {} to {}'.format(round(tme[tmp1], 3),
              round(tme[tmp2], 3)))
    for i in range(tmp1, tmp2):
        plt.plot(psi_N_rng, Te_interp[i])
        #plt.plot(psi_N[i], te_dat[i])
        #plt.plot(psi_peaks[i], Te_peaks[i], 'go')
    plt.show()

def psi_rz(cont_type='contour', chk_t=chk_t, res=33):
    plt.figure()
    if cont_type == 'contour':
        plt.contour(efm_grid_r, efm_grid_z, psi_dat[chk_t,:,:], res)
    elif cont_type == 'contourf':
        plt.contourf(efm_grid_r, efm_grid_z, psi_dat[chk_t,:,:], res)
    else:
        plt.contour(efm_grid_r, efm_grid_z, psi_dat[chk_t,:,:], res)
    #plt.axis('equal')
    plt.colorbar()
    plt.ylabel('z (m)')
    plt.xlabel('radial position (m)')
    plt.title('$\Psi (r,z)$ at t = {}'.format(chk_t))
    plt.show()

def psi_interp_multi():
    plt.figure()
    plt.contourf(psi_dat_z0_new2, 33)
    plt.colorbar()
    plt.xlabel('channel number')
    plt.ylabel('time index')
    plt.title('$\Psi (z=0)$ interpolated (2D interp)')
    
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
    plt.title('$\Psi (z=0)$ interpolated (2 x 1D interp)')
    plt.show()


#Te_vs_psiN()
#psi_N_interps(44)
#psi_plot()
#te_multi_psi(40,45)
#poly2dfit_plot()
#Te_vs_psiN_3d()
#psi_rz()

###############################################################################
#####    Fancy animations

#os.chdir('./pics')
#for i in range(0, len(tme)):
#    fig = plt.figure()
#    te_multi_psi(i, i+1)
#    print(i)
#    plt.savefig(str(i).zfill(4) +'.png')
#    plt.close(fig)
#os.chdir('../')
#plt.show()