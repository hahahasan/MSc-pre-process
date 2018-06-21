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
#####    useful functions

# finds array index corresponding to array value closest to some value
def find_closest(data, v):
	return (np.abs(data-v)).argmin()

def nan_finder(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def nan_interp(self,arr):
    '''
    insert a 2d array with length len(tme)
    '''
    y = np.copy(arr)
    for i in range(len(self.tme)):
       nans, x = nan_finder(y[i])
       y[i][nans]= np.interp(x(nans), x(~nans), y[i][~nans])
    return y

###############################################################################
#####    do some stuff

class load_data:

    def __init__(self):
        
    ###########################################################################
    #####    load the pickles
    
        os.chdir('./pickles')
        
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
            
        ###########################################################################
        #####    let's declare some arrays
        
        # 2-dimensional (r,z) at different times
        # psi as function of radius in m
        self.psi_x = psi_rz_28819['x']
        # psi as function of time in s
        self.psi_t = psi_rz_28819['time']
        # value of psi at specific radius, time, and z
        psi_dat = psi_rz_28819['data']
        
        # psi grid values
        # same as psi_x
        self.efm_grid_r = efm_grid_r_28819['data'].squeeze()
        # would be psi_z i.e. z location of flux
        efm_grid_z = efm_grid_z_28819['data'].squeeze()
        
        # channel number vs radius
        # x is the channel number
        self.ayc_r_x = ayc_r_28819['x']
        # t is the time
        self.ayc_r_t = ayc_r_28819['time']
        # dat is the radius correesponding to specific channel number at some time t
        self.ayc_r_dat = ayc_r_28819['data']
        
        # electron temperature data given as a function of time and channel number
        # channel number
        self.ayc_te_x = ayc_te_28819['x']
        # time
        self.ayc_te_t = ayc_te_28819['time']
        # T_e at channel number and time
        self.ayc_te_dat = ayc_te_28819['data']
        
        # magnetic axis and psi boundary
        # magnetic axis
        self.mag_axis = efm_psi_axis_28819['data']
        self.mag_axis_tme = efm_psi_axis_28819['time']
        # psi boundary
        self.psi_bound = efm_psi_boundary_28819['data']
        self.psi_bound_tme = efm_psi_boundary_28819['time']
        
        # arbitrary value to check slices of data
        self.chk_t = 44
        self.chk_x = 44
        # marker size for plotting
        self.mrk = 2
        
        self.tme = self.ayc_te_t
        self.psi_ch = np.linspace(1, len(self.ayc_te_x), len(self.ayc_te_x))
        
        # there is wobble in the ayc_r_dat that means the channel number as a
        # function of the radial positon changes with time by a very small amount
        # defining psi_rng basically ignores these tiny perturbations
        self.psi_rng = np.linspace(np.amin(self.ayc_r_dat),
                                   np.amax(self.ayc_r_dat), 
                                   self.ayc_r_dat.shape[1])
        
        self.psi_N_rng = np.linspace(-1, 1, 200)
    
        # find the z=0 coordinate
        z0_axis = np.where(efm_grid_z == 0)[0][0]
        # define psi only along the z0 axis
        self.psi_dat_z0 = psi_dat[:,z0_axis,:]
        
        def nan_interp(self,arr):
            '''
            insert a 2d array with length len(tme)
            '''
            y = np.copy(arr)
            for i in range(len(self.tme)):
               nans, x = nan_finder(y[i])
               y[i][nans]= np.interp(x(nans), x(~nans), y[i][~nans])
            return y
    

###########################################################################
#####    interpolation

# interpolation of psi data so that it corresponds to the same channel 
# number and time as the electron temperature data perform 2 seperate 1d
# interpolations. Not ideal but was struggling with 2d interpolation. 
# Have some fun trying scikit learn and knn approach :D

# time axis for psi_boundary data needs to be interpolated to coincide 
# with tme
    def mag(self):
        bound_interp = interpolate.interp1d(self.psi_bound_tme, self.psi_bound,
                                            kind='cubic', 
                                            fill_value='extrapolate')
        self.psi_boundary = bound_interp(self.tme)
        
        # mag axis data interpolated
        mag_axis_interp_tmp = interpolate.interp1d(self.mag_axis_tme,
                                                   self.mag_axis,
                                                   kind='cubic',
                                                   fill_value='extrapolate')
        self.mag_axis_interp = mag_axis_interp_tmp(self.tme)
        
        return 
    

    
    def interp_2d(self):
        f = interpolate.interp2d(self.psi_x, self.psi_t, self.psi_dat_z0,
                                 kind='cubic', fill_value='extrapolate')
        f_interp = f(self.psi_rng, self.tme)
        return f_interp
        
    psi_dat_z0_new2 = interp_2d()


###############################################################################
#####    Normalisation

def normalisation(self):
    mag_ax_psi = []
    mag_ax = []
    norm_ind = []
    for i in range(len(self.tme)):
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
    y = nan_interp(ayc_te_dat)
    for i in range(len(tme)):
        f = interpolate.interp1d(psi_N[i], y[i], kind='nearest',
                                 fill_value='extrapolate')
        y2.append(f(psi_N_rng))
        #print(y2[i])
    return y2
    
Te_interp = same_psi()

###############################################################################
#####    polynomial features

def bivar_polyfit(deg=5):
    x = psi_N_rng
    y = tme
    X1, Y1 = np.meshgrid(x, y, copy=False)
    B = np.array(Te_interp).flatten()
    
    X = X1.flatten()
    Y = Y1.flatten()
    
    aa = np.array(list(zip(X, Y)))
    
    poly = PolynomialFeatures(deg)
    X_fit = poly.fit_transform(aa)
    clf = linear_model.LinearRegression()
    clf.fit(X_fit, B)
    
    predict_x = np.concatenate((X1.reshape(-1,1), Y1.reshape(-1,1)), axis=1)
    pred_x = poly.fit_transform(predict_x)
    pred_y = clf.predict(pred_x)
    
    return pred_y.reshape(X1.shape)

te_fit = abs(bivar_polyfit(6))

diff = []
for i in range(len(tme)):
    for j in range(len(psi_N_rng)):
        diff.append(abs(te_fit[i][j] - Te_interp[i][j]))
print('average  difference between model and data:', np.mean(diff))

def fit_compare():
    fig = plt.figure(figsize=(11, 7))
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X1, Y1, te_fit, cmap="jet", lw=0.5,
                             rstride=1, cstride=1, alpha=0.7)
    fig.colorbar(surf1, ax=ax1)
    ax2 = fig.add_subplot(122, projection='3d')
    cs = ax2.plot_surface(X1, Y1, Te_interp, cmap="jet", lw=0.5,
                          rstride=1, cstride=1)
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
        title='T_e fitted with 6th order bivariate polynomial',
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

def te_multi_psi(strt=0 , stp=len(tme)-1):
    plt.figure()
    plt.xlabel('$\Psi$')
    plt.ylabel('$T_{e}$ (eV)')
    tmp1 = strt
    tmp2 = stp
    plt.title('time: {} to {}'.format(round(tme[tmp1], 3),
              round(tme[tmp2], 3)))
    for i in range(tmp1, tmp2):
        plt.plot(psi_N_rng, Te_interp[i])
        #plt.plot(psi_N[i], ayc_te_dat[i])
        #plt.plot(psi_peaks[i], Te_peaks[i], 'go')
        
    plt.figure()
    plt.xlabel('$\Psi$')
    plt.ylabel('$T_{e}$', rotation=0)
    tmp1 = strt
    tmp2 = stp
    plt.title('time: {} to {}'.format(round(tme[tmp1], 3),
              round(tme[tmp2], 3)))
    for i in range(tmp1, tmp2):
        plt.plot(psi_N_rng, Te_interp[i])
        #plt.plot(psi_N[i], ayc_te_dat[i])
        #plt.plot(psi_peaks[i], Te_peaks[i], 'go')
    plt.show()

Te_vs_psiN()

if __name__=="main":
    load_data




