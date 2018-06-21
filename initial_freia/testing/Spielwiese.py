
# coding: utf-8

# ## Imports:
# You can either install pandas and xarray with the existing data, or you find a different way to generate a matrix with columns (x, time, data) for all pairs (all coordinates)
# sklearn.neighbors is the module for the k-nearest neighbor (knn)

# In[1]:


import numpy as np
import pickle
import scipy
import matplotlib.pyplot as plt
from sklearn import neighbors
import xarray as xr
import pandas as pd


# ## Open datasets:
# This I took directly from your code

# In[2]:


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


# In[3]:


plt.figure(1)
plt.contour(psi['x'],psi['time'],psi['data'], 50)
plt.colorbar()
plt.ylabel('time (s)')
plt.xlabel('radial position (normalised?)')
plt.title('psi at z=0')


# ## Make array with data
# As I said above, either use this approach with xarray or create *mat1* in a different way. I checked the reshaping of d so that the dataset agrees with the plot above (see the plot below)

# In[4]:


X = np.array(psi['x'])
T = np.array(psi['time'])
d = np.array(psi['data']).reshape((len(T),len(X)))


# In[5]:


ds = xr.DataArray(data = d.T, coords = [X,T], name = 'input')
mat1 = ds.to_dataframe().reset_index().as_matrix()


# ## Fit regression model
# n_neighbors gives the number of nearest neighbors to consider (should be something between 1-5 for you) and weights='distance' means that inverse distance weighting is applied. You need to first declare the model and then fit it with one matrix of features (for you X and T) and a vector with data (here d)

# In[6]:


n_neighbors = 1
knn = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')


# In[7]:


knn.fit(mat1[:,:-1],mat1[:,-1])


# ## Predict interpolated data
# I changed the X_new to linear spacing between 0 and 1 with 130 steps, so that the range of X_new is equal to the range of X.

# In[8]:


X_new = np.linspace(0,1,130)
T_new = np.array(ayc_r['time'])


# Again, either use this approach with pandas or find another way of creating *mat2* with columns (X, T, d) - d is a vector of zeros and filled during prediction.

# In[9]:


# BOTTOM LINE: make list of new locations
d_new = np.zeros((len(X_new),len(T_new)))
ds_new = xr.DataArray(data = d_new, coords = [X_new,T_new], name = 'output')
mat2 = ds_new.to_dataframe().reset_index().as_matrix()


# Here the values for the interpolated (x, time) pairs is found using inverse distance weighting from the *n_neighbors* closest points.

# In[10]:


mat2[:,-1] = knn.predict(mat2[:,:-1])


# ## Plotting
# See original dataset (top) and interpolated data (bottom)

# In[11]:


ds2=pd.DataFrame(mat2,columns=['X','T','output']).set_index(['X','T']).to_xarray()


# In[12]:


ds.plot(x='dim_0', y='dim_1')
plt.title('original')
plt.show()


# In[13]:


ds2.output.plot(x='X', y='T')
plt.title('interpolated')
plt.show()

