# %% markdown
# ## Import packages.
# import packages
import numpy as np
import time
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib import gridspec

import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter

from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler
# %%
# define plot properties
from cycler import cycler
import matplotlib.cm as cm

from matplotlib import rcParams
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable

def rgb(r,g,b):
    return (float(r)/256.,float(g)/256.,float(b)/256.)

cb2 = [rgb(31,120,180), rgb(255,127,0), rgb(51,160,44), rgb(227,26,28), \
       rgb(166,206,227), rgb(253,191,111), rgb(178,223,138), rgb(251,154,153)]

rcParams['figure.figsize'] = (9,7.5)
#rcParams['figure.dpi'] = 300

rcParams['lines.linewidth'] = 1

rcParams['axes.prop_cycle'] = cycler('color', cb2)
rcParams['axes.facecolor'] = 'white'
rcParams['axes.grid'] = False

rcParams['patch.facecolor'] = cb2[0]
rcParams['patch.edgecolor'] = 'white'

#rcParams['font.family'] = 'Bitstream Vera Sans'
rcParams['font.size'] = 23
rcParams['font.weight'] = 300

# %% markdown
# ## GALAH age distribution.
#
# > Plot the distribution.
# %%
# restore data
temp = np.load("ages.npz")
print(temp.files)
age = temp["age"].astype("float32")
age = age.reshape(age.size,1)
print(age.shape)

plt.hist(age, bins=100);
# %%
# restore data
temp = np.load("weighted_ages.npz")
print(temp.files)
age = temp["age"].astype("float32")[::300]
age = age.reshape(age.size,1)
print(age.shape)

plt.hist(age, bins=100);
# %% markdown
# > Plot $\sigma$(age) as a function of age.
# %%
# color coded in log scale
plt.hexbin(age, eage, bins='log')
# %% markdown
# > Plot results.
# %%
# setup figure
plt.figure(figsize=[15,22]);

# the latent space
temp = np.load("real_nvp_results.npz")
z = temp["z1"]
print(z.shape)
plt.subplot(321)
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.scatter(z[:, 0], z[:, 1], s=0.1)
plt.title(r'$z = f(X)$')

z = temp["z2"]
print(z.shape)
plt.subplot(322)
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.scatter(z[:, 0], z[:, 1], s=0.1)
plt.title(r'$z \sim p(z)$')

#-----------------------------------------------------------------------------------------
# the 2D feature space
x = temp["x1"]
print(x.shape)
plt.subplot(323)
plt.xlim([0,20])
plt.ylim([0,20])
plt.scatter(x[:, 0], x[:, 1], c='r', s=0.1)
plt.title(r'$X \sim p(X)$ [2D]')

x = temp["x2"]
print(x.shape)
plt.subplot(324)
plt.xlim([0,20])
plt.ylim([0,20])
plt.scatter(x[:, 0, 0], x[:, 0, 1], c='r', s=0.1)
plt.title(r'$X = g(z)$ [2D]')

#-----------------------------------------------------------------------------------------
# 1D feature histogram
plt.subplot(325)

x = temp["x1"]
plt.hist(x[:, 0], bins=100, range=[0,20])

x = temp["x2"]
plt.hist(x[:, :, 0], alpha=0.5, bins=100, range=[0,20])

plt.title(r'$X \sim p(X)$ [1D]')

# %%
