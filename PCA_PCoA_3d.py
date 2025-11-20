# -*- coding: utf-8 -*-
# https://stackoverflow.com/questions/65241847/how-to-plot-3d-pca-with-different-colors
# https://stackoverflow.com/questions/32930086/matplotlib-3d-graph-giving-different-plot-when-used-inside-a-function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# %matplotlib notebook
data = load_breast_cancer() # 유방암 데이터
X = data.data
y = data.target
# print(X, "\r\n", y)
# sc = StandardScaler()

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

#
# PCA
#
pca = PCA(n_components=3)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

ex_variance = np.var(X_pca, axis=0)
ex_variance_ratio = ex_variance / np.sum(ex_variance)
ex_variance_ratio

Xax = X_pca[:, 0]
Yax = X_pca[:, 1]
Zax = X_pca[:, 2]

cdict = {0: 'red', 1: 'green'}
labl = {0: 'Malignant', 1: 'Benign'} # 악성의/비악성의
marker = {0: '*', 1: 'o'}
alpha = {0: .3, 1: .5}

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection='3d')

fig.patch.set_facecolor('white')
for l in np.unique(y):
    ix = np.where(y == l)
    ax.scatter(Xax[ix], Yax[ix], Zax[ix], c=cdict[l], s=40,
               label=labl[l], marker=marker[l], alpha=alpha[l])
# for loop ends
ax.set_xlabel("PCA#1", fontsize=12) # First Principal Component
ax.set_ylabel("PCA#2", fontsize=12) # Second Principal Component
ax.set_zlabel("PCA#3", fontsize=12) # Third Principal Component

ax.legend()
plt.show()

#
# MDS
#
mds = MDS(n_components=3)
mds.fit(X_scaled)
X_mds = mds.fit_transform(X_scaled)

ex_variance = np.var(X_mds, axis=0)
ex_variance_ratio = ex_variance / np.sum(ex_variance)
ex_variance_ratio

Xax = X_mds[:, 0]
Yax = X_mds[:, 1]
Zax = X_mds[:, 2]

cdict = {0: 'red', 1: 'green'}
labl = {0: 'Malignant', 1: 'Benign'} # 악성의/비악성의
marker = {0: '*', 1: 'o'}
alpha = {0: .3, 1: .5}

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection='3d')

fig.patch.set_facecolor('white')
for l in np.unique(y):
    ix = np.where(y == l)
    ax.scatter(Xax[ix], Yax[ix], Zax[ix], c=cdict[l], s=40,
               label=labl[l], marker=marker[l], alpha=alpha[l])
# for loop ends
ax.set_xlabel("PCoA#1", fontsize=12) # First Principal Component
ax.set_ylabel("PCoA#2", fontsize=12) # Second Principal Component
ax.set_zlabel("PCoA#3", fontsize=12) # Third Principal Component

ax.legend()
plt.show()