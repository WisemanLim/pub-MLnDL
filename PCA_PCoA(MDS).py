#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

# loading the iris data set
iris = datasets.load_iris()
X = pd.DataFrame(iris.data)   # features data frame
X.columns = ['sepal length', 'sepal width', 'petal length', 'petal width']
y = pd.DataFrame(iris.target)  # target data frame
y.columns = ['species']
target_names = iris.target_names

#
# HISTOGRAMS
#

# histogram of the petal length
plt.hist(X['petal length'])
plt.show()

# histogram with 20 bins
plt.hist(X['petal length'], 20)
plt.show()


#
# BOXPLOTS
#

# box plot of the petal length
X.boxplot(column='petal length')
plt.show()

# box plot of the petal length & width
X.boxplot(column=['petal length', 'petal width'])
plt.show()

# box plot of all the features
X.boxplot()
plt.show()

# notched box boxplots
X.boxplot(notch=True)
plt.show()

# describing various statsitics
print(X.describe())


#
# SCATTER PLOTS
#

# plotting petal width vs length (as a method)
X.plot.scatter(x='petal width', y='petal length')
plt.show()

# plotting petal width vs length (as a function)
plt.scatter(X['petal width'], X['petal length'])
plt.show()

# scatter plot matrix
pd.plotting.scatter_matrix(X)
plt.show()

# scatter plot matrix with different colors for species
pd.plotting.scatter_matrix(X, c=y['species'])
plt.show()



#
# IMPORTING SCIKIT-LEARN LIBRARIES
#
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


#
# NORMALIZING IRIS DATA FEATURES
#

# defining an normalization object
normData = StandardScaler()

# fitting the data to the normalization object
normData.fit(X)

# applying the normalization transformation to the data
X_norm = normData.transform(X)

# using fit_transform method
X_norm = StandardScaler().fit_transform(X)


#
# PCA
#

# applying PCA
pca = PCA()  # creating a PCA transformation ojbect
X_pc = pca.fit_transform(X_norm) # fit the data, and get PCs

# proportion of the variance explained
print(pca.explained_variance_ratio_)

# plotting the first 2 principal compnonents
plt.scatter(X_pc[:,0], X_pc[:,1], c=y['species'])
plt.show()


#
# MDS
#

from sklearn.manifold import MDS

# applying MDS
mds = MDS()
X_mds = mds.fit_transform(X_norm)

# plotting the MDS-transformed coordinates
plt.scatter(X_mds[:,0], X_mds[:,1], c=y['species'])
plt.show()


#
# PARALLEL COORDINATES PLOT
#

Xy = pd.concat([X,y], axis=1)
pd.plotting.parallel_coordinates(Xy,'species')
plt.show()


#
# CORRELATION
#

import scipy as sp

r_pearson, p_pearson = sp.stats.pearsonr(X['sepal length'], X['sepal width'])
r_spearman, p_spearman = sp.stats.spearmanr(X['sepal length'], X['sepal width'])
r_kendall, p_kendall = sp.stats.kendalltau(X['sepal length'], X['sepal width'])
print(r_pearson, p_pearson)
print(r_spearman, p_spearman)
print(r_kendall, p_kendall)