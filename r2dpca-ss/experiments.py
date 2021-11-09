"""Experiments for structured sparsity regularized robust 2D principal
component analysis.
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, SparsePCA
from PIL import Image
import SSR2DPCA
from plotter import *

# Load data
dir = '../datasets/yalefaces/subject*.*'
files = glob.glob(dir)
n = len(files)
images = np.asarray([np.asarray(Image.open(f)) for f in files])
ny, nx, n_images = np.shape(images)

# Preprocessing
images_1d = np.reshape(images, (nx * ny, n_images)).T   # 1D vectors for PCA, sparse PCA
images_2d = np.reshape(images, (n_images, ny, nx))

# Fit model to data using PCA
n_components = 10
pca = PCA(n_components=n_components)
X_r_pca = pca.fit_transform(images_1d)
pca_pc = pca.components_
pca_evr = pca.explained_variance_ratio_
pca_sv = pca.singular_values_

# # Fit model to data using sparse PCA
# spca = SparsePCA(n_components=n_components)
# X_r_spca = spca.fit_transform(images_1d)
# spca_pc = spca.components_
# spca_var = spca.explained_variance_
# spca_sv = spca.singular_values_

# Fit model to data using SSR-2D-PCA
r = 10
c = 8
ssr2dpca = SSR2DPCA.SSR2DPCA(n_components_x=r, n_components_y=c)
X_r_ssr2dpca = ssr2dpca.fit_transform(images_2d)
ssr2dpca_pc = pca.components_
ssr2dpca_var = pca.explained_variance_
ssr2dpca_s = pca.singular_values_

# Plot and compare results
plot_singular_values(pca_sv, 'pca_sv.png', 'Singular Values for PCA')
plot_explained_variance_ratio(pca_evr, 'pca_evr.png', 'Explained Variance Ratio for PCA')
