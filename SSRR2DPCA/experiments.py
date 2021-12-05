"""Experiments for structured sparsity regularized robust 2D principal
component analysis.
"""
import os
import glob
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, SparsePCA
from PIL import Image
from SSRR2DPCA import SSRR2DPCA, ssrr2dpca
from plotter import *
from metrics import psnr, ssim


# Load data
image_dir = "./datasets/yalefaces/subject*.*"
files = glob.glob(image_dir)
images = np.asarray([np.asarray(Image.open(f)) for f in files])
n_images, m, n = np.shape(images)

# Vectorize images for PCA
images_1d = images.reshape((n_images, m * n))  # 1D vectors for PCA, sparse PCA

# Fit model to data using PCA
n_components = 10
pca = PCA(n_components=n_components)
X_r_pca = pca.fit_transform(images_1d)
pca_pc = pca.components_
pca_evr = pca.explained_variance_ratio_
pca_sv = pca.singular_values_

# Plot and compare results
plot_singular_values(pca_sv, "pca_sv.png", "Singular Values for PCA")
plot_explained_variance_ratio(
    pca_evr, "pca_evr.png", "Explained Variance Ratio for PCA"
)

# Fit model to data using SSR-2D-PCA
scale = 40
ssrU, ssrV, ssrS, ssrE = ssrr2dpca(images, scale=scale, UV_file="./cache/UV.npz")

# Visualize principal components
plt.imshow(ssrU[0])
plt.show()