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
from metrics import mse, psnr, ssim

# Load data
image_dir = "./datasets/yalefaces/subject*.*"
files = glob.glob(image_dir)
images = np.asarray([np.asarray(Image.open(f)) for f in files])
# pdb.set_trace()
n_images, m, n = np.shape(images)

# # Vectorize images for PCA
# images_1d = images.reshape((n_images, m * n))  # vectorize for PCA

# # Fit model to data using PCA
# n_components = 20
# pca = PCA(n_components=n_components)
# X_transformed_pca = pca.fit_transform(images_1d)
# pca_pc = pca.components_
# pca_evr = pca.explained_variance_ratio_
# pca_sv = pca.singular_values_

# # Plot singular values and explained variance ratio for PCA
# plot_singular_values(pca_sv, "pca_sv.png", "Singular Values for PCA")
# plot_explained_variance_ratio(
#     pca_evr, "pca_evr.png", "Explained Variance Ratio for PCA"
# )

# # Plot reconstructed images using PCA
# rand_idx = np.random.choice(np.arange(n_images), 10)
# fig, axs = plt.subplots(
#     nrows=2, ncols=5, figsize=(10, 4), subplot_kw={"xticks": [], "yticks": []}
# )
# axs = axs.reshape(-1)
# for i, idx in enumerate(rand_idx):
#     axs[i].imshow(
#         pca.inverse_transform(X_transformed_pca[idx]).reshape((m, n)), cmap="gray"
#     )
#     axs[i].set_title(str(idx))

# plt.suptitle("Reconstructions with PCA (n_pcs={})".format(n_components))
# plt.savefig(
#     "./figures/recon_pca_{}.png".format(n_components), bbox_inches="tight", dpi=200
# )
# plt.show()

# Fit model to data using SSR-2D-PCA
scale = 40
# ssrU, ssrV, ssrS, ssrE = ssrr2dpca(images, scale=scale, UV_file="./cache/UV.npz")
ssrU, ssrV, ssrS, ssrE = ssrr2dpca(images, scale=scale)
# Reconstruct
X_transformed_ssrr2dpca = np.einsum("ij,ljk->lik", ssrU, ssrS.dot(ssrV.T)) - ssrE

# Visualize principal components
plt.figure(1)
plt.imshow(X_transformed_ssrr2dpca[0], cmap="gray")
# plt.show()

plt.figure(2)
plt.imshow(images[0], cmap="gray")
# plt.show()

plt.figure(3)
plt.imshow(ssrE[0], cmap="gray")
# plt.show()

plt.figure(4)
plt.imshow(X_transformed_ssrr2dpca[0] + ssrE[0], cmap="gray")
plt.show()

# Visualize reconstructed images
