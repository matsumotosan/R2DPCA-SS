"""Preprocessing to generate corrupted images"""
import os
import glob
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image

# Load data
image_dir = "./datasets/yalefaces/subject*.*"
files = glob.glob(image_dir)
images = np.asarray([np.asarray(Image.open(f)) for f in files])
n_images, m, n = np.shape(images)

# Plot 10 random images
nrows = 2
ncols = 5
fig, axs = plt.subplots(
    nrows=nrows, ncols=ncols, figsize=(8, 3), gridspec_kw={"wspace": 0, "hspace": 0}
)
axs = axs.reshape(-1)
for i, idx in enumerate(random.sample(range(n_images), nrows * ncols)):
    axs[i].imshow(images[idx], cmap="gray")
    axs[i].set_xticks([])
    axs[i].set_yticks([])

fig.tight_layout()
plt.savefig("./figures/yalefaces.png", bbox_inces="tight", dpi=200)
plt.show()

# Generate corrupted images
