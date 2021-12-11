import numpy as np
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio


def ssim(img, img_noise):
    n_images = img.shape[0]
    x = np.zeros((n_images,))
    for i in range(n_images):
        x[i] = structural_similarity(img[i], img_noise[i])
    return x


def mse():
    return 0


def psnr():
    return 0
