import numpy as np


def psnr(X, Xhat):
    """Calculate PSNR"""
    m, n = X.shape()
    return 10 * np.log((255 ** 2 * m * n) / (np.linalg.norm(X - Xhat, ord="fro") ** 2))


def ssim():
    """Calculate SSIM"""
    # TODO : Calculate SSIM
    return None
