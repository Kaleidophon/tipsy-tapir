import numpy as np


def plm_kernel(sigma, func):
    k = np.zeros(2 * sigma + 1)
    for j in range(sigma):
        k[j] = k[-(j + 1)] = func(sigma, j)  # Symmetry
    k[sigma] = 1  # Always one at actual position
    return k


def k_passage(sigma=50):
    return plm_kernel(sigma, func=lambda sigma, j: 1 if abs(sigma - j) <= sigma else 0)


def k_gaussian(sigma=50):
    return plm_kernel(sigma, func=lambda sigma, j: np.exp(-((sigma-j)**2) / (2 * (sigma ** 2))))


def k_triangle(sigma=50):
    return plm_kernel(sigma, func=lambda sigma, j: 1 - (abs(sigma - j) / sigma) if abs(sigma - j) <= sigma else 0)


def k_circle(sigma=50):
    return plm_kernel(
        sigma, func=lambda sigma, j: np.sqrt(1 - ((abs(sigma - j) / sigma)) ** 2) if abs(i - j) <= sigma else 0
    )