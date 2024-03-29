# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2



"""
Gaussian function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian values Gx computed at the indexes x
"""
def gauss(sigma):
    sigma = int(sigma)
    x = np.arange(-3 * sigma, 3 * sigma + 1)
    Gx = [1 / (math.sqrt(2 * math.pi) * sigma) * math.exp(- ((i**2) / (2*(sigma**2)))) for i in x]
    Gx = np.array(Gx)

    return Gx, x





"""
Implement a 2D Gaussian filter, leveraging the previous gauss.
Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
Leverage the separability of Gaussian filtering
Input: image, sigma (standard deviation)
Output: smoothed image
"""
# def gaussianfilter_slow(img, sigma):
#     """Slower version of gaussianfilter"""
#     import time
#     start = time.time()
#     Gx, _ = gauss(sigma)
#     kernel = conv2(Gx, Gx.T)
#     smooth_img = conv2(img, kernel, "same")
#     print("Ho impiegato", time.time() - start, "sec")

#     return smooth_img

def gaussianfilter(img, sigma):
    # import time
    # start = time.time()
    Gx, _ = gauss(sigma)
    Gx = Gx.reshape((1, -1))
    smooth_img = conv2(img, Gx, "same")
    smooth_img = conv2(smooth_img, Gx.T, "same")
    # print("Ho impiegato", time.time() - start, "sec")

    return smooth_img



"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""
def gaussdx(sigma):
    sigma = int(sigma)
    x = np.arange(-3 * sigma, 3 * sigma + 1)
    Dx = [- (1 / (math.sqrt(2 * math.pi) * (sigma**3))) * i * math.exp(- ((i**2) / (2*(sigma**2)))) for i in x]

    Dx = np.array(Dx)

    return Dx, x



def gaussderiv(img, sigma):
    Dx, _ = gaussdx(sigma)
    kernel = Dx.reshape((1, Dx.shape[0]))

    imgDx = conv2(img, kernel, "same")
    imgDy = conv2(img, kernel.T, "same")
    
    return imgDx, imgDy

