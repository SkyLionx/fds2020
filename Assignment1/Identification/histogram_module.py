import numpy as np
from numpy import histogram as hist


# Add the Filtering folder, to import the gauss_module.py file, where gaussderiv is defined (needed for dxdy_hist)
import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
filteringpath = os.path.join(parentdir, 'Filtering')
sys.path.insert(0, filteringpath)

import gauss_module

#  compute histogram of image intensities, histogram should be normalized so that sum of all values equals 1
#  assume that image intensity varies between 0 and 255
#
#  img_gray - input image in grayscale format
#  num_bins - number of bins in the histogram
def normalized_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

    hists = np.zeros((num_bins))
    bins = np.array([val * (255 / num_bins) for val in range(num_bins + 1)])

    for pixel in img_gray.flat:
        index = int(pixel // (255 / num_bins))
        hists[index] += 1

    # normalization
    hists /= hists.sum()

    return hists, bins


#  Compute the *joint* histogram for each color channel in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^3
#
#  E.g. hists[0,9,5] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
#       - their B values fall in bin 5
def rgb_hist(img_color_double, num_bins):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'

    # Define a 3D histogram  with "num_bins^3" number of entries
    hists = np.zeros((num_bins, num_bins, num_bins))

    linear_img = img_color_double.reshape(
        img_color_double.shape[0] * img_color_double.shape[1], 3)

    # Loop for each pixel i in the image
    for i in range(img_color_double.shape[0]*img_color_double.shape[1]):
        # Increment the histogram bin which corresponds to the R,G,B value of the pixel i
        rgb = linear_img[i]
        rIndex = int(rgb[0] // (255 / num_bins))
        gIndex = int(rgb[1] // (255 / num_bins))
        bIndex = int(rgb[2] // (255 / num_bins))
        hists[rIndex, gIndex, bIndex] += 1

    # Normalize the histogram such that its integral (sum) is equal 1
    hists /= hists.sum()

    # Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)
    return hists


#  Compute the *joint* histogram for the R and G color channels in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^2
#
#  E.g. hists[0,9] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
def rg_hist(img_color_double, num_bins):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'

    # Define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))

    linear_img = img_color_double.reshape(
        img_color_double.shape[0] * img_color_double.shape[1], 3)

    for i in range(img_color_double.shape[0] * img_color_double.shape[1]):
        rgb = linear_img[i]
        rIndex = int(rgb[0] // (255 / num_bins))
        gIndex = int(rgb[1] // (255 / num_bins))
        hists[rIndex, gIndex] += 1

    hists /= hists.sum()

    # Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)

    return hists


def get_bin_index(value, num_bins, values_range=(0, 255)):
    r_min, r_max = values_range
    intervals = r_max - r_min
    values_per_bin = intervals / num_bins
    index = int((value - r_min) / values_per_bin)
    if index >= num_bins:
        return num_bins-1
    return index

#  Compute the *joint* histogram of Gaussian partial derivatives of the image in x and y direction
#  Set sigma to 3.0 and cap the range of derivative values is in the range [-6, 6]
#  The histogram should be normalized so that sum of all values equals 1
#
#  img_gray - input gray value image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
#
#  Note: you may use the function gaussderiv from the Filtering exercise (gauss_module.py)
def dxdy_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

    sigma = 3.0

    # Define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))

    imgDx, imgDy = gauss_module.gaussderiv(img_gray, sigma)

    linear_imgDx = imgDx.ravel()
    linear_imgDy = imgDy.ravel()


    # Clamp values between -6 and 6
    linear_imgDx[linear_imgDx > 6] = 6
    linear_imgDx[linear_imgDx < -6] = -6

    linear_imgDy[linear_imgDy > 6] = 6
    linear_imgDy[linear_imgDy < -6] = -6

    for i in range(img_gray.shape[0] * img_gray.shape[1]):
        dxPixel = linear_imgDx[i]
        dyPixel = linear_imgDy[i]
        dxIndex = get_bin_index(dxPixel, num_bins, values_range=(-6, 6))
        dyIndex = get_bin_index(dyPixel, num_bins, values_range=(-6, 6))
        hists[dxIndex, dyIndex] += 1

    hists /= hists.sum()

    # Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)
    return hists


def is_grayvalue_hist(hist_name):
    if hist_name == 'grayvalue' or hist_name == 'dxdy':
        return True
    elif hist_name == 'rgb' or hist_name == 'rg':
        return False
    else:
        assert False, 'unknown histogram type'


def get_hist_by_name(img, num_bins_gray, hist_name):
    if hist_name == 'grayvalue':
        return normalized_hist(img, num_bins_gray)
    elif hist_name == 'rgb':
        return rgb_hist(img, num_bins_gray)
    elif hist_name == 'rg':
        return rg_hist(img, num_bins_gray)
    elif hist_name == 'dxdy':
        return dxdy_hist(img, num_bins_gray)
    else:
        assert False, 'unknown distance: %s' % hist_name
