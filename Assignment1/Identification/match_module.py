import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import histogram_module
import dist_module

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray



# model_images - list of file names of model images
# query_images - list of file names of query images
#
# dist_type - string which specifies distance type:  'chi2', 'l2', 'intersect'
# hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'
#
# note: use functions 'get_dist_by_name', 'get_hist_by_name' and 'is_grayvalue_hist' to obtain 
#       handles to distance and histogram functions, and to find out whether histogram function 
#       expects grayvalue or color image

def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):

    hist_isgray = histogram_module.is_grayvalue_hist(hist_type)
    
    model_hists = compute_histograms(model_images, hist_type, hist_isgray, num_bins)
    query_hists = compute_histograms(query_images, hist_type, hist_isgray, num_bins)
    
    D = np.zeros((len(model_images), len(query_images)))
    
    # For each pair (model, query)
    for y in range(len(model_images)):
        for x in range(len(query_images)):
            model_hist = model_hists[y]
            query_hist = query_hists[x]
            # Compute the distance
            D[y, x] = dist_module.get_dist_by_name(model_hist, query_hist, dist_type)
    
    # Find the best_match for each query by taking the index of the minimum value in the i-th column
    best_match = np.array([np.argmin(D[:, i]) for i in range(len(query_images))])

    return best_match, D



def compute_histograms(image_list, hist_type, hist_isgray, num_bins):
    image_hist = []

    # Compute histogram for each image and add it at the bottom of image_hist
    for filename in image_list:
        img = np.array(Image.open(filename)).astype("double")
        # Convert the image in grayvalue
        if hist_isgray:
            img = rgb2gray(img)
        hist = histogram_module.get_hist_by_name(img, num_bins, hist_type)
        # We ignore the bins and get only the histogram
        hist = hist[0] if isinstance(hist, tuple) else hist
        image_hist.append(hist)

    return image_hist



# For each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.
#
# Note: use the previously implemented function 'find_best_match'
# Note: use subplot command to show all the images in the same Python figure, one row per query image

def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):
    plt.figure()

    num_nearest = 5  # show the top-5 neighbors
    
    _, D = find_best_match(model_images, query_images, dist_type, hist_type, num_bins)

    nrows = len(query_images)
    ncols = num_nearest + 1

    for y in range(nrows):
        # Get the indexes sorted by distance of the y-th column
        indexes = np.argsort(D[:, y])
        for x in range(ncols):
            index = y * ncols + x + 1
            plt.subplot(nrows, ncols, index)
            plt.axis("off")
            # If it's the first column
            if x == 0:
                # Plot the query image
                plt.title("Q" + str(y))
                img = np.array(Image.open(query_images[y]))
            else:
                # Plot the x-th minus 1 best match
                min_index = indexes[x - 1]
                min_val = D[min_index, y]
                plt.title("M{:.2}".format(min_val))
                img = np.array(Image.open(model_images[min_index]))
            plt.imshow(img)
    
    plt.show()
