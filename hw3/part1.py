import math

from scipy.io import loadmat
import numpy as np
import cv2
import glob
import os
# İsmail Çetin
# 150180065
path = 'C:/Users/nitec/Desktop/BSR_bsds500/images/test/images' # Path to save images



def calc_precision(img_canny, ground_truth):
    t_p = np.count_nonzero(np.logical_and((img_canny == 255), (ground_truth == 255))) # if both img_canny and ground_truth
    # find the value as 1 then this value is true positive
    f_p = np.count_nonzero(np.logical_and((img_canny == 255), (ground_truth == 0))) # if canny found but ground_truth
    # does not contain that pixel
    return t_p / (t_p + f_p)  # True positives / total positives


# Apply canny edge filter and make edges 255. Then find the total_positives and true_positives. True_positive (where
# both ground truth and canny returns 1). total_positives is all 1 s in canny detector
my_images = []
for file in glob.glob("*.mat"): # for each matrix file reach the file
    f_name = file.split('.')[0] # get its name for future use
    mat = loadmat(file) # load the file into numpy array
    ph_sum = np.zeros_like(mat['groundTruth'][0][0][0][0][1]) # create empty img which is in the same form with groundTruth
    photo_num = mat['groundTruth'][0].shape[0] # Count how many boundary is given in MATLAB file.
    for j in range(photo_num): # traverse the boundaries just for one image
        ph_sum = ph_sum + mat['groundTruth'][0][j][0][0][1] # sum and at the end obtain ground truth
    ph_sum[ph_sum > 0] = 255 # thresholding
    my_images.append(ph_sum) # add to python array to compare later

    # In the end we have ground truth correctly, compare this with the canny version later in the second for loop
    cv2.imwrite(os.path.join(path, f"{f_name}groundTruth.png"), ph_sum) # also save the image into images folder
    # which is in the same location with code

precision = 0
k = 0  # array counter for my_images
for file in glob.glob("*.jpg"):
    f_name = file.split('.')[0]
    img = cv2.imread(file, 0)
    upper, thresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Optimal threshold among my trials
    lower = .5 * upper
    blur = cv2.GaussianBlur(img, (7, 7), 0) # blur the image for better edge detection
    canny = cv2.Canny(blur, lower, upper) # apply canny filter
    canny[canny > 120] = 255 # do thresholding
    cv2.imwrite(os.path.join(path, f"{f_name}canny.png"), canny) # Save canny version of the img again to some folder

    precision += calc_precision(canny, my_images[k]) # Obtain the precision for the k'th image and accumulate the results
    k += 1 # holds how many images (200 for our case)

print("{:.4f}".format(precision / (k+1))) # This is the correct result until 4th decimal place.