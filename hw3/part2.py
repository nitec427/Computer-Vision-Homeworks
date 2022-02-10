import cv2
import numpy as np
from scipy.io import loadmat
import glob
import os

# İsmail Çetin
# 150180065

# same thresholding from part 1
def calc_precision(img_canny, ground_truth):
    t_p = np.count_nonzero(np.logical_and((img_canny == 255), (ground_truth == 255)))
    f_p = np.count_nonzero(np.logical_and((img_canny == 255), (ground_truth == 0)))
    return t_p / (t_p + f_p)  # True positives / total positives

i = 0
prec = 0

for file in glob.glob("*.mat"): # For each matrix file get the file
    f_name = file.split('.')[0] # obtain file_name to find conv. network corrrespondence
    mat = loadmat(file) # load the matrix
    ph_sum = np.zeros_like(mat['groundTruth'][0][0][0][0][1])
    photo_num = mat['groundTruth'][0].shape[0]
    for j in range(photo_num):
        ph_sum = ph_sum + mat['groundTruth'][0][j][0][0][1] # ground truth is obtained and saved to ph_sum
    # Now compare network result with ground Truth results
    ph_sum[ph_sum > 0] = 255 # Thresholding for ground truth
    i+=1 # how many images are there (200 for our case)
    # Obtain test photo
    test_img = cv2.imread(f"{f_name}.png",0) # used 0 to read in grayscale
    test_img[test_img > 120] = 255 #thresholding
    prec += calc_precision(test_img, ph_sum) # accumulate precision for each imaage to prec

print(prec / (i+1)) # precision avg is printed