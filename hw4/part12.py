import cv2
import numpy as np
import moviepy.editor as mpye
import glob

# İsmail Çetin
# 150180065
# Computer Vision | Homework 4
from numba import jit

img_arr = []  # define list to write your images
img_arr_color = []
still_img = cv2.imread("00000.png", cv2.IMREAD_GRAYSCALE)
threshold  = 80
result_arr = []
i = 0 #counter for finding still bground
kernel = np.ones((7,7), np.uint8) # kernel for dilation
erode_ker = np.ones((2,2), np.uint8) # erosion kernel
for file in glob.glob("*.png"):
    cur_img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)  # Read the img in gray
    img_color = cv2.imread(file)
    i+=1
    # For every image subtract the img from still img
    height, width = cur_img.shape
    if i % 11 == 0:
        still_img = cur_img # For every 11 iteration update your background estimate
        continue # And skip the reference frame (otherwise sync is lost)
    #For every location x,y subtract and do thresholding
    for x in range(height):
        for y in range(width):
            if abs(int(cur_img[x][y]) - int(still_img[x][y])) > threshold:
                img_color[x,y,:] = (255,255,255)
            else:
                img_color[x,y,:] = (0,0,0)
    print(f"{i}. is OK")
    erode = cv2.erode(img_color, erode_ker,iterations=1)
    dilated = cv2.dilate(erode, kernel, iterations=1)
    img_arr_color.append(dilated)
clip = mpye.ImageSequenceClip(img_arr_color, fps=25)
clip.write_videofile("result.mp4", codec="libx264")
