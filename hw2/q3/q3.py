import pyautogui
import time
import numpy as np
import cv2
# İsmail Çetin
# 150180065
time.sleep(5)

# This function traverse the square with area 24px * 24px, and if it locates any corner
# write it as a corner. (by incrementing counter)
def count_corner(img,size):
    height, width = img.shape
    counter = 0
    for i in range(0, height, size):
        for j in range(0, width,size):
            filter = img[i:i+size, j:j+size]
            if np.sum(filter) > 255 * 9: # I randomly find this value to (trial and error)
                img[i:i+size, j:j+size] = 127 # paint these square area to gray (for seeing easily)
                counter+=1
    return counter
# Push to button according to corner count
def push_button(count):
    if count == 3:
        pyautogui.press('a')
    elif count == 4:
        pyautogui.press('s')
    elif count > 4 and count < 10 :
        pyautogui.press('f')
    else:
        if count!=0:
            pyautogui.press('d')

i = 0
while True:
    # First take ss
    ss = pyautogui.screenshot()
    i += 1
    ss.save(f'test.png')
    img = cv2.imread(f"test.png",1)
    cropped_img = img[810:1080, 770:1130] # I cropped out the necessary part
    # now edge detection
    imgGray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    ret, imgThresh = cv2.threshold(imgGray, 180,255, cv2.THRESH_BINARY) # Delete the
    # rectangular area below
    imgBlur = cv2.GaussianBlur(imgThresh, (3,5), 1) # Use gaussian blur to eliminiate some faulty points
    dst = cv2.cornerHarris(imgBlur, 2, 1, 0.02) # I used this method by looking OpenCV documentation
    dst = cv2.dilate(dst,None) # Also this is provided there

    dst[dst>0.01*dst.max()]=255 #Locate the corner points
    dst = np.uint8(dst) # I converted dst to uint8 in order to eliminate when hexagonal creates 3 corners in the image
    y,x = np.nonzero(dst)
    corner_count = 0
    # Eliminate faulty corner counts
    if(len(x) > 0):
        if min(x) < 150:
            corner_count = count_corner(dst, 24) # traverse with a square whose area is 24*24 px
    else:
        corner_count = 0
    #Corner count is obtained
    push_button(corner_count)
    time.sleep(.25) # wait some time

# Obtained by using pyautogui.position function at the beginning of solving question
#Point(x=869, y=810) upper left
# Point(x=869, y=1077) lower left
# Point(x=1050, y=1077) lower right
# Point(x=1051, y=807) upper right