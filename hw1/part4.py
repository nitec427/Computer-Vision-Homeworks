import math

import cv2
import numpy as np

cos = np.cos
sin = np.sin
angle = math.pi/3 # Rotation angle
matrix = np.asarray([[cos(angle),sin(angle)], [-sin(angle),cos(angle)]]) # Matrix to rotate
def part4():
    cover = cv2.imread("cover.jpeg")
    height, width, c = cover.shape
    centered = np.zeros((cover.shape),dtype="uint8")

    #This part is taken from the lecture slides and slightly rewritten
    centerx =   width // 2
    centery = height // 2
    for j in range(0,height):
        for k in range(0,width):
            new_coords = np.matmul(matrix, np.asarray([k - centerx, j - centery]))
            new_coords = new_coords + np.asarray([centerx, centery])
            if new_coords[0] < height and new_coords[1] < width:
                centered[int(new_coords[0]), int(new_coords[1]), :] = cover[j,k,:]
    cv2.imshow("Rotation from the Center", centered)
    #Rotate from the top left
    height, width, c = cover.shape
    top_left = np.zeros((cover.shape), dtype="uint8")

    for j in range(0,height):
        for k in range(0,width):
            new_coords = np.matmul(matrix, np.asarray([k,j]))
            if new_coords[0] < height and new_coords[1] < width:
                top_left[int(new_coords[0]), int(new_coords[1]), :] = cover[j,k,:]
    cv2.imshow("Corner Rotation",top_left)
    cv2.waitKey(0)


    #REMARK
    """
    I outut the resulting images by cv.imshow(). As can be seen from the photos, for rotation from the corner case, 
    the pixel lost for each edge is same. Though, it is not the case for top left rotation. For top left rotation the image
    is dispositined differently for each side.
    """
part4()