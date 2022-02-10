import cv2
import numpy as np
import moviepy.editor as mpy
import os

#K = np.uint8(LUT[double(I)])
# np.dstack concatenates the matrices along a new dimension
def hist_matching(img, LUT1,LUT2,LUT3):
    mat1 = np.uint8(LUT1[img[:, :, 0]])
    mat2 = np.uint8(LUT2[img[:, :, 1]])
    mat3 = np.uint8(LUT3[img[:, :, 2]])
    return np.dstack((mat1,mat2,mat3))

# This part is obtained from the lecture notes
def LUT(pi,target):
    matrix = np.zeros((256,1))
    hist, bins = np.histogram(target.flatten(),256,[0,256])
    pj = hist.cumsum() / hist.cumsum().max()
    gj = 0
    for gi in range(255):
        while (pj[gj] < pi[gi] and gj < 255):
            gj = gj+1
        matrix[gi] = gj
    return matrix
hist1 = hist2 = hist3 = 0
for i in range(180):
    img = cv2.imread(f"cat/cat_{i}.png")
    fg = np.logical_or(img[:,:,1]<180, img[:,:,0]>150) #Pixels containing cat

    # Obtaining a histogram for each channel
    blue = img[:,:,0][fg]
    green = img[:,:,1][fg]
    red = img[:,:,2][fg]
    hist1 += np.histogram(blue.flatten(),256,[0,256])[0]
    hist2 += np.histogram(green.flatten(), 256, [0, 256])[0]
    hist3 += np.histogram(red.flatten(), 256, [0, 256])[0]

# Histogram equalization and normalization
#Take the average
hist1 = hist1 / 180
hist2 = hist2 / 180
hist3 = hist3 / 180

cdf1 = hist1.cumsum() / hist1.cumsum().max()
cdf2 = hist2.cumsum() / hist2.cumsum().max()
cdf3 = hist3.cumsum() / hist3.cumsum().max()

tar_get = cv2.imread("target.jpg")
l1 = LUT(cdf1,tar_get[:,:,0])
l2 = LUT(cdf2,tar_get[:,:,1])
l3 = LUT(cdf3,tar_get[:,:,2])

background = cv2.imread("Malibu.jpg")
back_height = background.shape[0]
back_width = background.shape[1]
ratio = 360 / back_height
background = cv2.resize(background,(int(back_width * ratio),360))

cat_list = os.listdir("cat")
ct_files = len(cat_list)
images_list = list()

for i in range(ct_files):
    cat_image = cv2.imread(f"cat/cat_{i}.png")
    foreground = np.logical_or(cat_image[:,:,1] < 180, cat_image[:,:,0]>150) # The pixels having cat image
    nonzero_x, nonzero_y = np.nonzero(foreground)
    nonzero_cat_values = cat_image[nonzero_x, nonzero_y, :]
    new_frame = background.copy()
    new_frame[nonzero_x, nonzero_y, :] = nonzero_cat_values

    # Cat, with the histogram matching
    new_cat = hist_matching(cat_image,l1,l2,l3)
    flipped = cv2.flip(new_cat,1)
    flipped_fg = np.logical_or(flipped[:,:,1] < 180, flipped[:,:,0]>150)
    nz_x, nz_y = np.nonzero(flipped_fg)
    nz_cat = flipped[nz_x, nz_y, :]
    new_frame[nz_x , nz_y + new_frame.shape[1] - cat_image.shape[1] , :] = nz_cat
    new_frame = new_frame[:, :, [2,1,0]]
    images_list.append(new_frame)

clip = mpy.ImageSequenceClip(images_list, fps = 25)
audio = mpy.AudioFileClip("selfcontrol_part.wav").set_duration(clip.duration)
clip = clip.set_audio(audioclip=audio)
clip.write_videofile("part2_video.mp4", codec = "libx264")
