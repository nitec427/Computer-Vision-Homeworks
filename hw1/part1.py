import numpy as np
import os
import cv2
import moviepy.editor as mpy

background = cv2.imread("Malibu.jpg")
cv2.imshow("Background Image Window", background)

# Resize the background image according to cat frames. Cat frames have a shape of 360, 640w

back_height = background.shape[0]
back_width = background.shape[1]

ratio = 360 / back_height

background = cv2.resize(background, (int(back_width * ratio),360))
#Read every iamage inside the cat folder

# First count files in cat folder

cat_list = os.listdir("cat")
ct_files = len(cat_list)
images_list = list()
for i in range(ct_files):
    cat_image = cv2.imread(f"cat/cat_{i}.png")
    foreground = np.logical_or(cat_image[:,:,1] < 180, cat_image[:,:,0]>150) # Extract the pixels containing cat info
    nonzero_x, nonzero_y = np.nonzero(foreground) # Foreground variable (matrix) helps us to find entries which carries
    #x and y values
    nonzero_cat_values = cat_image[nonzero_x, nonzero_y, :]

    ###############    FLIPPED CATS     ############
    flipped = cv2.flip(cat_image, 1)
    flipped_fg = np.logical_or(flipped[:,:,1] < 180, flipped[:,:,0]>150)
    nz_x, nz_y = np.nonzero(flipped_fg)
    flipped_nz = flipped[nz_x, nz_y, :]

    ########## COPY THE IMAGES IN FRONT OF BG IMAGE ######
    new_frame = background.copy()
    new_frame[nz_x , nz_y + new_frame.shape[1] - cat_image.shape[1], :] = flipped_nz
    new_frame[nonzero_x, nonzero_y, :] = nonzero_cat_values
    new_frame = new_frame[:, :, [2,1,0]]
    images_list.append(new_frame)

clip = mpy.ImageSequenceClip(images_list, fps = 25)
audio = mpy.AudioFileClip("selfcontrol_part.wav").set_duration(clip.duration)
clip = clip.set_audio(audioclip=audio)
clip.write_videofile("part1_video.mp4", codec = "libx264")
cv2.waitKey(0)