import moviepy.video.io.VideoFileClip as mpy
import numpy as np
import cv2
from numba import jit
import moviepy.editor as mpye

vid = mpy.VideoFileClip("shapes_video.mp4")
frame_count = vid.reader.nframes
video_fps = vid.fps
# I use numba for fast computation
@jit(nopython=True, fastmath=True)
def obtain_frame(frame, size):
    width, height = frame.shape
    new_frame = np.zeros(frame.shape) #create new frame
    k = size // 2 #for for loop
    for i in range(k, width - k):
        for j in range(k, height-k):
            med_arr = frame[i-k:i+k+1, j-k: j+k+1] #obtain media filter 3x3 size
            new_frame[i,j] = np.median(med_arr) #get the median of the filter and set it to current px
    return new_frame
@jit(nopython=True, fastmath=True)
def apply_mean(frame, size):
    width, height = frame.shape
    new_frame = np.zeros(frame.shape)  # create new frame
    k = size // 2
    for i in range(k, width - k):
        for j in range(k, height - k):
            mean_arr = frame[i - k:i + k + 1, j - k: j + k + 1]
            cum_sum = np.sum(mean_arr)
            size_sqr = size * size
            px_intensity = cum_sum / size_sqr
            new_frame[i, j] = px_intensity
    return new_frame
# I tried to code subtract frames but I wasn't able to do it. So, I couldn't finish part 1.2
@jit(nopython=True, fastmath=True)
def subtract_frames(cur, prev):
    r,c,b = cur.shape
    res = np.zeros(cur.shape)
    for i in range(r):
        for j in range(c):
            for k in range(b):
                if prev[i,j,k] > 250:
                    prev[i,j,k] = 0
                if cur[i,j,k] < prev[i,j,k]:
                    res[i,j,k] = 0
                else:
                    res[i,j,k] = cur[i,j,k] - prev[i,j,k]
    return res
images_list = []
for i in range(frame_count):
    frame = vid.get_frame(i * 1.0 / video_fps)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame(x,y) = intensity
    width, height ,c = frame.shape #obtain the shape
    frame_smoothed = obtain_frame(gray_frame, 3) # Frame smoothed with median filter
    new_frame = cv2.merge((frame_smoothed, frame_smoothed, frame_smoothed)) # merge the frames so as to put them into ImageSequenceClip

    # Subtract the second frame from the first one

    # Hold prev_frame in variable
    images_list.append(new_frame)
    print(i)
    # Obtain 5x5 median filter for every pixel of the frame
    # starting from (1,1) to (width - 1, height - 1)
    # each time compute new value of the pixels
    # write them to new frame
clip = mpye.ImageSequenceClip(images_list,fps = video_fps)
clip.write_videofile("result.mp4",codec="libx264")
