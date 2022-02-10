import cv2
import numpy as np
import moviepy.editor as mpye
import glob

# İsmail Çetin
# 150180065
# Computer Vision | Homework 4
img_arr = []  # define list to write your images
img_arr_color = []
result_arr = []
# Save the images into the image list
for file in glob.glob("*.png"):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)  # Read the img in gray
    img_color = cv2.imread(file)
    img_arr.append(img)
    img_arr_color.append(img_color)


# Sigma is assumed to be 1px for all the cases
def find_gradient_x(x, y, index): # Use index to find which frame we are dealing with
    img_cur = img_arr[index]
    img_next = img_arr[index + 1]
    Ix = (int(img_cur[x + 1, y]) + int(img_next[x + 1, y]) + int(img_cur[x + 1, y + 1]) + int(img_next[x + 1, y + 1]))
    Ix -= (int(img_cur[x, y]) + int(img_next[x, y]) + int(img_cur[x, y + 1]) + int(img_next[x, y + 1]))
    return Ix / 4


def find_gradient_y(x, y, index):
    img_cur = img_arr[index]
    img_next = img_arr[index + 1]
    Iy = (int(img_cur[x, y + 1]) + int(img_next[x, y + 1]) + int(img_cur[x + 1, y + 1]) + int(img_next[x + 1, y + 1]))
    Iy -= (int(img_cur[x, y]) + int(img_next[x, y]) + int(img_cur[x + 1, y]) + int(img_next[x + 1, y]))
    return Iy / 4


def find_gradient_time(x, y, index):
    img_cur = img_arr[index]
    img_next = img_arr[index + 1]
    It = int(img_next[x, y]) + int(img_next[x + 1, y]) + int(img_next[x, y + 1]) + int(img_next[x + 1, y + 1])
    It -= int((img_cur[x, y]) + int(img_cur[x + 1, y]) + int(img_cur[x, y + 1]) + int(img_cur[x + 1, y + 1]))
    return It / 4
# Gradient finding functins are coded according to the lecture slides. I needed to change img px value to int to avoid unsigned subtraction overflow

def lucas_kanade(position, index, window_size,threshold):
    # in the lucas kanade first construct the A matrix by using points
    x, y = position # Get current position
    # Create numpy arrays and then assign corresponding values
    A = np.empty((window_size * window_size, 2))
    b = np.empty((window_size * window_size, 1))
    q = window_size // 2 #2
    # I used the center of the frames to obtain vectors which center is at the center point of the local window
    for i in range(-q ,q + 1):
        for j in range(-q, q+1):
            A[(i+2) * window_size + (j+2), 0] = find_gradient_x(x + i, y + j, index)
            A[(i+2) * window_size + (j+2), 1] = find_gradient_y(x + i, y + j, index)
            b[(i+2) * window_size + (j+2)] = -find_gradient_time(x + i, y + j, index) #-It value
    # A and b matrices are created
    A_transpose = np.transpose(A) # To find aT * a multiplication
    eig_matrix = np.matmul(A_transpose, A) # Matrix to extract eigenvalues
    eigval1, eigval2 = np.linalg.eigvals(eig_matrix)
    if eigval1 > threshold and eigval2 > threshold:
        if (max((eigval1, eigval2)) / min((eigval1, eigval2))) < 2.3: #Eigenvalues must be close to each other
            return np.matmul(np.linalg.pinv(A), b) # Some matrices are non-singular, so I needed to use pseudo-inverse method from numpy library
        return np.array((0, 0)) # If eigenvalue conditions do not suffice then send empty arrays
    return np.array((0, 0))


# Now, for every scene (until n-1) take the scene and compute every possible OF for every 5 * 5 window
w_size = 5  # window size
for i in range(1, len(img_arr) - 1):
    color_frame = img_arr_color[i]
    height, width = img_arr[i].shape
    # The center of the image is used to obtain necessary arrows.
    for a in range(3, height - w_size, w_size):
        for b in range(3, width - w_size, w_size):
            u, v = lucas_kanade((a, b), i, w_size, 30) # Lucas kanade with threshold 30
            if abs(u) + abs(v) > 0.63: # u, v must satisfy magnitude condition and if so add vector
                color_frame = cv2.arrowedLine(color_frame, (b + int(v * 13), a + int(u * 13)), (b, a), (255, 69, 0), 1)
    result_arr.append(color_frame)
    print(f"{i}. is OK")
clip = mpye.ImageSequenceClip(result_arr, fps=25)
clip.write_videofile("result.mp4", codec="libx264")
