import pickle
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np

with open ("stylegan3-t-ffhq-1024x1024.pkl", "rb") as f:
    a = pickle.load(f)
#Unpack the GAN network downloaded from the official NVIDIA website.
gan = a["G_ema"]
#StyleGAN contains both the Generator and the Discriminator. We only need
#the Generator to generate the images.
gan.eval()
#In PyTorch, every network has to be set as 'train' or 'eval'. To evaluate
#an input via the network, we should set it as eval.
for param in gan.arameters():
    param.requires_grad = False
#When training the network, the parameter change is done via the gradients.
#Setting that none of the parameters need gradients will speed up our
#process.
z1 = torch.randn(1, 512)
z2 = torch.randn(1, 512)
# The StyleGAN generator takes a vector of size (512) to generate an image.
img1 = gan(z1, 0).numpy().squeeze()
img2 = gan(z2, 0).numpy().squeeze()
#Obtain the image from the generator.
img1 = np.transpose(img1, (1,2,0))
img2 = np.transpose(img2, (1,2,0))

#Initially, the output of the network is like (RGB Channel, Width, Height).
#We should change the order of the channels to make it look like an OpenCV image.
img1[img1 > 1] = 1
img1[img1 < -1] = -1
img1 = 255*(img1 + 1) / 2

img2[img2 > 1] = 1
img2[img2 < -1] = -1
img2 = 255*(img2 + 1) / 2
#The network outputs images with float values, centered at zero. Thus, we
#should make the values between [0 - 255).

cv2.imwrite('test1.png', img1[:,:,[2,1,0]])
cv2.imwrite('test2.png', img1[:,:,[2,1,0]])