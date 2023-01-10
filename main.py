import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time
from function import *


def logtime(L):
    L.append(time.monotonic())

Ltime = [time.monotonic()]

#image reading
target_file = "images\\dragons.png"
image = mpimg.imread(target_file)
logtime(Ltime)

#To gray
gray_image = imageToGrayNormalize(image)
logtime(Ltime)

#Blurr
blurred_image = blurrImage(gray_image)
logtime(Ltime)

#Finding the gradient of the image
gradient, theta = findGradient(blurred_image)
logtime(Ltime)

gradient_nonmax_supress = non_max_suppression(gradient,theta)
nx,ny = gradient.shape
logtime(Ltime)

#thresholding
threshold_high = 0.5
threshold_low = 0.05
edges = thresholding(gradient_nonmax_supress, threshold_high, threshold_low)
logtime(Ltime)

#histeresis
edges_histeresis = histeresis(edges)
logtime(Ltime)


for i in range(1,len(Ltime)):
    print(Ltime[i]-Ltime[i-1])

## plot
fig1, axs = plt.subplots(2, 2)
axs[0,0].imshow(blurred_image,"gray")
axs[0,1].imshow(gradient,"gray")
axs[1,0].imshow(gradient_nonmax_supress,"gray")
axs[1,1].imshow(edges,"gray")

fig2, ax2 = plt.subplots(1, 1)
ax2.imshow(edges_histeresis,"gray")
plt.show()

