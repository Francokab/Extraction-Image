import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time
from function import *


def logtime(name):
    Ltime.append(time.monotonic_ns())
    LtimeName.append(name)
LtimeName = ['start']
Ltime = [time.monotonic_ns()]

#image reading
target_file = "images\\medieval_house.jpg"
image = mpimg.imread(target_file)
logtime('image reading')

#To gray
gray_image = imageToGrayNormalize(image)
logtime("gray")

#Blurr
blurred_image = blurrImage(gray_image)
logtime("blurr")

#Finding the gradient of the image
gradient, theta = findGradient(blurred_image)
logtime("gradient")

#non max suppression
gradient_nonmax_supress = nonMaxSuppression(gradient,theta)
nx,ny = gradient.shape
logtime("non max suppression")

#thresholding
threshold_high = otsuMethod(gradient_nonmax_supress)
logtime("otsu method")
edges = thresholding(gradient_nonmax_supress, threshold_high, threshold_high/2)
logtime("thresholding")

#histeresis
edges_histeresis = histeresis(edges)
logtime("histeresis")

#debug time
for i in range(1,len(Ltime)):
    print((Ltime[i]-Ltime[i-1])*1e-9,"\t",LtimeName[i])

## plot
fig1, axs = plt.subplots(2, 2)
axs[0,0].imshow(blurred_image,"gray")
axs[0,1].imshow(gradient,"gray")
axs[1,0].imshow(gradient_nonmax_supress,"gray")
axs[1,1].imshow(edges,"gray")

fig2, ax2 = plt.subplots(1, 1)
ax2.imshow(edges_histeresis,"gray")
plt.show()

