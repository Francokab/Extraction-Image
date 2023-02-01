import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from function import *
from algo import *

#image reading
target_file = "images\\medieval_house.jpg"
image = mpimg.imread(target_file)

#To gray
gray_image = imageToGrayNormalize(image)


#Blurr
blurred_image = blurrImage(gray_image, sigma = 1.4)

#Finding the gradient of the image
gradient, theta = findGradient(blurred_image)

#non max suppression
gradient_nonmax_supress = nonMaxSuppression(gradient,theta)

#thresholding
threshold_high = otsuMethod(gradient_nonmax_supress)
edges = adaptiveThresholding(gradient_nonmax_supress,-0.05,70)

## plot
fig1, axs = plt.subplots(1, 1)
# axs[0].imshow(gray_image,"gray")
# axs[1].imshow(gradient_nonmax_supress,"gray")
axs.imshow(edges,"gray")



plt.show()

