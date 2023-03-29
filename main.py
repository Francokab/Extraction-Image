import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from function import *
from algo import *

#image reading
target_file = "images\\dragons.png"
image = mpimg.imread(target_file)

#To gray
gray_image = imageToGrayNormalize(image)

blurred_image = blurrImage(gray_image, kernel_size=9, sigma = 3)

output = SUSANpart1(blurred_image)
output2 = SUSANpart2(output)

# #Blurr
# blurred_image = blurrImage(gray_image, sigma = 1.4)

# #Finding the gradient of the image
# gradient, theta = findGradient(blurred_image)

# #non max suppression
# gradient_nonmax_supress = nonMaxSuppression(gradient,theta)

# #thresholding
# threshold_high = otsuMethod(gradient_nonmax_supress)
# edges = adaptiveThresholding(gradient_nonmax_supress,-0.05,70)

## plot
fig1, axs = plt.subplots(1, 2)
axs[0].imshow(blurred_image, "gray", vmin = 0, vmax = 1)
axs[1].imshow(output2,"gray")





plt.show()

