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

output_image = SUSANpart1(gray_image)

output_image2 = SUSANpart2(output_image)

## plot
fig1, axs = plt.subplots(1, 3)
axs[0].imshow(gray_image,"gray")
axs[1].imshow(output_image,"gray")
axs[2].imshow(output_image2,"gray")


plt.show()

