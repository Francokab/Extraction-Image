import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time
from function import *
from algo import *


# def logtime(name):
#     Ltime.append(time.monotonic_ns())
#     LtimeName.append(name)
# LtimeName = ['start']
# Ltime = [time.monotonic_ns()]

#image reading
target_file = "images\\Sunflowers_in_July.jpg"
image = mpimg.imread(target_file)

#To gray
gray_image = imageToGrayNormalize(image)

img_deriche = deriche(gray_image,1.4)

#debug time
# for i in range(1,len(Ltime)):
#     print((Ltime[i]-Ltime[i-1])*1e-9,"\t",LtimeName[i])

## plot
fig1, axs = plt.subplots(1, 2)
axs[0].imshow(gray_image,"gray")
axs[1].imshow(img_deriche,"gray")
plt.show()

