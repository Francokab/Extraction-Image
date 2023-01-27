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

blurred_image = blurrImage(gray_image, sigma = 1.4)

Lx, Ly, Lxx, Lxy, Lyy = Laplacian(blurred_image)

Lap = Lx**2 * Lxx + 2*Lx*Ly*Lxy + Ly**2 * Lyy

# th = 0.0005
# Lap[abs(Lap) >= th ] = 1
# Lap[abs(Lap) < th ] = 0

Lap2 = Laplacian2(gray_image)

print("test")
## plot
fig1, axs = plt.subplots(1, 2)
axs[0].imshow(gray_image,"gray")
axs[1].imshow(Lap2,"gray")


plt.show()

