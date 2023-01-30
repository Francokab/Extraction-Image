import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from function import *
from algo import *
from math import floor, ceil

#image reading
target_file = "images\\medieval_house.jpg"
image = mpimg.imread(target_file)

#To gray
gray_image = imageToGrayNormalize(image)

# blurred_image = blurrImage(gray_image, sigma = 1.4)

# Lx, Ly, Lxx, Lxy, Lyy = Laplacian(blurred_image)

# Lap = Lx**2 * Lxx + 2*Lx*Ly*Lxy + Ly**2 * Lyy

# # th = 0.0005
# # Lap[abs(Lap) >= th ] = 1
# # Lap[abs(Lap) < th ] = 0

# Lap2 = Laplacian2(gray_image)

@timer
def SUSANpart1(img, rmax = 3.4, t = 0.2):
    imax, jmax = img.shape
    output = np.zeros(img.shape)
    for i0 in range(imax):
        for j0 in range(jmax):
            for i in range(max(0,floor(i0-rmax)), min(imax,ceil(i0+rmax))):
                for j in range(max(0,int(j0-rmax-1)), min(jmax,int(j0+rmax+1))):
                    if (i0-i)**2 + (j0-j)**2 < rmax**2:
                        output[i0,j0] += np.exp(-((img[i,j] - img[i0,j0])/t)**6)
    return output

output_image = SUSANpart1(gray_image)

@timer
def SUSANpart2(img):
    output = np.zeros(img.shape)
    g = 3* img.argmax()/4
    for index, x in np.ndenumerate(img):
        if x<g:
            output[index] = g - x
    return output

output_image2 = SUSANpart2(output_image)
print("test")
## plot
fig1, axs = plt.subplots(1, 3)
axs[0].imshow(gray_image,"gray")
axs[1].imshow(output_image,"gray")
axs[2].imshow(output_image2,"gray")


plt.show()

