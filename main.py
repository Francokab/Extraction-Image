import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.signal as sig

def gaussian_filter(kernel_size, sigma=1, muu=0):
    #Initializing 
    gauss = np.zeros((kernel_size,kernel_size))
    dst = 1/(2 * np.pi * sigma * sigma)
    k = int((kernel_size - 1)/2) 
    for i in range(kernel_size):
        for j in range(kernel_size):
            gauss[i,j] = dst*np.exp(-(((i+1)-(k+1)) * ((i+1)-(k+1)) + ((j+1)-(k+1)) * ((j+1)-(k+1)))/(2 * sigma * sigma))

    return gauss

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


#image reading
target_file = "images\\dragons.png"
image = mpimg.imread(target_file)

#To gray
gray = rgb2gray(image)

#Blurr
gauss =gaussian_filter(5)
out = sig.convolve2d(gray,gauss)

ker = np.array([-1,0,1])
out2 = np.array([np.convolve(i,ker,'same') for i in out])
out3 = np.transpose(np.array([np.convolve(i,ker,'same') for i in np.transpose(out)]))
out4 = (out2**2 + out3**2)**(1/2)

print(out2)
print(out2*out2)

## plot
fig1, axs = plt.subplots(2, 2)
axs[0,0].imshow(image)
axs[0,1].imshow(out,"gray")
axs[1,0].imshow(out2,"gray")
axs[1,1].imshow(out4,"gray")
plt.show()

