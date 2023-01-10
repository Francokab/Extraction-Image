import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.signal as sig
import time

def gaussian_filter(kernel_size, sigma=1, muu=0):
    #Initializing 
    gauss = np.zeros((kernel_size,kernel_size))
    dst = 1/(2 * np.pi * sigma * sigma)
    k = int((kernel_size - 1)/2) 
    for i in range(kernel_size):
        for j in range(kernel_size):
            gauss[i,j] = dst*np.exp(-(((i+1)-(k+1)) * ((i+1)-(k+1)) + ((j+1)-(k+1)) * ((j+1)-(k+1)))/(2 * sigma * sigma))

    return gauss

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def logtime(L):
    L.append(time.monotonic())

Ltime = [time.monotonic()]

#image reading
target_file = "images\\Large_Scaled_Forest_Lizard.jpg"
image = mpimg.imread(target_file)
logtime(Ltime)

#To gray
gray_image = rgb2gray(image)
if gray_image.max() > 1:
    gray_image = gray_image/255
logtime(Ltime)

#Blurr
gauss =gaussian_kernel(5)
blurred_image = sig.convolve2d(gray_image,gauss,'same')
logtime(Ltime)

#Finding the gradient of the image
Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
gradient_x = sig.convolve2d(blurred_image,Kx,'same')
gradient_y = sig.convolve2d(blurred_image,Ky,'same')
gradient = (gradient_x**2 + gradient_y**2)**(1/2)
theta = np.arctan2(gradient_y,gradient_x)
logtime(Ltime)

gradient_nonmax_supress = gradient.copy()
nx,ny = gradient.shape
for i in range(1,nx-1):
    for j in range(1,ny-1):
        local_angle = theta[i,j]
        local_angle = round(4*local_angle/np.pi)%4
        if local_angle == 0:
            index1 = (i,j+1)
            index2 = (i,j-1)
        elif local_angle == 1:
            index1 = (i-1,j+1)
            index2 = (i+1,j-1)
        elif local_angle == 2:
            index1 = (i+1,j)
            index2 = (i-1,j)
        elif local_angle == 3:
            index1 = (i+1,j+1)
            index2 = (i-1,j-1)
        if gradient[i,j] <= gradient[index1] or gradient[i,j] <= gradient[index2]:
            gradient_nonmax_supress[i,j] = 0
logtime(Ltime)

#thresholding
threshold_high = 0.3
threshold_low = 0.1
edges = gradient_nonmax_supress.copy()
for i in range(nx):
    for j in range(ny):
        if edges[i,j] > threshold_high:
            edges[i,j] = 1.0
        elif edges[i,j] > threshold_low:
            edges[i,j] = 0.5
        else:
            edges[i,j] = 0.0
logtime(Ltime)

#histeresis
edges_histeresis = edges.copy()
for i in range(nx):
    for j in range(ny):
        if edges_histeresis[i,j] == 0.5:
            indexes = []
            for k in [(i-1,j-1), (i-1,j), (i-1,j+1), (i,j-1), (i,j+1), (i+1,j-1), (i+1,j), (i+1,j+1)]:
                if 0 < k[0] and k[0] < nx and 0 < k[1] and k[1] < ny:
                    indexes.append(k)
            for k in indexes:
                if edges_histeresis[k] == 1.0:
                    edges_histeresis[i,j] = 1.0
                if edges_histeresis[i,j] != 1.0:
                    edges_histeresis[i,j] = 0.0
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

