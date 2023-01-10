import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.signal as sig
import time

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

def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N))
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 1
                r = 1
                
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0.0

            except IndexError as e:
                pass
    return Z

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

gradient_nonmax_supress = non_max_suppression(gradient,theta)
nx,ny = gradient.shape
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

