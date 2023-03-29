import numpy as np
import scipy.signal as sig

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def blurrImage(img,kernel_size = 5, sigma=1):
    """doc test
        123
        test"""
    gauss = gaussian_kernel(kernel_size,sigma=sigma)
    return sig.convolve2d(img,gauss,'same')

def func(a,b,c):
    print(a,b,c)

dict = {"a":1, "b":3, "c":2}

func(**dict)
print(7//2)