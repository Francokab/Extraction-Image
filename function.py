import numpy as np
import scipy.signal as sig
from decorator import timer
from math import floor, ceil

@timer
def imageToGrayNormalize(img, cr = 0.2126, cg = 0.7152, cb = 0.0722):
    if len(img.shape)>2:
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        gray = cr * r + cg * g + cb * b
    else:
        gray = img.copy()
    if gray.max() > 1:
        gray = gray/255
    return gray

@timer
def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

@timer
def blurrImage(img,kernel_size = 5, sigma=1):
    gauss = gaussian_kernel(kernel_size,sigma=sigma)
    return sig.convolve2d(img,gauss,'same')

@timer
def gradientOperator(gradientType = "regular"):
    if gradientType == "regular":
        Kx = np.array([[-1, 0, 1]])
        Ky = np.array([[1], [0], [-1]])
    elif gradientType == "roberts":
        Kx = np.array([[1, 0], [0, -1]])
        Ky = np.array([[0, 1], [-1, 0]])
    elif gradientType == "prewitt":
        Kx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        Ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    elif gradientType == "sobel":
        Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    else:
        raise ValueError
    return (Kx,Ky)

@timer
def findGradient(img, gradientType = "regular"):
    Kx, Ky = gradientOperator(gradientType=gradientType)
    gradient_x = sig.convolve2d(img,Kx,'same')
    gradient_y = sig.convolve2d(img,Ky,'same')
    gradient = (gradient_x**2 + gradient_y**2)**(1/2)
    theta = np.arctan2(gradient_y,gradient_x)
    if gradientType == "roberts":
        theta = theta - 3*np.pi/4
    return (gradient,theta)

@timer    
def nonMaxSuppression(img, D):
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

@timer
def thresholding(img, high, low, out1 = 1.0, out2 = 0.5, out3 = 0.0):
    edges = img.copy()
    edges[edges>high] = out1
    edges[(edges<high) & (edges>low)] = out2
    edges[edges<low] = out3
    return edges

@timer
def histeresis(img):
    edges_histeresis = img.copy()
    nx, ny = img.shape
    explored = np.zeros((nx,ny))
    queue = []
    label = 1
    for i in range(nx):
        for j in range(ny):
            if edges_histeresis[i,j] == 1.0 and explored[i,j] == 0:
                explored[i,j] = label
                queue.append((i,j))
            
            while(len(queue)>0):
                i1, j1 = queue[0]
                for k in [(i1-1,j1-1), (i1-1,j1), (i1-1,j1+1), (i1,j1-1), (i1,j1+1), (i1+1,j1-1), (i1+1,j1), (i1+1,j1+1)]:
                    try:
                        if explored[k] == 0:
                            if edges_histeresis[k] == 1.0:
                                explored[k] = label
                                queue.append(k)
                            elif edges_histeresis[k] == 0.5:
                                explored[k] = label
                                queue.append(k)
                                edges_histeresis[k] = 1.0
                    except IndexError as e:
                        pass
                queue.pop(0)
            else:
                label = label + 1
                
    edges_histeresis[edges_histeresis < 1.0] = 0.0
    return edges_histeresis

def computeOtsuCriteria(im, th):
    # create the thresholded image
    thresholded_im = np.zeros(im.shape)
    thresholded_im[im >= th] = 1

    # compute weights
    nb_pixels = im.size
    nb_pixels1 = np.count_nonzero(thresholded_im)
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1

    # if one the classes is empty, eg all pixels are below or above the threshold, that threshold will not be considered
    # in the search for the best threshold
    if weight1 == 0 or weight0 == 0:
        return np.inf

    # find all pixels belonging to each class
    val_pixels1 = im[thresholded_im == 1]
    val_pixels0 = im[thresholded_im == 0]

    # compute variance of these classes
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0

    return weight0 * var0 + weight1 * var1

@timer
def otsuMethod(im):
    # testing all thresholds from 0 to the maximum of the image
    threshold_range = np.arange(0,1,0.05)
    criterias = [computeOtsuCriteria(im, th) for th in threshold_range]

    # best threshold is the one minimizing the Otsu criteria
    best_threshold = threshold_range[np.argmin(criterias)]
    return best_threshold

@timer
def dericheFilter(img,a,b,c1,axis):
    if axis == 0:
        invAxis = 1
    else:
        invAxis = 0
    y1 = sig.lfilter([a[0],a[1]],[1,-b[0],-b[1]],img,axis=axis)
    y2 = sig.lfilter([0,a[2],a[3]],[1,-b[0],-b[1]],np.flip(img,axis=axis),axis=axis)
    y2 = np.flip(y2, axis=axis)
    # nx, ny = img.shape
    # y1 = np.zeros(img.shape)
    # if axis == 1:
    #     for i in range(nx):
    #         X=[0,0]
    #         Y=[0,0,0]
    #         for j in range(ny):
    #             X[0] = img[i,j]
    #             Y[0] = a[0] * X[0] + a[1] * X[1]


    theta = c1*(y1+y2)

    return theta

@timer    
def fullDericheFilter(img,a1,a2,b,c):
    theta1 = dericheFilter(img,a1,b,c[0],1)
    theta2 = dericheFilter(theta1,a2,b,c[1],0)
    return theta2

@timer
def Laplacian(img):
    Kxx = np.array([[1, -2, 1]])
    Kxy = 0.25 * np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]])
    Kyy = np.array([[1], [-2], [1]])
    Lxx = sig.convolve2d(img,Kxx,'same')
    Lxy = sig.convolve2d(img,Kxy,'same')
    Lyy = sig.convolve2d(img,Kyy,'same')
    Kx, Ky = gradientOperator()
    Lx = sig.convolve2d(img,Kx,'same')
    Ly = sig.convolve2d(img,Ky,'same')
    return (Lx, Ly, Lxx, Lxy, Lyy)

@timer
def Laplacian2(img):
    K = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    Lap = sig.convolve2d(img,K,'same')
    return (Lap)

@timer
def SUSANpart1(img, rmax = 3.4, t = 0.1):
    imax, jmax = img.shape
    halfWindowSize = floor(rmax)
    mask = np.zeros((2*halfWindowSize+1,2*halfWindowSize+1))
    for index, value in np.ndenumerate(mask):
        if np.linalg.norm(np.array(index) - np.array([halfWindowSize,halfWindowSize])) < rmax:
            mask[index] = 1
    sliding_window = np.lib.stride_tricks.sliding_window_view(np.pad(img,halfWindowSize),mask.shape)
    output = np.zeros(img.shape)
    for index0, value0 in np.ndenumerate(img):
        view = sliding_window[index0]
        output[index0] = np.sum(mask * np.exp(- ((view - value0) / t)**6))
    return output

@timer
def SUSANpart2(img):
    output = np.zeros(img.shape)
    g = 3* img.argmax()/4
    for index, x in np.ndenumerate(img):
        if x<g:
            output[index] = g - x
    return output

@timer
def adaptiveThresholding(img, C, radius):
    imax, jmax = img.shape
    halfWindowSize = floor(radius)
    output = np.zeros(img.shape)
    for index0, value0 in np.ndenumerate(img):
        i0, j0 = index0
        view = img[max(0,i0-halfWindowSize):min(imax,i0+halfWindowSize+1), max(0,j0-halfWindowSize):min(jmax,j0+halfWindowSize+1)]
        mean = np.mean(view) - C
        if mean < value0:
            output[index0] = 1.0
    return output
