from function import *

def canny(img, th_high, th_low):
    #Blurr
    blurred_image = blurrImage(img)

    #Finding the gradient of the image
    gradient, theta = findGradient(blurred_image)

    #non max suppression
    gradient_nonmax_supress = nonMaxSuppression(gradient,theta)

    #thresholding
    edges = thresholding(gradient_nonmax_supress, th_high, th_low)

    #histeresis
    edges_histeresis = histeresis(edges)


    return edges_histeresis

def cannyWithOtsu(img):
    #Blurr
    blurred_image = blurrImage(img)

    #Finding the gradient of the image
    gradient, theta = findGradient(blurred_image)

    #non max suppression
    gradient_nonmax_supress = nonMaxSuppression(gradient,theta)

    #thresholding
    threshold_high = otsuMethod(gradient_nonmax_supress)
    edges = thresholding(gradient_nonmax_supress, threshold_high, threshold_high/2)

    #histeresis
    edges_histeresis = histeresis(edges)

    return edges_histeresis

def deriche(img,alpha):
    b = [2*np.exp(-alpha), -np.exp(-2*alpha)]
    cst = (1-np.exp(-alpha))**2
    k = cst/(1+alpha*b[0]+b[1])
    a1 = [k, k*b[0]*(alpha-1)/2, k*b[0]*(alpha+1)/2, k*b[1]]
    a0 = [0, 1, -1, 0]
    print(b)
    print(cst)
    print(k)
    print(a1)
    print(a0)

    #Blurr
    blurred_image = fullDericheFilter(img,a1,a1,b,[1,1])

    #Finding the gradient of the image
    gradient_x = fullDericheFilter(blurred_image,a0,a1,b,[-cst,1])
    gradient_y = fullDericheFilter(blurred_image,a1,a0,b,[1,-cst])
    gradient = (gradient_x**2 + gradient_y**2)**(1/2)
    theta = np.arctan2(gradient_y,gradient_x)

    #non max suppression
    gradient_nonmax_supress = nonMaxSuppression(gradient,theta)

    #thresholding
    threshold_high = otsuMethod(gradient_nonmax_supress)
    edges = thresholding(gradient_nonmax_supress, threshold_high, threshold_high/2)

    #histeresis
    edges_histeresis = histeresis(edges)

    #return [blurred_image,gradient_x,gradient_y,gradient,theta,gradient_nonmax_supress,edges,edges_histeresis]
    return edges_histeresis
