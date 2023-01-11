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

