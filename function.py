import numpy as np
import scipy.signal as sig
from decoratorGUI import *
import matplotlib.image as mpimg
from decorator import timer
from math import floor, ceil

imageList = [
    ("Bike","images\\Bikesgray.jpg"),
    ("Dragons","images\\dragons.png"),
    ("Lizard","images\\Large_Scaled_Forest_Lizard.jpg"),
    ("Lizard Reduced","images\\Large_Scaled_Forest_Lizard_reduced.jpg"),
    ("Le Chat Botte","images\\Lechatbotte.jpg"),
    ("Medieval House","images\\medieval_house.jpg"),
    ("Sunflower","images\\Sunflowers_in_July.jpg"),
    ("Valve","images\\Valve_original.png")
]

for image in imageList:
    IMAGE_DICT[image[0]] = image[1]

algoList = [
    algo("naive","Aproche naïve", "", ["imageToGrayNormalize","blurrImage","findGradient"]),
    algo("canny","Filtre de Canny", "", ["imageToGrayNormalize","blurrImage","findGradient","nonMaxSuppression","thresholding","histeresis"]),
    algo("deriche","Filtre de Deriche", "", ["imageToGrayNormalize","dericheBlurr","dericheGradient","nonMaxSuppression","thresholding","histeresis"])
]

for _algo in algoList:
    ALGO_DICT[_algo.name] = _algo

@imageReadingGUI
@timer
def readImageFromFile(target_file):
    image = mpimg.imread(target_file)
    return image

@parameterGUI
@timer
def imageToGrayNormalize(img, cr = 0.2126, cg = 0.7152, cb = 0.0722):
    """Transforme en Noir et Blanc
    img; Image; Image en entrée; image
    cr; Coef Rouge; proportion de de rouge qui et pris en compte pour calculer le gris; slider; 0.2126; [0.0, 1.0]
    cg; Coef Vert; proportion de de vert qui et pris en compte pour calculer le gris; slider; 0.7152; [0.0, 1.0]
    cb; Coef Bleu; proportion de de bleu qui et pris en compte pour calculer le gris; slider; 0.0722; [0.0, 1.0]
    end_parameter
    
    Les informations en couleurs d'une image sont souvent superflu, l'intensité lumineuse d'une image est suffisante pour faire les détection d'objet que l'on veut, \
de plus cela permet de réduire le nombre de dimensions à considérer quand on fait nos calcul, il est donc souvent utile de prendre une version en Noir et blanc \
de notre image.

    La valeur finale d'un pixale va être déterminée à partir de la somme pondérée des 3 couleurs du pixel. \
Comme chaque couleur n'est pas détecter par nos yeux (et nos caméras) avec la même force, on prend en compte chaque \
couleur avec des proportions différentes, nottament le vert est énormément pris en compte
    """
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

@parameterGUI
@timer
def blurrImage(img,kernel_size = 5, sigma=1):
    """Flou gaussien
    img; Image; Image en entrée; image
    kernel_size; Taille du noyau; Taille de la matrice gaussienne à appliquer; int; 5; [1, 31]
    sigma; Sigma; Intensité du flou crée par la matrice; float; 1; [0.01, 10]
    end_parameter

    Cette fonction floutte la fonction que l'on a en entrée à l'aide d'un flou gaussien. 

    Pour faire ça la fonction utilise une convolution entre l'image et une matrice gaussienne, la conséquence est que chaque point de l'image se retrouve diluer dans les points voisins. 
    
    En théorie un flou gaussien prend en compte tout les points de l'image mais en pratique comme le poids de chaque des points descends de manière exponentielle avec la distance (c'est la partie gaussienne), on peut ce permettre de n'utiliser que une petite matrice.
    """
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

@parameterGUI
@timer
def findGradient(img, gradientType = "regular"):
    """Calculer le gradient
    img; Image; Image en entrée; image
    gradientType; Type de Gradient; Type d'oppérateur de gradient à utiliser pour calculer le gradient; list; regular; [regular, roberts, prewitt, sobel]
    end_parameter
    
    Cette fonction permet de trouver le gradient de l'images, c'est à dire comment l'intensité d'un pixel change par rapport à celle des pixel voisins.

    Pour faire cela la fonction va d'abord calculer le gradient selon x et selon y en en faisant une convolution avec un opérateur. \
Il y a 4 opérateur qui sont communément utiliser :
    L'opérateur régulier : [-1, 0, 1]
    L'opérateur de Roberts : [1,  0]
                                            [0, -1]
    L'opérateur de Prewitt : [1, 0, -1]
                                           [1, 0, -1]
                                           [1, 0, -1]
    L'opérateur de Sobel : [1, 0, -1]
                                         [2, 0, -2]
                                         [1, 0, -1]
    Pour trouver le gradient selon l'autre direction, il suffit de prendre la transposé.
    
    Une fois que l'on a le gradient selon x et selon y on peut trouver la norme et l'orientation du gradient avec (norme = sqrt(x^2+y^2) et orientation = arctan2(y,x))
    """
    Kx, Ky = gradientOperator(gradientType=gradientType)
    gradient_x = sig.convolve2d(img,Kx,'same')
    gradient_y = sig.convolve2d(img,Ky,'same')
    gradient = (gradient_x**2 + gradient_y**2)**(1/2)
    theta = np.arctan2(gradient_y,gradient_x)
    if gradientType == "roberts":
        theta = theta - 3*np.pi/4
    return (gradient,theta)

@parameterGUI
@timer    
def nonMaxSuppression(img, D):
    """Suppression des non-Maximum-locaux
    img; Gradient; Gradient de l'image; image
    D; Theta; Orientation de l'image; image 
    end_parameter
    
    Apres avoir calculer le gradient d'une image, on se retrouve avec une image qui a des pixels non nuls au \
endroit où il a des changements d'intensité, c'est à dire les bords. Mais ces changements d'intensité ce font \
sur une distance de plusieurs pixels, or on voudrait montrait les bords comme ayant une épaisseur de seulment un pixel.

    Cette fonction va donc chercher à désépaissir les bords en ne gardant que les maximums locaux et notament on va \
utiliser les informations d'orientations que l'on a obtenue en calculant le gradient pour faire la recherche de maximum local \
seulement selon la diraction du gradient.
    """
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

@parameterGUI
@timer
def thresholding(img, high, low, out1 = 1.0, out2 = 0.5, out3 = 0.0):
    """Double seuillage de l'image
    img; Image; Image en entrée; image
    high; Seuil Haut; Les valeurs au dessus de ce seuil vont être mis à 1; float; 0.5; [0.0, 1.0]
    low; Seuil Bas; Les valeurs en dessous de ce seuil vont être mis à 0; float; 0.3; [0.0, 1.0]
    otsu; Méthode d'Otsu; Utiliser la méthode d'Otsu pour trouver le meilleur seuil; special_bool; True; [otsuMethod, img, high:low]
    end_parameter
    
    Une fois que l'on a des bords de 1 d'épaisseur, il faut que l'on supprime les bords qui sont considérés trop mineur, \
ceux pour qui le gradient était faible.

    Pour faire cela on va faire un seuillage, c'est à dire que l'on va supprimer toute les valeurs qui sont en dessous d'un \
certain seuil, et mettre à 1 ceux qui sont au dessus de ce seuil.
    En faisant à double seuillage, on a deux seuil: les valeurs au dessus du seuil haut sont 1, les valeurs en dessous du seuil bas sont 0, \
et les valeurs intermédiaire sont mis à la valeurs arbitraires de 0.5. On va ensuite utilisé l'histeresis pour dire si on va garder ou pas ces valeurs intermédiaires

    Une méthode existante pour trouver une valeur de seuil automatiquement est la méthode d'Otsu. La méthode d'Otsu \
cherche à trouver le seuil qui va minimiser la variance de l'intensité intra-classe, c'est à dire qu'il va trouver \
le seuil pour lequel la variance en intensité pour les points qui sont en dessous du seuil et la variance en intensité \
pour les points qui sont au dessus du seuil va être minimiser.
    Il est commun de prendre ensuite le seuil bas comme étant la moitié du seuil que l'on a trouvé avec la méthode d'Otsu
    """
    edges = img.copy()
    edges[edges>high] = out1
    edges[(edges<high) & (edges>low)] = out2
    edges[edges<low] = out3
    return edges

@parameterGUI
@timer
def histeresis(img):
    """Histeresis
    img; Image; Image en entrée; image
    end_parameter
    
    Une Fois que l'on a fais un double seuillage, on se retrouve avec des valeurs intermédiaire, ils correspondent au \
bords dont on est pas sur s'il sont assez important pour être garder. Pour pouvoir décider lesquelles de ces valeurs \
intermédiaires on va garder, on va regarder si elles sont connectées à d'autre bord qui ont des valeurs de 1. Ainsi \
vont être gardée seulement les bords intermédiaires qui font partie d'un bord important.
    """
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

@secondaryFunction
@timer
def otsuMethod(im):
    if im is not None:
        # testing all thresholds from 0 to the maximum of the image
        threshold_range = np.arange(0,1,0.05)
        criterias = [computeOtsuCriteria(im, th) for th in threshold_range]

        # best threshold is the one minimizing the Otsu criteria
        best_threshold = threshold_range[np.argmin(criterias)]
        return best_threshold, best_threshold/2
    else:
        return 0.5, 0.25

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

@parameterGUI
def dericheBlurr(img, alpha):
    """Flouttage de Deriche
    img; Image; Image en entrée; image
    alpha; Alpha; Inverse de l'intensité du flou; float; 1; [0.001, 10]
    end_parameter

    Comme le floutage gaussien, cette fonction permet de flouter une image mais au lieu \
d'utiliser une convolution, elle utilise un filtre IIR (Filtre à réponse impulsionnelle infine) \
de Deriche avec des valeurs bien défnie qui dépendent de exp(-alpha)
    """
    b = [2*np.exp(-alpha), -np.exp(-2*alpha)]
    cst = (1-np.exp(-alpha))**2
    k = cst/(1+alpha*b[0]+b[1])
    a1 = [k, k*b[0]*(alpha-1)/2, k*b[0]*(alpha+1)/2, k*b[1]]
    a0 = [0, 1, -1, 0]

    #Blurr
    blurred_image = fullDericheFilter(img,a1,a1,b,[1,1])
    return blurred_image

@parameterGUI
def dericheGradient(img, alpha):
    """Calculer le gradient avec la méthode de Deriche
    img; Image; Image en entrée; image
    alpha; Alpha; Alpha; float; 1; [0.001, 10]
    end_parameter

    Cette fonction permet de calculer le gradient selon x et selon y d'une image mais au lieu \
d'utiliser une convolution, elle utilise un filtre IIR (Filtre à réponse impulsionnelle infine) \
de Deriche avec des valeurs bien défnie qui dépendent de exp(-alpha)

    Comme pour la manière conventionelle de trouver le gradient, on va ensuite prendre le gradient selon x et selong y \
et calculer la norme et l'orientation du gradient pour chaque pixel
    """
    b = [2*np.exp(-alpha), -np.exp(-2*alpha)]
    cst = (1-np.exp(-alpha))**2
    k = cst/(1+alpha*b[0]+b[1])
    a1 = [k, k*b[0]*(alpha-1)/2, k*b[0]*(alpha+1)/2, k*b[1]]
    a0 = [0, 1, -1, 0]

    #Finding the gradient of the image
    gradient_x = fullDericheFilter(img,a0,a1,b,[-cst,1])
    gradient_y = fullDericheFilter(img,a1,a0,b,[1,-cst])
    gradient = (gradient_x**2 + gradient_y**2)**(1/2)
    theta = np.arctan2(gradient_y,gradient_x)

    return (gradient, theta)

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
