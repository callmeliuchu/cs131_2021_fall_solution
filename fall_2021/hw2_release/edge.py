"""
CS131 - Computer Vision: Foundations and Applications
Assignment 2
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/18/2017
Python Version: 3.5+
"""

import numpy as np

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')
#     print(Wk)
    ### YOUR CODE HERE
    for i in range(Hi):
        for j in range(Wi):
#             print(j,j+Wk)
#             print(padded[i:i+Hk][j:j+Wk])
            out[i][j] = np.sum(padded[i:i+Hk,j:j+Wk]*kernel)
    ### END YOUR CODE

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))

    ### YOUR CODE HERE
    k = (size - 1) // 2
    for i in range(size):
        for j in range(size):
            sg = sigma*sigma
            kernel[i][j] = np.exp(-((i-k)**2+(j-k)**2)/2/sg)/2/np.pi/sg
    ### END YOUR CODE

    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[-0.5,0,0.5]])
    out = conv(img,kernel)
    ### END YOUR CODE

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[-0.5,0,0.5]]).T
    out = conv(img,kernel)
    ### END YOUR CODE

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)
    ### YOUR CODE HERE
    px = partial_x(img)
    py = partial_y(img)
    h,w = img.shape
    for j in range(h):
        for i in range(w):
            G[j][i] = np.sqrt(px[j][i]**2+py[j][i]**2)
            theta[j][i] = np.rad2deg(np.arctan2(py[j][i],px[j][i]))
    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
#     H, W = G.shape
#     out = np.zeros((H, W))

#     # Round the gradient direction to the nearest 45 degrees
#     theta = np.deg2rad(theta)
#     theta = np.floor((theta + 22.5) / 45) * 45
#     theta = (theta % 360.0).astype(np.int32)
    
#     #print(G)
#     ### BEGIN YOUR CODE
#     out = G.copy()
#     for j in range(1,H-1):
#         for i in range(1,W-1):
#             angle = theta[j][i]
#             if angle == 0 or angle == 180:
#                 ma = max(G[j][i-1],G[j][i+1])
#             elif angle == 45 or angle == 45 + 180:
#                 ma = max(G[j+1][i+1],G[j-1][i-1])
#             elif angle == 90 or angle == 90 + 180:
#                 ma = max(G[j+1][i],G[j-1][i])
#             elif angle == 135 or angle == 135 + 180:
#                 ma = max(G[j+1][i-1],G[j-1][i+1])
#             if ma > G[j][i]:
#                 out[j][i] = 0
#     ### END YOUR CODE

#     return out

    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45  # 方向定位
    # 添加一层padding
    padd = np.zeros((H+2,W +2))
    padd[1:H+1, 1:W+1] = G
    for m in range(1, H+1):
        for n in range(1, W+1):
            # 题目定义为顺时针方向，和逆时针相反,y方向相反
            rad = np.deg2rad(theta[m-1, n-1])
            i =int(np.around(np.sin(rad)))   # 行
            j =int(np.around(np.cos(rad)))   # 列
            p1 = padd[m+i, n+j]
            p2 = padd[m-i, n-j]
            if(padd[m, n] > p1 and padd[m, n] > p2): # 一个方向上
                out[m-1, n-1] = padd[m, n]
            else:
                out[m-1, n-1] = 0
    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype=np.bool)
    weak_edges = np.zeros(img.shape, dtype=np.bool)

    ### YOUR CODE HERE
    strong_edges = img > high
    low = img < low
    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=np.bool)

    # Make new instances of arguments to leave the original
    # references intact
    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)

    ### YOUR CODE HERE
    indices = list(indices)
    while indices:
        i,j = indices.pop(0)
        for k1 in [-1,0,1]:
            for k2 in [-1,0,1]:
                ik = i + k1
                jk = j + k2
                if 0 <= ik < H and  0 <= jk < W and not (ik==i and jk == j) and not edges[ik][jk]:
                    if weak_edges[ik][jk]:
                        indices.append([ik,jk])
                        edges[ik][jk] = True
                   
    ### END YOUR CODE

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    ### YOUR CODE HERE
    kernel = gaussian_kernel(kernel_size,sigma)
    smooth = conv(img,kernel)
    G,theta = gradient(smooth)
    out = non_maximum_suppression(G,theta)
    strong,weak=double_thresholding(out,high,low)
    edge = link_edges(strong,weak)
    ### END YOUR CODE

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W).
        
    Returns:
        accumulator: numpy array of shape (m, n).
        rhos: numpy array of shape (m, ).
        thetas: numpy array of shape (n, ).
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    ### YOUR CODE HERE
    for i, j in zip(ys, xs):
        for idx in range(thetas.shape[0]):
            r = j * cos_t[idx] + i * sin_t[idx]
            accumulator[int(r + diag_len), idx] += 1
    ### END YOUR CODE

    return accumulator, rhos, thetas
