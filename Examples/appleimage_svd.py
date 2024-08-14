import numpy as np
import math
import imageio
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import matplotlib.cm as cm

apple_jpeg = imageio.imread('apple.jpeg')
apple_jpeg = rgb2gray(apple_jpeg)
M = np.linalg.norm(apple_jpeg)
plt.imshow(apple_jpeg, interpolation='nearest', cmap=cm.Greys_r)
plt.axis('off')
plt.show()
plt.imsave('original.png',apple_jpeg,format='png',cmap=cm.Greys_r)
n = np.shape(apple_jpeg)[1]
U, sg_values, V = np.linalg.svd(apple_jpeg)
plt.semilogy(np.arange(n), sg_values/M)
plt.xlabel("Index of singular values")
plt.ylabel("$\sigma_k/M$")
plt.title('Singular values of the image as a matrix')
plt.show()

val = [10,50,150,200]

for i in range(1,5):
    
    k = val[i-1]

    image = U[:,:k]@np.diag(sg_values[:k])@V[:k,:]
    plt.imshow(image, interpolation='nearest', cmap=cm.Greys_r)
    plt.axis('off')
    plt.show()
    plt.imsave(f'{k}sgvalues.png',image,format='png',cmap=cm.Greys_r)

