import numpy as np
from scipy.misc import imread, imsave, imresize
import pylab as plt
import sys
root = '/Users/matthewtheisen/Google_Drive/ImageProc/'
sys.path.append(root)
from retargeting import Retargeter

image_data = imread(root+'marco.jpg').astype(np.float32)
scaled_image_data = image_data / 255.
image_slice_red =  scaled_image_data[:,:,0]
image_slice_green =  scaled_image_data[:,:,1]
image_slice_blue =  scaled_image_data[:,:,2]

r = Retargeter()

plt.subplot(221)
plt.imshow(1-r._Deriv(scaled_image_data)[:,:,0], cmap=plt.cm.Reds_r)
plt.subplot(222)
plt.imshow(1-r._Deriv(scaled_image_data)[:,:,1], cmap=plt.cm.Greens_r)
plt.subplot(223)
plt.imshow(1-r._Deriv(scaled_image_data)[:,:,2], cmap=plt.cm.Blues_r)  
plt.subplot(224)
plt.imshow(scaled_image_data)  
plt.show()
