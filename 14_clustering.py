from skimage.io import imread
from skimage import img_as_float
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image_loc = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image_loc[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image_loc


image = imread('parrots.jpg')
image_f = img_as_float(image)
w, h, d = original_shape = tuple(image_f.shape)
image_array = np.reshape(image_f, (w * h, d))
kmeans = KMeans(n_clusters=11, init='k-means++', random_state=241).fit(image_array)

labels = kmeans.predict(image_array)

codebook_random = shuffle(image_array, random_state=0)[:64 + 1]
labels_random = pairwise_distances_argmin(codebook_random,
                                          image_array,
                                          axis=0)

for k in range(1, 3):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=241).fit(image_array)
    labels = kmeans.predict(image_array)
    predict_image = recreate_image(kmeans.cluster_centers_, labels, w, h)
    w, h, d = original_shape = tuple(predict_image.shape)
    predict_array = np.reshape(predict_image, (w * h, d))
    mse = mean_squared_error(image_array, predict_array)
    psnr = 20*np.log10(1)-10*np.log10(mse)
    print("{} clusters, psnr = {:.3f}".format(k, psnr))

plt.figure(1)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Original image (96,615 colors)')
plt.imshow(image)

plt.figure(2)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image (64 colors, K-Means)')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

plt.figure(3)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image (64 colors, Random)')
plt.imshow(recreate_image(codebook_random, labels_random, w, h))
plt.show()
