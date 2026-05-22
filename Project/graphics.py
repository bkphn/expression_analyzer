import os

from skimage.feature import hog
from skimage import io
import matplotlib.pyplot as plt

IMAGE_PATH = os.path.join("Data", "test", "happy", "PrivateTest_2626531.jpg")

image = io.imread(IMAGE_PATH)

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(6, 6),
                    cells_per_block=(2, 2), visualize=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Oryginał')

import numpy as np
hog_image_rescaled = np.power(hog_image, 1/2)

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Wizualizacja HOG')

plt.show()