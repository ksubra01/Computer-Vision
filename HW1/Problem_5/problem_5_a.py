import cv2
import numpy as np
from matplotlib import pyplot as plt
path = r'D:/Sem 3/EEE515/HW1/Problem_5/ww.jpg'
path2 = r'D:/Sem 3/EEE515/HW1/Problem_5/ss.jpg'
image= cv2.imread(path)
image2 = cv2.imread(path2)

# mask1 = image2
mask1 = np.zeros(image.shape, dtype=np.uint8)
mask2 = np.zeros(image2.shape, dtype=np.uint8)

mask1 = cv2.rectangle(mask1, (0, 384), (176, 0), (255,255,255), thickness=-1)

mask2 = cv2.rectangle(mask2, (177, 384), (384, 0), (255,255,255), thickness=-1)

# image, centre coordinates, radius, color, -1 means rather than thickness fill the shape 

# Mask input image with binary mask
result1 = cv2.bitwise_and(image, mask1)

result2 = cv2.bitwise_and(image2, mask2)

fin= result1+result2

plt.imshow(fin)
plt.show()
cv2.imwrite('fin.jpg', fin)