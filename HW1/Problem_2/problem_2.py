import cv2
import numpy as np
import matplotlib.pyplot as plt
import ipdb
from PIL import Image


# img = cv2.imread("D:/Sem 3/EEE515/HW1/Problem 2/Test.png")

# plt.imshow(img)
# plt.show()


img_2 = cv2.imread("D:/Sem 3/EEE515/HW1/Problem_4/Mine/figfig.png")

img_3 = cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)
img_4 = cv2.cvtColor(img_2,cv2.COLOR_BGR2RGB)

cv2.imshow("img",img_3)
cv2.imshow("imgg",img_4)
cv2.waitKey(0)


#### Problem 2.2 #######


# img = cv2.imread("D:/Sem 3/EEE515/HW1/Problem 2/Test.png")

# cv2.imshow("Original Image",img)

# img_cropped = img[400:500,400:500]

# cv2.imshow("cropped",img_cropped)
# cv2.waitKey(0)



##### Problem 2.3 ####

### Part a ###

# img = cv2.imread("D:/Sem 3/EEE515/HW1/Problem 2/Test.png")

# ### Part b - Downsample the image ###

width = int(img.shape[1] / 10)
height = int(img.shape[0] / 10)
dim = (width, height)
Downsampled = cv2.resize(img,dim)

# cv2.imwrite("Downsampled.jpg", Downsampled)

# cv2.imshow("Original Image", img)

# cv2.imshow("Donwsampled Image", Downsampled)

### Part c ###

# width = int(img.shape[1])
# height = int(img.shape[0])
# dim_2 = (width,height)
# Upsampled_NN = cv2.resize(Downsampled,dim_2, interpolation = cv2.INTER_NEAREST)
# Upsampled_cubic = cv2.resize(Downsampled, dim_2, interpolation = cv2.INTER_CUBIC)


# cv2.imwrite("Upsampled Image (Using Nearest Neighbour).jpg", Upsampled_NN)
# cv2.imwrite("Upsampled Image (Using Bicubic interpolation).jpg", Upsampled_cubic)

# cv2.imshow("Upsampled Image (Using Nearest Neighbour)",Upsampled_NN)
# cv2.imshow("Upsampled Image (Using Bicubic interpolation)",Upsampled_cubic)



# # ### Part d ###

# diff_nn = img - Upsampled_NN
# diff_cubic = img - Upsampled_cubic

# nn_sum = diff_nn.sum()
# cubic_sum = diff_cubic.sum()


# print("the number of pixels for the difference image between groud truth and Nearest Neighbour: ",nn_sum)
# print("the number of pixels for the difference image between groud truth and Cubic: ",cubic_sum)


# cv2.imwrite("Difference Image (Using Nearest Neighbour).jpg", diff_nn)
# cv2.imwrite("Difference Image (Using Bicubic interpolation).jpg", diff_cubic)


# cv2.waitKey(0)
# cv2.destroyAllWindows()