import cv2
import numpy as np
import matplotlib.pyplot as plt

fed = cv2.imread("D:/Sem 3/EEE515/HW1/Problem_5/images/matt2.png") #375x462
djk = cv2.imread("D:/Sem 3/EEE515/HW1/Problem_5/images/mark2.png")
fed = cv2.cvtColor(fed, cv2.COLOR_BGR2GRAY)
djk = cv2.cvtColor(djk, cv2.COLOR_BGR2GRAY)
row,col = fed.shape
djk = cv2.resize(djk,(col,row))

# print(djk[0,0])

# for i in range(len(djk)):
#     for j in range(len(djk[i])):
#         if djk[i,j] > 215:
#             djk[i,j] = 0
#         else:
#             continue


center_rect_size = (200, 200)

image_height, image_width = fed.shape  # Get the height and width of the image

# Calculate the coordinates for the center rectangle
center_x = image_width // 2
center_y = image_height // 2
top_left = (center_x - center_rect_size[0] // 2, center_y - center_rect_size[1] // 2)
bottom_right = (center_x + center_rect_size[0] // 2, center_y + center_rect_size[1] // 2)

# Create a mask with 1s on the left half of the rectangle and 0s on the right half
mask = np.zeros((image_height, image_width), dtype=np.uint8)
mask[:, :center_x] = 1

normalized_mask = mask



opp_mask = 1 - normalized_mask

fed = fed.astype(np.float32)
djk = djk.astype(np.float32)

blended = np.zeros((fed.shape))
for i in range(3):
    # blended[:,:,i] = (fed[:,:,i] * normalized_mask + djk[:,:,i] * opp_mask)
    blended = (fed * normalized_mask + djk * opp_mask)

blended = np.round(blended).astype(np.uint8)

cv2.imshow("blended.jpg",blended)
cv2.imwrite("Fed.jpg",fed)
cv2.imwrite("djk.jpg",djk)
cv2.waitKey(0)