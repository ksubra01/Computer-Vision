import cv2
import numpy as np
import matplotlib.pyplot as plt



def Gaussian_pyramid(img, no_of_layers):
    new = img.copy()
    gp = [new]
    for i in range(no_of_layers):
        new = cv2.pyrDown(new)
        gp.append(new)

    return gp

def Laplacian_pyramid(gp, no_of_layers):
    new = gp[-1]
    lp = []
    for i in range(no_of_layers, 0, -1):

        expand = cv2.pyrUp(gp[i])
        laplacian =cv2.subtract(gp[i-1], expand)
        lp.append(laplacian)
    return lp

def multiresolutionblending(img1,img2):
    no_of_layers = 5
    center_rect_size = (200, 200)

    image_height, image_width,_ = img1.shape  # Get the height and width of the image

    # Calculate the coordinates for the center rectangle
    center_x = image_width // 2
    center_y = image_height // 2
    top_left = (center_x - center_rect_size[0] // 2, center_y - center_rect_size[1] // 2)
    bottom_right = (center_x + center_rect_size[0] // 2, center_y + center_rect_size[1] // 2)

    # Create a mask with 1s on the left half of the rectangle and 0s on the right half
    mask = np.zeros((image_height, image_width,3), dtype=np.uint8)
    mask[:, :center_x] = 1

    normalized_mask = mask

    opp_mask = 1 - normalized_mask
    img1 = img1 * mask
    img2 = img2 * opp_mask
   # plt.imshow(img1)
    # plt.imshow(img2)
    #plt.show()

    kk= img1+img2
    # plt.imshow(kk)
    # plt.show()


    # ## For image 1
    # gp_img1 = Gaussian_pyramid(img1, no_of_layers)
   


    # lp_img1 = Laplacian_pyramid(gp_img1, no_of_layers)

    # gp_img2 = Gaussian_pyramid(img2, no_of_layers)
    # lp_img2 = Laplacian_pyramid(gp_img2, no_of_layers)

    # lp_final = []
    # gp_final = []


    gppp= Gaussian_pyramid(kk, no_of_layers)
    lpp = Laplacian_pyramid(gppp,no_of_layers )





    # for i in range(len(lp_img1)):
    #     lp_final.append(lp_img1[i] + lp_img2[i])
    #     gp_final.append(gp_img1[i] + gp_img2[i])
    
    # plt.imshow(gp_final[3])
    # plt.show()
    

    # ll = gp_final[-1]
    # lol = []
    # lol.append(ll)
    # for i in range(0,len(lp_final)-1):
    #     temp = cv2.pyrUp(lol[i])
    #     lol.append(np.add(temp,lp_final[i+1]))

    ll = gppp[-1]
    lol = []
    lol.append(ll)
    for i in range(0,len(lpp)):

        temp = cv2.pyrUp(lol[i])
        lol.append(np.add(temp,lpp[i]))



        
    print(lol[-1].shape)

    cv2.imshow("Lol",lol[-1])
    




fed = cv2.imread("D:/Sem 3/EEE515/HW1/Problem_5/images/matt2.png") #375x462
djk = cv2.imread("D:/Sem 3/EEE515/HW1/Problem_5/images/mark2.png") #375x472

# fed = cv2.cvtColor(fed, cv2.COLOR_BGR2GRAY)
# djk = cv2.cvtColor(djk, cv2.COLOR_BGR2GRAY)
row,col = fed.shape[:2]
djk = cv2.resize(djk,(col,row))



# for i in range(len(djk)):
#     for j in range(len(djk[i])):
#         if djk[i,j] > 215:
#             djk[i,j] = 0
#         else:
#             continue

# fed1 = np.zeros((1,462),dtype=np.uint8)
# djk1 = np.zeros((1,462),dtype=np.uint8)


fed1 = np.zeros((8,375,3),dtype=np.uint8)
djk1 = np.zeros((8,375,3),dtype=np.uint8)

print(np.shape(fed))

print(np.shape(djk))


print(np.shape(fed1))

print(np.shape(djk1))


fed=np.concatenate((fed, fed1),axis=0)
djk=np.concatenate((djk, djk1),axis=0)


fed1 = np.zeros((480,9,3),dtype=np.uint8)
djk1 = np.zeros((480,9,3),dtype=np.uint8)

fed=np.concatenate((fed, fed1),axis=1)
djk=np.concatenate((djk, djk1),axis=1)

print(np.shape(fed))

print(np.shape(djk))

multiresolutionblending(fed,djk)

cv2.waitKey(0)


