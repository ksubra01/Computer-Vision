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
def Gaussian_up(gp):
    old = []
    for i in range(len(gp)):
        old.append(cv2.pyrUp(gp[i]))
    return old

def Laplacian_pyramid(gp, no_of_layers):
    new = gp[-1]
    lp = []
    for i in range(no_of_layers, 0, -1):
        expand = cv2.pyrUp(gp[i])
        laplacian =cv2.subtract(gp[i-1], expand)
        lp.append(laplacian)
    return lp

def print_pyr(p):
    for i in range(len(p)):
        cv2.imshow(f"level {i+1}",p[i])


if __name__ == '__main__':
    img = cv2.imread("D:/Sem 3/EEE515/HW1/Problem_5/test.jpg")
    imf = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    no_of_layers = 4
    gp = Gaussian_pyramid(img,no_of_layers)
    retreived_img = Gaussian_up(gp)
    lp = Laplacian_pyramid(gp,no_of_layers)
    #print_pyr(gp)
    #print_pyr(lp)
    #print_pyr(retreived_img)
    cv2.waitKey(0)









