import cv2
import numpy as np
import matplotlib.pyplot as plt



########### Problem 4.1 ####################

# img = cv2.imread("D:/Sem 3/EEE515/HW1/Problem_4/image1.jpg")

# # Convert to gray scale
# gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Find the Fourier Transform
# fft = np.fft.fft2(gray_scale)

# fft_shift = np.fft.fftshift(fft)
# phase = np.angle(fft_shift)
# magnitude = np.abs(fft_shift)

# magnitude = 20*np.log1p(magnitude)
# magnitude = ((magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude)) * 255).astype(np.uint8)

# cv2.imshow("Original Image",img)
# cv2.imshow("Gray scale image",gray_scale)
# cv2.imshow("Phase of image",phase)
# cv2.imshow("Magnitude of image",magnitude)
# cv2.waitKey(0)

##########################################

###### Problem 4.2 #######################


# Low Pass Filter
# sigma = 10
# s = 8*sigma+1
# if not s % 2:
#     s = s + 1

# center = s // 2
# kernel = np.zeros((s, s))
# for y in range(s):
#     for x in range(s):
#         diff = (y - center) ** 2 + (x - center) ** 2
#         kernel[y, x] = np.exp(-diff / (2 * sigma ** 2))

# kernel = kernel / np.sum(kernel)
# img_h,img_w = gray_scale.shape
# k_h,k_w = kernel.shape
# padded_k = np.zeros(gray_scale.shape)
# start_h = (img_h - k_h) // 2
# start_w = (img_w - k_w) // 2
# padded_k[start_h:start_h+k_h, start_w:start_w+k_w] = kernel
# ker = np.fft.fft2(padded_k)

# low_pass = fft * ker
# high_pass = fft - low_pass

# mag_low = np.abs(np.fft.fftshift(low_pass))
# mag_low = 20*np.log1p(mag_low)
# mag_low = ((mag_low - np.min(mag_low)) / (np.max(mag_low) - np.min(mag_low)) * 255).astype(np.uint8)


# mag_high = np.abs(np.fft.fftshift(high_pass))
# mag_high = 20*np.log1p(mag_high)
# mag_high = ((mag_high - np.min(mag_high)) / (np.max(mag_high) - np.min(mag_high)) * 255).astype(np.uint8)


# out_low_pass = np.fft.fftshift(np.fft.ifft2(low_pass))
# out_high_pass = np.fft.fftshift(np.fft.ifft2(high_pass))

# out_low_pass = out_low_pass.astype(np.uint8)
# out_high_pass = out_high_pass.astype(np.uint8)




# # cv2.imshow(out_band_pass)



# sigma = 30
# s = 8*sigma+1
# if not s % 2:
#     s = s + 1

# center = s // 2
# kernel = np.zeros((s, s))
# for y in range(s):
#     for x in range(s):
#         diff = (y - center) ** 2 + (x - center) ** 2
#         kernel[y, x] = np.exp(-diff / (2 * sigma ** 2))

# kernel = kernel / np.sum(kernel)
# img_h,img_w = gray_scale.shape
# k_h,k_w = kernel.shape
# padded_k_2 = np.zeros(gray_scale.shape)
# start_h = (img_h - k_h) // 2
# start_w = (img_w - k_w) // 2
# padded_k_2[start_h:start_h+k_h, start_w:start_w+k_w] = kernel

# band_pass = padded_k_2 - padded_k

# band_pass = fft * band_pass
# b_p = np.abs(np.fft.fftshift(band_pass))
# b_p = 20*np.log1p(b_p)
# b_p = ((b_p - np.min(b_p)) / (np.max(b_p) - np.min(b_p)) * 255).astype(np.uint8)


# b_p_i = np.fft.fftshift(np.fft.ifft2(b_p))
# b_p_i = b_p_i.astype(np.uint8)


# cv2.imshow("Band_pass_mag",b_p)
# cv2.imshow("Low_pass_mag",mag_low)
# cv2.imshow("High_pass_mag",mag_high)
# cv2.imshow("Low_pass_image",out_low_pass)
# cv2.imshow("High_pass_image",out_high_pass)
# cv2.imshow("Band_pass_image",b_p_i)
# cv2.waitKey(0)





# lpf = cv2.GaussianBlur(magnitude,(5,5),3)

# # ## High pass filter


# hpf = magnitude - lpf


# ## diagonal Band pass filter
# dbpf_1 = cv2.GaussianBlur(magnitude,(3,3),1)
# dbpf_2 = cv2.GaussianBlur(magnitude,(3,3),0)
# diff = dbpf_1 - dbpf_2



# cv2.imshow("Original magnitude",magnitude)
# cv2.imshow("Filtered magnitude (Low pass)",lpf)
# cv2.imshow("Filtered magnitude (High pass)",hpf)
# cv2.imshow("Filtered magnitude (Band pass filter)",diff)
# cv2.waitKey(0)

########################################################

################# Problem 4.3 ####################

# img1 = cv2.imread("D:/Sem 3/EEE515/HW1/Problem_4/1.jpg")
# img2 = cv2.imread("D:/Sem 3/EEE515/HW1/Problem_4/2.jpg")


# gray_1 =  cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# gray_2 =  cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# rows, cols = gray_1.shape
# gray_2 = cv2.resize(gray_2, (cols, rows))


# ## for image 1
# fft_1 = np.fft.fft2(gray_1)
# fftshift_1 = np.fft.fftshift(fft_1)
# fftshift_1_phase = np.angle(fftshift_1)
# fftshift_1_mag = np.abs(fftshift_1)


# ## for image 2
# fft_2 = np.fft.fft2(gray_2)
# fftshift_2 = np.fft.fftshift(fft_2)
# fftshift_2_phase = np.angle(fftshift_2)
# fftshift_2_mag = np.abs(fftshift_2)


# fftshift_2_phase = fftshift_2_phase*5
# fftshift_1_phase = fftshift_1_phase*5

# ## Swap the phases of the images
# img1_exp_img2 = fftshift_1_mag * (np.exp(1j*fftshift_2_phase))
# img2_exp_img1 = fftshift_2_mag * (np.exp(1j*fftshift_1_phase))

# ## Convert the images back to spatial domain
# img1_back = np.uint8(np.abs(np.fft.ifft2(img1_exp_img2)))
# img2_back = np.uint8(np.abs(np.fft.ifft2(img2_exp_img1)))


# # cv2.imshow("Phase 1",fftshift_1_phase)
# # cv2.imshow("Phase 2",fftshift_2_phase)
# cv2.imshow("New image 1",img1_back)
# cv2.imshow("New image 2",img2_back)
# cv2.waitKey(0)

###################################################

################ problem 4.4 #######################

tree = cv2.imread("D:/Sem 3/EEE515/prithvi/HW1_4d/fish.bmp")
cacti = cv2.imread("D:/Sem 3/EEE515/prithvi/HW1_4d/submarine.bmp")

row,col = tree.shape[:2]
cacti = cv2.resize(cacti,(col,row))

def low_pass_fourier(img,sig):
    s = 8*sig+1
    if not s % 2:
        s = s + 1

    center = s // 2
    #print(s)
    kernel = np.zeros((s, s))
    for y in range(s):
        for x in range(s):
            diff = (y - center) ** 2 + (x - center) ** 2
            kernel[y, x] = np.exp(-diff / (2 * sig ** 2))

    kernel = kernel / np.sum(kernel)
    print(kernel)

    img_h,img_w = img.shape[:2]
    k_h,k_w = kernel.shape[:2]
    padded_k = np.zeros(img.shape[:2])
    start_h = (img_h - k_h) // 2
    start_w = (img_w - k_w) // 2
    padded_k[start_h:start_h+k_h, start_w:start_w+k_w] = kernel
    kk = np.abs(np.fft.fftshift(np.fft.fft2(padded_k)))
    #kk = kernel
    out = np.zeros(img.shape)
    for i in range(3):
        fft_i = np.fft.fft2(img[:,:,i])
        fft_k = np.fft.fft2(padded_k)
        ans = fft_i * fft_k
        out[:,:,i] = np.fft.fftshift(np.fft.ifft2(ans)) /255
    return kk,out

kk,lpf = low_pass_fourier(tree,7)


kk = 20*np.log1p(kk)
kk = ((kk - np.min(kk)) / (np.max(kk) - np.min(kk)) * 255).astype(np.uint8)
cv2.imshow("Low",kk)
cv2.waitKey(0)

# hpf = (cacti/255) - low_pass_fourier(cacti,5)

# output = lpf + hpf

# cv2.imshow("ans",output)
# cv2.waitKey(0)


