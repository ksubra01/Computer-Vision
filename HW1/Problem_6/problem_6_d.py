import cv2
import numpy as np
import matplotlib.pyplot as plt

vid = cv2.VideoCapture("D:/Sem 3/EEE515/HW1/Problem_6/livingroom.mov")

vid_2 = cv2.VideoCapture("D:/Sem 3/EEE515/HW1/Problem_6/livingroom.mov")

frame_count = 0
average_frame = None


num_frames_to_average = 50

while frame_count < num_frames_to_average:

    ret, frame = vid.read()

   
    if not ret:
        break

   
    frame_count += 1

   
    if average_frame is None:
        average_frame = np.zeros_like(frame, dtype=np.float32)
    
    average_frame += frame.astype(np.float32)


average_frame /= num_frames_to_average
reference_frame = average_frame.astype(np.uint8)

# i = 1
# while True:
#         # Read a frame from the video
#     ret, frame = vid.read()

#     # Check if we have reached the end of the video
#     if not ret or i > 448:
#         req_frame = frame
#         break
#     i = i+1
# print(i)
# i = 1
# while True:
#         # Read a frame from the video
#     ret_2, frame_2 = vid_2.read()

#     # Check if we have reached the end of the video
#     if not ret_2 or i == 1516:
#         reference_frame = frame_2
#         break
#     i = i+1

# print(i)

# cv2.imshow("Gooo",reference_frame)
# cv2.waitKey(0)
vid.set(cv2.CAP_PROP_POS_MSEC, 18000)  
ret, req_frame = vid.read()
if not ret:
    print("Error: Could not read the reference frame.")
    exit()

img_result =  reference_frame - req_frame



cv2.imshow("Frame",img_result*3)
cv2.waitKey(0)

