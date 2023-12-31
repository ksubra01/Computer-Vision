import cv2
import numpy as np
import matplotlib.pyplot as plt

vid = cv2.VideoCapture("D:/Sem 3/EEE515/HW1/Problem_6/livingroom.mov")


if not vid.isOpened():
    print("Error: Could not open video file.")
    exit()


# vid.set(cv2.CAP_PROP_POS_MSEC, 1000)  
# ret, reference_frame = vid.read()
# if not ret:
#     print("Error: Could not read the reference frame.")
#     exit()

frame_width = int(vid.get(3))
frame_height = int(vid.get(4))
fps = int(vid.get(5))

print(fps)

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


fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for H.264 codec
out = cv2.VideoWriter('difference_video.mov', fourcc, fps, (frame_width, frame_height))

while True:
    # Read a frame from the video
    ret, frame = vid.read()

    # Check if we have reached the end of the video
    if not ret:
        break

   
    difference = cv2.absdiff(reference_frame, frame)

    # Write the difference frame to the output video file
    out.write(difference)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
vid.release()
out.release()

cv2.destroyAllWindows()



