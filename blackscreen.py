import cv2
import time
import numpy as np

#to save the output in .avi format
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter("output.avi", fourcc, 20.0, (640, 480))

#to start the webcam
capture = cv2.VideoCapture(0)

#to start the program after 2 seconds
time.sleep(2)

bg = 0

#to capture the background for 60 frames
for i in range(60):
    ret, bg = capture.read()

#to flip the background
bg = np.flip(bg, axis = 1)

#to read the captured frames until the camera is open
while(capture.isOpened()):
    ret, img = capture.read()
    
    if not ret:
        break
    img = np.flip(img, axis = 1)
    
    #to convert the color from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #generate masks to detect the color black
    lower_b = np.array([30, 30, 0])
    upper_b = np.array([104, 153, 70])

    mask1 = cv2.inRange(hsv, lower_b, upper_b)

    lower_b = np.array([30, 30, 0])
    upper_b = np.array([104, 153, 70])

    mask2 = cv2.inRange(hsv, lower_b, upper_b)
    
    mask1 = mask1 + mask2

    #to open the image where ther is the color of mask1
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8()))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8()))

    #to remove the color b from the frame
    mask2 = cv2.bitwise_not(mask1)

    #to create 2 resolutions, 1 for the bg, another for the mask
    res1 = cv2.bitwise_and(img, img, mask = mask2)
    res2 = cv2.bitwise_and(bg, bg, mask = mask1)

    #to generate the output by merging res1 and res2
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)
    output.write(final_output)

    #to display the output
    cv2. imshow("invisibility cloak", final_output)
    cv2.waitKey(1)

#to stop the camera and close all the windows that were opened
capture.release()
out.release()
cv2.destroyAllWindows()