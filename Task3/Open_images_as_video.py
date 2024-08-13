import cv2
import sys
import numpy as np


filename = r"Computer Vision - Project\Github\Task 3\MovieRenders\NewLevelSequence."

i=0
while True:
# Capture frame-by-frame
     if i<10:
      frame = cv2.imread(filename+'000'+str(i)+'.jpeg')
     elif i<100:
      frame = cv2.imread(filename+'00'+str(i)+'.jpeg')
     elif i<1000:
      frame = cv2.imread(filename+'0'+str(i)+'.jpeg')
     else:
      frame = cv2.imread(filename+str(i)+'.jpeg')
     i = i+1
     # if frame is read correctly ret is True
     if frame is None:
        print("Could not read the image.")
        break
     
     # Display the resulting frame
     cv2.imshow('frame', frame)
     if cv2.waitKey(1) == ord('q'):
         break
