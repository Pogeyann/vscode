#Importing the dependencies

import cv2
import pandas as pd
import numpy as np

input = cv2.imread('/home/mush/Computer_vision/Project/WhatsApp Image 2023-01-17 at 18.05.13.jpeg')
cv2.imshow('image',input)
cv2.waitkey(5000)
cv2.destroyAllWindows()

# Classifier training 

classifier = cv2.CascadeClassifier('/home/mush/Computer_vision/Project/haarcascade_fullbody.xml')

image = cv2.imread('/home/mush/Computer_vision/Project/WhatsApp Image 2023-01-17 at 18.05.13.jpeg')

def pedestrian_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #To detect full body in the gray scale image we use detectMultiscale
    pedestrians = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

    total_detections = 0
    #Detection of rectangle
    for (x,y,w,h) in pedestrians:
        
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        
        font = cv2.FONT_HERSHEY_DUPLEX
        
        cv2.putText(frame, 'person', (x+5, y-5), font, 0.5, (0,0,255),1)
        # 0.5 and 1 are size and thickness of the test respectively
        total_detections +=1

    #Processing the image with detections
    cv2.imshow('pedestrian Detection', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print('Total pedestrian Detected: ', total_detections)

output = pedestrian_detection(image)


# Pedestrian detection in a video

cap = cv2.VideoCapture('/home/mush/Computer_vision/Project/WhatsApp Video 2023-01-17 at 13.07.42.mp4')

while True:
    success, frame = cap.read()
    cv2.imshow('Video', frame)

    if cv2.waitKey(5) & 0xFF==ord('d'):
        break


cap.release()
cv2.destroyAllWindows()

classifier = cv2.CascadeClassifier('/home/mush/Computer_vision/Project/haarcascade_fullbody.xml')

#Video  capture and reading

#Video writing , pedestrian detection drawing rectangle aroung detection

cap = cv2.VideoCapture('/home/mush/Computer_vision/Project/WhatsApp Video 2023-01-17 at 13.07.42.mp4')

ret, frame = cap.read()
frame_height,frame_width,'_' = frame.shape

# (height, width & no.of color channels) is then determined using attribute 
# and assigned to the variables
# Defining the codec and creating a video writer object

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output_video1.avi', fourcc, 30,(frame_width, frame_height))

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BG2GRAY)

    #Detecting pedestrians in the frame

    pedestrians = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


    for (x,y,w,h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255, 0),2)
        cv2.putText(frame, 'pedestrian', (x,y - 10),cv2.FONT_HERSHEY_COMPLEX,0.9, (0,255,0),2)


    out.write(frame)


#Calculates pedestrian count

cap = cv2.VideoCapture('/home/mush/Computer_vision/Project/WhatsApp Video 2023-01-17 at 13.07.42.mp4')
ret, frame = cap.read()
frame_height, frame_width, _ = frame.shape

#Define the codec and create a video writer object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
ou = cv