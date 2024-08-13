from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

#load face data and corresponding labels from pickle files
with open('data/names.pkl', 'rb') as f:
    LABELS=pickle.load(f)
with open('data/faces_data.pkl', 'rb') as f:
    FACES=pickle.load(f)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

imgBackground=cv2.imread("background.png")


while True:
    ret,frame=video.read()
    #covert frame to grayscale for face detection
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detect faces in the grayscale image
    faces=facedetect.detectMultiScale(gray, 1.3 ,5)
    #process each detected face
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        resized_img=cv2.resize(crop_img, (50,50)).flatten().reshape(1,-1)
        output=knn.predict(resized_img)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame,(x,y), (x+w, y+h), (204,153,255),2)
        cv2.rectangle(frame, (x,y-40), (x+w, y), (204,153,255), -1)
        cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (204,153,255), 1)
    imgBackground[162:162 + 480, 55:55 + 640] = frame
    #show video feed with the annotated frame
    cv2.imshow("Frame",imgBackground)
    k=cv2.waitKey(1)
    #exit loop when q is pressed
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()
