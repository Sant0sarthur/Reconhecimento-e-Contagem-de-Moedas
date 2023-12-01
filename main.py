import cv2
import numpy as np
from keras.models import load_model 

video = cv2.VideoCapture(0,cv2.CAP_DSHOW) # Inicializando camera

def preProcess(img): 
    imgPre = cv2.GaussianBlur(img, (5,5), 3)
    imgPre = cv2.Canny (imgPre, 90, 140)
    kernel = np.ones((4,4), np.uint8)
    imgPre = cv2.dilate(imgPre, kernel, iterations = 2 )
    imgPre = cv2.erode(imgPre, kernel, iterations = 1 )
    return imgPre




while True: 
    _,img = video.read()
    img = cv2.resize(img,(500, 380))
    imgPre  = preProcess (img)
    countors, hi = cv2.findContours(imgPre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in countors: 
        x, y, w, h = cv2.boundingRect(cnt)     

        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)   

    cv2.imshow("window_name1", img) 
    cv2.imshow("window_name2", imgPre) 
    cv2.waitKey(1)