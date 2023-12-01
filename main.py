import cv2
import numpy as np
from keras.models import load_model 

video = cv2.VideoCapture(0,cv2.CAP_DSHOW) # Inicializando camera

def preProcess(img): 
    imgPre = cv2.GaussianBlur(img, (5,5), 3)
    imgPre = cv2.Canny (imgPre, 90, 140)
    kernel = np.ones((4,4), np.unit8)
    imgPre = cv2.dilate(imgPre, kernel, iterations = 2 )
    imgPre = cv2.erode(imgPre, kernel, iterations = 2 )
    return imgPre

while True: 
    _,img = video.read()
    img = cv2.resize(img,(640, 480))
    imgPre  = preProcess (img)

    cv2.imshow("window_name", img) 
    cv2.imshow("window_name", imgPre) 
    cv2.waitKey(1)