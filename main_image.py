import cv2
import numpy as np
import functions as f

filepath = "Assets/Kridtgrav-lflaske - frame at 0m30s.jpg"
filename,sep,format = filepath.partition(".")
folder,sep,filename = filename.partition("/")
img = cv2.imread(filepath)

pointOne,pointTwo,thresh = f.getPoints(img)

thresh = cv2.resize(thresh,(1280,720))
cv2.imshow("thresh",thresh)
cv2.waitKey()
