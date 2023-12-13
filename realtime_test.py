import cv2
import time as t
import functions as f
import pandas as pd

filepath = "Assets/Kridtgrav-Ølflaske.mp4"
feed = cv2.VideoCapture(filepath)

frametimes = []
key = None
resolutions = ["1280 × 720","1920 × 1080","2560 × 1440","3840 × 2160"]
while(key != 27):
    ret,frame = feed.read()
    if not ret:
        break

    frame_start = t.time()
    frame_new = cv2.resize(frame, (1280,720))
    #frame_new = f.compensateChannels(frame_new)
    #frame_new = f.whiteBalance(frame_new)
    #frame_new = f.applyCLAHE(frame_new, 4, (8, 8))
    pointOne, pointTwo, frameThresh = f.getPoints(frame_new)
    distance = f.calcDist(pointOne, pointTwo, "mm",frame_new)
    cv2.putText(frame_new, f"{distance} mm",(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)
    cv2.imshow("frame",frame_new)

    frame_end = t.time()
    frame_time = (frame_end - frame_start) * 1000
    frametimes.append(frame_time)

    key = cv2.waitKey(1)

feed.release()
cv2.destroyAllWindows()

df = pd.DataFrame(frametimes)
df.to_excel(excel_writer = "output/frametimes.xlsx")