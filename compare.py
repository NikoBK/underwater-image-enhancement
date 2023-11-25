import cv2
import numpy as np

feed1 = cv2.VideoCapture("output/Kridtgrav -Taburet_noCLAHE_output.avi")
feed2 = cv2.VideoCapture("output/Kridtgrav -Taburet_CLAHE_output.avi")
fps = int(feed1.get(cv2.CAP_PROP_FPS))
w,h = (1280, 720)
res = (2*w, h)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out_path = f"output/compare.avi"
output = cv2.VideoWriter(out_path, fourcc, fps, res)

key = None
while(key != 27):
    ret,frame1 = feed1.read()
    if not ret:
        break
    ret,frame2 = feed2.read()
    if not ret:
        break
    comb = np.hstack((frame1,frame2))
    output.write(comb)
    cv2.imshow("frame",comb)
    key = cv2.waitKey(1)

feed1.release()
feed2.release()
output.release()
cv2.destroyAllWindows()