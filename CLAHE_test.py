import cv2
import functions as f

feed = cv2.VideoCapture("output/Kridtgrav -Taburet_noCLAHE_output.avi")

fps = int(feed.get(cv2.CAP_PROP_FPS))
res = (1280, 720)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out_path = f"output/output_CLAHEtest.avi"
output = cv2.VideoWriter(out_path, fourcc, fps, res)

key = None
while(key != 27):
    ret,frame = feed.read()
    if not ret:
        break
    frame = cv2.resize(frame,res)
    frame_new = f.applyCLAHE(frame,1.0,(8,8))

    #output.write(frame_new)
    cv2.imshow("frame",frame_new)
    key = cv2.waitKey(1)

feed.release()
output.release()
cv2.destroyAllWindows()