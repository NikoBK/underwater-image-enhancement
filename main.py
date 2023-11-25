import cv2
import functions as f
import time as t

#Specifying filepath and creating file name
path = "Assets/GLDS1017_144639752.mp4"
feed = cv2.VideoCapture(path)
filename = path.replace("Assets/","")
filename = filename.replace(".mp4","")

#Grabbing FPS from source and defining variables
fps = int(feed.get(cv2.CAP_PROP_FPS))
res = (1280, 720)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out_path = f"output/{filename}_noCLAHE_output.avi"

#Creating the Video Writer
output = cv2.VideoWriter(out_path, fourcc, fps, res)

#Entering loop to save video, and defining variables for scale and time
key = None
scale = 0.2
start = t.time()
while(key != 27):
    #Reading source frame by frame
    ret,frame = feed.read()
    if not ret:
        break

    #Applying processing
    frame = cv2.resize(frame, res)
    frame_new = f.compensateChannels(frame)
    frame_new = f.whiteBalance(frame_new)
    #frame_new = f.applyCLAHE(frame_new)

    #Writing video and displaying current frame
    output.write(frame_new)
    cv2.imshow("frame",frame_new)
    key = cv2.waitKey(1)

#Calculating process time and closing all windows
end = t.time()
dur = end - start
print(f"Video took {round(dur,2)} second to process!")
feed.release()
output.release()
cv2.destroyAllWindows()