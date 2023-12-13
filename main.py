import tkinter as tk
from tkinter import filedialog,messagebox,ttk
import cv2
import functions as f
import time as t
import sys
import os

filepath = None
filename = None
res_str = None
def openFile():
    global filepath
    global filename
    global filepath_text
    filepath_text.delete('1.0', tk.END)
    filepath = filedialog.askopenfilename(title="Chose video to process")
    filename = os.path.basename(filepath)
    if filename == "":
        filepath_text.insert(tk.END, "Chose video to process")
    else:
        filepath_text.insert(tk.END, "Current file:\n")
        filepath_text.insert(tk.END, filepath)

def endGUI():
    global filepath
    global filepath_text
    global res_str
    global error
    res_str = res_dropdown.get()
    if filepath == None or filepath == "":
        error.set("No filepath chosen!")
        return
    if res_str == "":
        error.set("No output resolution chosen!")
        return
    else:
        if not cc_var.get() and not CLAHE_var.get() and not dist_var.get():
            answer = messagebox.askyesno("Exit","No processing was chosen. Are you sure you want to continue?")
            if not answer:
                return
        window.destroy()

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        window.destroy()
        sys.exit()

def pointVariables():
    if dist_var.get():
        window.geometry('200x450')
        units = ["mm", "cm", "m"]
        unit_label = tk.Label(pointvar_frame, text="Chose display unit")
        unit_label.pack()
        unit_dropdown = ttk.Combobox(pointvar_frame, values=units)
        unit_dropdown.pack()
    else:
        for widget in pointvar_frame.winfo_children():
            widget.destroy()
        window.geometry('200x350')
        tmp = tk.Frame(pointvar_frame, width=1, height=1, borderwidth=0, highlightthickness=0)
        tmp.pack()
window = tk.Tk()
window.title('My Window')
window.geometry('200x350')
window.resizable(False, False)

browse_label = tk.Label(window,text="Chose video file to process")
browse_label.pack()
browse_btn = tk.Button(window, text="Browse", command=openFile)
browse_btn.pack()

filepath_label = tk.Label(window,text="Current file:")
filepath_label.pack()
filepath_text = tk.Text(window, height=5, width=52)
filepath_text.pack()
filepath_text.insert(tk.END, "None!")

cc_var = tk.BooleanVar()
CLAHE_var = tk.BooleanVar()
dist_var = tk.BooleanVar()
unit_var = tk.IntVar()

options_label = tk.Label(window,text="Video Processing Options:")
options_label.pack()
check_cc = tk.Checkbutton(window, text='Apply Color Correction', variable=cc_var, onvalue=1, offvalue=0, command=None)
check_cc.pack()
check_CLAHE = tk.Checkbutton(window, text='Apply CLAHE', variable=CLAHE_var, onvalue=1, offvalue=0, command=None)
check_CLAHE.pack()
check_dist = tk.Checkbutton(window, text='Find distance to laser points', variable=dist_var, onvalue=1, offvalue=0, command=pointVariables)
check_dist.pack()

pointvar_frame = tk.Frame(window)
pointvar_frame.pack()

res_label = tk.Label(window,text="Chose output resolution")
res_label.pack()
resolutions = ["1280 × 720","1920 × 1080","2560 × 1440","3840 × 2160"]
res_dropdown = ttk.Combobox(window,values=resolutions)
res_dropdown.pack()

process_btn = tk.Button(window, text="Process video", command=endGUI)
process_btn.pack(pady=5)
error = tk.StringVar()
error_label = tk.Label(window,textvariable=error,fg='red')
error_label.pack()

window.protocol("WM_DELETE_WINDOW", on_closing)
window.mainloop()

if filepath == None or filepath == "":
    print("No file was selected. Please run program again!")
    sys.exit()
if res_str == "":
    print("No output resolution was chosen. Please run program again!")
    sys.exit()

#Specifying filepath and creating file name
feed = cv2.VideoCapture(filepath)
filename,sep,filetype = filename.partition(".")
filetype = sep + filetype

#Grabbing FPS from source and defining variables
fps = int(feed.get(cv2.CAP_PROP_FPS))
w,sep,h = res_str.partition(" × ")
res = (int(w), int(h))
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out_path = f"output/{filename}_output.avi"

#Creating the Video Writer
output = cv2.VideoWriter(out_path, fourcc, fps, res)

#Entering loop to save video, and defining variables for scale and time
key = None
scale = 0.2
start = t.time()

unit = unit_var.get()
unitArr = ["mm","cm","m"]
while(key != 27):
    #Reading source frame by frame
    ret,frame = feed.read()
    if not ret:
        break

    #Applying processing
    frame_start = t.time()
    frame_new = cv2.resize(frame, res)
    if cc_var.get():
        frame_new = f.compensateChannels(frame_new)
        frame_new = f.whiteBalance(frame_new)
    if CLAHE_var.get():
        frame_new = f.applyCLAHE(frame_new,4,(8,8))
    if dist_var.get():
        #Insert function for calculating distance here
        pointOne, pointTwo, frameThresh = f.getPoints(frame_new)
        cv2.line(frame_new,pointOne,pointTwo,(255,255,255),5)
        distance = f.calcDist(pointOne, pointTwo, unitArr[unit],frame_new)
        if frame_new.shape[0] == 2160:
            cv2.putText(frame_new,f"{distance} {unitArr[unit]}",(1520,2020),cv2.FONT_HERSHEY_SIMPLEX,5,(255,255,255),5)
        if frame_new.shape[0] == 1440:
            cv2.putText(frame, f"{distance} {unitArr[unit]}", (1050, 1350), cv2.FONT_HERSHEY_SIMPLEX, 3,(255, 255, 255), 3)
        if frame_new.shape[0] == 1080:
            cv2.putText(frame_new, f"{distance} {unitArr[unit]}", (820, 1010), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),2)
        if frame_new.shape[0] == 720:
            cv2.putText(frame_new, f"{distance} {unitArr[unit]}", (500, 680), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),2)
    frame_end = t.time()
    #Writing video and displaying current frame
    output.write(frame_new)
    cv2.imshow("frame",frame_new)
    key = cv2.waitKey(1)

#Calculating process time and closing all windows
end = t.time()
dur = end - start
feed.release()
output.release()
cv2.destroyAllWindows()

window = tk.Tk()
window.eval("tk::PlaceWindow %s center" % window.winfo_toplevel())
window.withdraw()

messagebox.showinfo("Video done processing!",f"Video took {round(dur,2)} second to process and is saved at {out_path}")

window.deiconify()
window.destroy()
window.quit()