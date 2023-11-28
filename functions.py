import cv2
import numpy as np
import math

def compensateChannels(img):
    #Splitting the channels
    b,g,r = cv2.split(img)

    #Difining min and max for each channel
    b_min,b_max = np.min(b),np.max(b)
    g_min,g_max = np.min(g),np.max(g)
    r_min, r_max = np.min(r), np.max(r)

    #Converting from [0,255] to [0,1]
    b = (b - b_min) / (b_max - b_min)
    g = (g - g_min) / (g_max - g_min)
    r = (r - r_min) / (r_max - r_min)

    #Difining mean for each channel
    b_mean = np.mean(b)
    g_mean = np.mean(g)
    r_mean = np.mean(r)

    #Correcting B and R channel in relation to G channel
    b = b + (g_mean - b_mean) * (1-b)*g
    r = r + (g_mean - r_mean) * (1-r)*g

    #Converting back to [0,255]
    b = b * b_max
    g = g * g_max
    r = r * r_max

    #Clipping values to ensure we stay between 0 and 255
    b = np.clip(b, 0, 255)
    g = np.clip(g, 0, 255)
    r = np.clip(r, 0, 255)

    #Merging channels again and converting to 8-bit integer
    newImg = cv2.merge((b,g,r))
    newImg = newImg.astype(np.uint8)

    return newImg
def whiteBalance(img):
    # Splitting the channels
    b, g, r = cv2.split(img)

    #Converting to grayscale
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Defining mean for each channel and grayscale image
    b_mean = np.mean(b)
    g_mean = np.mean(g)
    r_mean = np.mean(r)
    gray_mean = np.mean(img_gray)

    #Converting to 32-bit integers
    b = b.astype(np.uint32)
    g = g.astype(np.uint32)
    r = r.astype(np.uint32)

    #performing white balancing using the Gray World algorithm
    b = b * (gray_mean/b_mean)
    g = g * (gray_mean/g_mean)
    r = r * (gray_mean/r_mean)

    # Clipping values to ensure we stay between 0 and 255, and converting back to 8-bit integers
    b = np.clip(b,0,255)
    g = np.clip(g, 0, 255)
    r = np.clip(r, 0, 255)

    # Merging channels again and converting to 8-bit integer
    newImg = cv2.merge((b, g, r))
    newImg = newImg.astype(np.uint8)

    return newImg

def applyCLAHE(img,clipLimit,tileGridSize):
    #Converting to LAB color space and splitting channels
    img_lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(img_lab)

    #Creating an CLAHE object and applying to L channel
    clahe = cv2.createCLAHE(clipLimit=clipLimit,tileGridSize=tileGridSize)
    l = clahe.apply(l)

    #Merging channels and converting back to BGR
    lab = cv2.merge((l,a,b))
    newImg = cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)

    return newImg

def getPoints(frame,thresh):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    try:
        contoursOne = contours[0]
        contoursTwo = contours[1]
    except:
        pointOne = [0,0]
        pointTwo = [0,0]
        return pointOne, pointTwo,thresh

    pointOne = []
    pointTwo = []
    for cont in range(contoursOne.shape[0]):
        y, x = contoursOne[cont][0]
        pointOne.append((y, x))
    for cont in range(contoursTwo.shape[0]):
        y, x = contoursTwo[cont][0]
        pointTwo.append((y, x))
    pointOne = np.array(pointOne)
    pointTwo = np.array(pointTwo)

    pointOneY = pointOne[:, 0]
    pointOneX = pointOne[:, 1]
    pointTwoY = pointTwo[:, 0]
    pointTwoX = pointTwo[:, 1]

    pointOneY = round(np.sum(pointOneY) / pointOneY.shape[0])
    pointOneX = round(np.sum(pointOneX) / pointOneX.shape[0])
    pointTwoY = round(np.sum(pointTwoY) / pointTwoY.shape[0])
    pointTwoX = round(np.sum(pointTwoX) / pointTwoX.shape[0])

    pointOne = [pointOneY, pointOneX]
    pointTwo = [pointTwoY, pointTwoX]

    return pointOne,pointTwo,thresh

def calcDist(pointLeft,pointRight,unit):
    if pointLeft == [0,0] or pointRight == [0,0]:
        distance = "No points found"
        return distance
    v = [(pointLeft[0] - pointRight[0]), (pointLeft[1] - pointRight[1])]
    frameW = 3840
    frameH = 2160
    sensorW = 6.17
    sensorH = 4.55
    pixelW = sensorW / frameW
    pixelH = sensorH / frameH
    fov = 150 * (math.pi / 180)
    vMM = [v[0] * pixelH, v[1] * pixelW]
    B = math.sqrt((vMM[0] ** 2) + (vMM[1] ** 2))  # Lenght of vector on screen (in mm)
    b = (sensorW/2) / math.tan((fov/2))
    G = 100  # mm. Real size of object
    g = (G * b / B)
    g = g + 65
    distance = round(g, 2)
    if unit == "mm":
        pass
    if unit == "cm":
        distance = distance / 10
        distance = round(distance,2)
    if unit == "m":
        distance = distance / 1000
        distance = round(distance, 2)
    return distance