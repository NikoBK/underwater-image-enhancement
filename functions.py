import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

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