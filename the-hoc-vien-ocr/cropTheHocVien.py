import argparse
import cv2
from cropper.cropper import crop_card
from detector.detector import detect_info
from reader import reader
import matplotlib.pyplot as plt
import numpy as np
import sys
import pytesseract
from PIL import Image
import os

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", default="image",
                help="Path to the folder contain images to be scanned")
args = vars(ap.parse_args())


folder = args["folder"]
newFolder = folder +"/cropped"
if not os.path.exists(newFolder):
        os.makedirs(newFolder)
lsImg = [folder +"/" + x for x in os.listdir(folder) if x.endswith(".jpg")]
for i in lsImg:
    img = cv2.imread(i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    croppedImg = i.replace(folder,newFolder)
    print(croppedImg)
    warped = crop_card(i)
    cv2.imwrite(croppedImg, warped)

