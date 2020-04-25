import argparse
import cv2
from cropper.cropper import crop_card
from detector.detector import detect_info
from reader import reader
import matplotlib.pyplot as plt
import numpy as np
import sys
import pytesseract

import os
def show_img(img):
    cv2.imshow('', img)
    cv2.waitKey(0)


def plot_img(img):
    plt.imshow(img)
    plt.show()


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", default="image",
                help="Path to the image to be scanned")
args = vars(ap.parse_args())

folder = args["folder"]
originFolder = folder + "/cropped"
outputFolder = folder + "/ouput"
if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
clearBackgroundFolder = folder + "/clearBG"
lsImg = [originFolder +"/" + x for x in os.listdir(originFolder) if x.endswith(".jpg")]
# print(lsImg)
lsClearBG = [x.replace(originFolder,clearBackgroundFolder) for x in lsImg]
# print(lsClearBG)
lsOutputAreas = [x.replace(clearBackgroundFolder + "/",outputFolder +"/areas_") for x in lsClearBG]
# print(lsOutputAreas)
lsOutputResult = [x.replace(clearBackgroundFolder + "/",outputFolder +"/result_") for x in lsClearBG]
# print(lsOutputResult)
lsOutputText = [x.replace(clearBackgroundFolder + "/",outputFolder +"/text_").replace(".jpg",".txt") for x in lsClearBG]
# print(lsOutputText)
count = ''
for i in range(0,lsImg.__len__()):
    # img = cv2.imread("image/cropped/20.jpg")
    # clearBG = cv2.imread("image/clearBG/20.jpg")
    img = cv2.imread(lsImg[i])
    clearBG = cv2.imread(lsClearBG[i])
    H, W, _ = img.shape
    dim = (W, H)
    clearBG = cv2.resize(clearBG, dim, interpolation = cv2.INTER_AREA)
    print(lsImg[i], lsClearBG[i], lsOutputResult[i], lsOutputText[i])
    try:
        face, number_img, name_img, dob_img, class_img, time_img = detect_info(img, clearBG)
    except Exception as ex:
        print('Cant find id card in image - Cant detect area', ex)
        sys.exit()


    list_image = [face, number_img, name_img, dob_img, class_img, time_img]

    for y in range(len(list_image)):
        plt.subplot(len(list_image), 1, y+1)
        plt.imshow(list_image[y])
    plt.savefig(lsOutputAreas[i], bbox_inches='tight')
    # plt.show()
    number_text = reader.get_id_numbers_text(number_img)
    name_text = reader.get_name(name_img)
    dob_text = reader.get_dob_text(dob_img)
    class_text = reader.get_name_text(class_img)
    time_text = reader.get_time(time_img)
    count += number_text + name_text + dob_text + class_text + time_text
    texts = ['Mã học viên:'+number_text,
            'Họ tên: ' + name_text,
            'Ngày sinh: ' + dob_text,
            'Lớp: ' + class_text,
            'Niên khóa:' + time_text]


    plt.figure(figsize=(8, (len(texts) * 1) + 2))
    plt.plot([0, 0], 'r')
    plt.axis([0, 3, -len(texts), 0])
    plt.yticks(-np.arange(len(texts)))
    for t, s in enumerate(texts):
        plt.text(0.1, -t-1, s, fontsize=16)
    plt.savefig(lsOutputResult[i], bbox_inches='tight')
    f = open(lsOutputText[i], "w")
    f.writelines(texts)
    f.close()
print(count.replace(" ","").__len__())