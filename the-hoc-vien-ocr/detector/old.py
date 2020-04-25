import cv2
import numpy as np
import statistics
import copy
import pytesseract
from PIL import Image
from util.util import get_contour_boxes, get_img_from_box, get_threshold_img, find_max_box, show_img, draw_rec, plot_img
from util.resize import resize_img_by_height, resize_img_by_width
import subprocess
from matplotlib import pyplot as plt

def cropout_unimportant_part(img):
    h, w, _ = img.shape
    x = get_information_x_axis(img)
    y = get_information_y_axis(img)
    pic = img[int(1.2*y):int(0.9*h), 0:int(0.9 * x)]
    img = img[y:h, x:w]

    return img, pic


def crop_label(img):
    h, w, _ = img.shape
    img = img[0:int(0.9*h), 0:int(0.1 * w)]
    return img


def get_info_list(img, contour_boxes):
    contour_boxes.sort(key=lambda tup: tup[1])
    height, width, _ = img.shape
    list_info = []
    for index, l in enumerate(contour_boxes):
        x, y, w, h = l
        y = y - 20
        if index != len(contour_boxes) - 1:
            x1, y1, _, _ = contour_boxes[index+1]
            list_info.append((x, y, width, y1))
        else:
            list_info.append((x, y, width, height))
    return list_info


def get_main_text(img, box, kernel_height):
    x0, y0, x1, y1 = box
    img = img[y0:y1, x0:x1]
    # plot_img(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    thresh = get_threshold_img(img, kernel)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (thresh.shape[1], kernel_height))
    dilation = cv2.dilate(thresh, kernel, iterations=1)
    # plot_img(dilation)
    contour_boxes = get_contour_boxes(dilation)
    max_box = max(contour_boxes, key=lambda tup: tup[2] * tup[3])
    x, y, w, h = max_box
    return (x0+x, y0+y, x0+x+w, y0+y+h)


def remove_name_label(group, width):
    avg = statistics.mean(map(lambda t: t[-1], group))
    group_orig = copy.deepcopy(group)
    for element in group_orig:
        if element[0] < width/10:
            group.remove(element)
        elif element[-1] < avg and element[0] < width/5:
            group.remove(element)
    return group


def remove_smaller_area(group, width):
    avg = statistics.mean(map(lambda t: t[-1] * t[-2], group))
    group_orig = copy.deepcopy(group)
    for element in group_orig:
        if element[0] < width/10:
            group.remove(element)
        elif element[-1] * element[-2] < avg and element[0] < width/5:
            group.remove(element)
    return group


def get_name(img, box):
    x0, y0, x1, y1 = box
    img = img[y0:y1, x0:x1]
    height, width, _ = img.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    thresh_img = get_threshold_img(img, kernel)
    contour_boxes = get_contour_boxes(thresh_img)
    contour_boxes = remove_smaller_area(contour_boxes, width)
    contour_boxes = remove_name_label(contour_boxes, width)
    contour_boxes.sort(key=lambda t: t[0])
    x, y, w, h = find_max_box(contour_boxes)
    return (x0+x, y0+y, x0+x+w, y0+y+h)


def get_text_from_two_lines(img, box):
    x0, y0, x1, y1 = box
    img = img[y0:y1, x0:x1]
    kernel = np.ones((25, 25), np.uint8)
    thresh = get_threshold_img(img, kernel)
    height, width = thresh.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilation = cv2.dilate(thresh, kernel, iterations=1)
    contour_boxes = get_contour_boxes(dilation)
    avg = statistics.mean(map(lambda t: t[-1]*t[-2], contour_boxes))
    boxes_copy = copy.deepcopy(contour_boxes)
    for box in boxes_copy:
        box_height = box[1] + box[3]
        height_lim = 0.9 * height
        if box[1] > height_lim:
            contour_boxes.remove(box)
        elif box_height == height and box[1] > 0.8 * height:
            contour_boxes.remove(box)
        elif box[-1] * box[-2] < avg/3:
            contour_boxes.remove(box)
    x, y, w, h = find_max_box(contour_boxes)
    if h < 55:
        return (x0+x, y0+y, x0+x+w+5, y0+y+h+5)
    else:
        crop_img = thresh[y:y+h, x:width]
        height, width = crop_img.shape
        hist = cv2.reduce(crop_img, 1, cv2.REDUCE_AVG).reshape(-1)
        hist = uppers = [hist[y] for y in range(height//3, 2*height//3)]
        line = uppers.index(min(uppers)) + height//3
        first_line = (x0+x, y0+y, x0+x+w, y0+y+line)
        second_line = (x0+x, y0+y+line, x0+x+w, y0+y+h)
        return [first_line, second_line]


def get_two_lines_img(img, box):
    x0, y0, x1, y1 = box
    img = img[y0:y1, x0:x1]
    kernel = np.ones((25, 25), np.uint8)
    thresh = get_threshold_img(img, kernel)
    height, width = thresh.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilation = cv2.dilate(thresh, kernel, iterations=1)
    contour_boxes = get_contour_boxes(dilation)
    avg = statistics.mean(map(lambda t: t[-1]*t[-2], contour_boxes))
    boxes_copy = copy.deepcopy(contour_boxes)
    for box in boxes_copy:
        box_height = box[1] + box[3]
        height_lim = 0.9 * height
        if box[1] > height_lim:
            contour_boxes.remove(box)
        elif box_height == height and box[1] > 0.8 * height:
            contour_boxes.remove(box)
        elif box[-1] * box[-2] < avg/3:
            contour_boxes.remove(box)
    x, y, w, h = find_max_box(contour_boxes)
    return (x0+x, y0+y, x0+x+w+5, y0+y+h+5)


def process_result(orig, ratio, result):
    if type(result) is tuple:
        return [get_img_from_box(orig, ratio, result, padding=2)]
    if type(result) is list:
        first_line = get_img_from_box(orig, ratio, result[0], padding=2)
        first_line = cut_blank_part(first_line)
        second_line_img = get_img_from_box(orig, ratio, result[1], padding=2)
        second_line = cut_blank_part(second_line_img)
        return [first_line, second_line]


def get_last_y(result):
    if type(result) is tuple:
        return result[-1]
    if type(result) is list:
        return result[1][-1]


def cut_blank_part(img, padding=5):
    img_h, img_w, _ = img.shape
    kernel = np.ones((25, 25), np.uint8)
    thresh = get_threshold_img(img, kernel)
    contour_boxes = get_contour_boxes(thresh)
    avg = statistics.mean(map(lambda t: t[-1], contour_boxes))
    boxes_copy = copy.deepcopy(contour_boxes)
    for box in boxes_copy:
        if box[-1] < avg/2:
            contour_boxes.remove(box)
        elif box[1] > img_h/2 and box[0] < img_w/10:
            contour_boxes.remove(box)
        elif box[1] < img_h/10 and box[-1] < img_h/5:
            contour_boxes.remove(box)
    x, y, w, h = find_max_box(contour_boxes)
    new_width = x + w + padding
    if new_width > img_w:
        new_width = img_w
    return img[0:img_h, x: new_width]


def get_information_x_axis(img):
    img, ratio = resize_img_by_height(img)
    h, w, _ = img.shape
    img_resize = img[100:400, int(0.25*w):int(0.4*w)]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))
    thresh = get_threshold_img(img_resize, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, h))
    dilation = cv2.dilate(thresh, kernel, iterations=1)
    cnts = get_contour_boxes(dilation)
    cnts_copy = copy.deepcopy(cnts)
    for cnt in cnts_copy:
        if cnt[0] < 0.1*img_resize.shape[1]:
            cnts.remove(cnt)
    max_cnt = max(cnts, key=lambda x: x[-1] * x[-2])
    return int((max_cnt[0]-5+0.25*w)*ratio)


def get_information_y_axis(img):
    img, ratio = resize_img_by_width(img)
    h, w, _ = img.shape
    img_resize = img[0:int(0.4*h), 125:w]
    gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((25, 25), np.uint8)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    thresh = cv2.threshold(
        blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
    dilation = cv2.dilate(thresh, kernel, iterations=1)
    cnts = get_contour_boxes(dilation)
    cnts_copy = copy.deepcopy(cnts)
    for cnt in cnts_copy:
        if cnt[1]+cnt[-1] > 0.95 * img_resize.shape[0]:
            cnts.remove(cnt)
        elif cnt[-2] < 150:
            cnts.remove(cnt)
    max_cnt = max(cnts, key=lambda x: x[1])
    return int((max_cnt[1]-5)*ratio)

def getTwoSmallest(arr):
    min1, min2 = float('inf'), float('inf')
    tmp = arr.copy()
    l = tmp.__len__()
    for i in range(0, l):
        if tmp[i] > 0:
            min1 = i
            break
    tmp = np.flip(tmp)
    for i in range(0, l):
        if tmp[i] > 0:
            min2 = 250 - i
            break
    return min1, min2

def removeBackGround(orig):
    img = orig.copy()
    h, w = img.shape[:2]
    tempH = int(0.8 * h)
    tempW = int(0.3 * w)
    tempBG = img[tempH:h, 0:tempW]
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([tempBG],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()
    plot_img(tempBG)
    histr = []
    maxH = []
    minH = []
    for i in range(0,3):
        histr.append(cv2.calcHist([tempBG],[i],None,[256],[0,256]))
        y, x = getTwoSmallest(histr[i])
        maxH.append(x)
        minH.append(y)
    print(maxH, minH)
    for x in range (0, h):
        for y in range(0, w):
            if minH[0] <= img[x][y][0] and img[x][y][0] <= maxH[0] \
            and minH[1] <= img[x][y][1] and img[x][y][1] <= maxH[1] \
            and minH[2] <= img[x][y][2] and img[x][y][2] <= maxH[2]:
                img[x][y] = [255, 255, 255]
                # print(img[x][y])
    # plot_img(img)
    return img

def detect_info(img):
    img, face = cropout_unimportant_part(img)
    orig = img.copy()
    plot_img(orig)
    clearedBackGround = removeBackGround(orig) # xoa nen
    clearedBackGround, ratio = resize_img_by_height(clearedBackGround)    
    H, W, _= clearedBackGround.shape
    plot_img(clearedBackGround)
    # img, ratio = resize_img_by_height(img)
    label_img = crop_label(clearedBackGround) # Lay phan tieu de de tim vi tri cac vung du lieu
    plot_img(label_img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    threshold_img = get_threshold_img(label_img, kernel)
    plot_img(threshold_img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (label_img.shape[1]//2, 5))
    dilation = cv2.dilate(threshold_img, kernel, iterations=1)
    plot_img(dilation)
    contour_boxes = get_contour_boxes(dilation)
    contour_boxes.sort(key=lambda t: t[2] * t[3], reverse=True)
    contour_boxes = contour_boxes[:4]
    info_list = get_info_list(img, contour_boxes)
    # print(info_list) # (x, y, width, height)
    # get number part
    x, y, _, _ = info_list[0]
    number_box = (int(0.55*W), int(0.85*H), W, H)
    number_box = get_main_text(clearedBackGround, number_box, 5)
    number_img = clearedBackGround[number_box[1]:number_box[3], number_box[0]:number_box[2]]
    # print(number_img)
    # text = pytesseract.image_to_string(number_img, lang='vie', config='--psm 7')
    # print(text.split(':'))
    # number_img = get_img_from_box(orig, ratio, number_box)
    # plot_img(number_img)
    # get name part
    name_box = info_list[0]
    name_box = get_name(clearedBackGround, get_main_text(clearedBackGround, name_box, 5))
    tmpls = list(name_box)
    tmpls[1] = int(tmpls[1] * 0.9) # Keo len 1 chut de tranh mat dau
    name_box = tuple(tmpls)
    # name_img = get_img_from_box(orig, ratio, name_box, padding=2)
    name_img = clearedBackGround[name_box[1]:name_box[3], name_box[0]:name_box[2]]
    name_img = cut_blank_part(name_img)
    # text = pytesseract.image_to_string(name_img, lang='vie', config='--psm 7')
    # print(text)
    # get dob part
    dob_box = info_list[1]
    dob_box = get_main_text(clearedBackGround, dob_box, 5)
    tmpls = list(dob_box)
    tmpls[1] = int(tmpls[1] * 0.9) # Gian no len 1
    tmpls[0] = int(tmpls[2] * 0.3) # Cat phan chu di
    dob_box = tuple(tmpls)
    dob_img = clearedBackGround[dob_box[1]:dob_box[3], dob_box[0]:dob_box[2]]
    # dob_img = get_img_from_box(clearedBackGround, ratio, dob_box)
    # plot_img(dob_img)
    # cv2.imwrite("tempDOB.jpg",dob_img)
    # text = pytesseract.image_to_string(dob_img, lang='vie', config='--psm 7') # can xoa dau :, cac ky tu la
    # print(text)

    # get class part
    class_box = info_list[2]
    class_box = get_main_text(clearedBackGround, class_box, 5)
    tmpls = list(class_box)
    tmpls[1] = int(tmpls[1] * 0.9) # Gian no len 1
    tmpls[0] = int(tmpls[2] * 0.2) # Cat phan chu di
    class_box = tuple(tmpls)
    class_img = clearedBackGround[class_box[1]:class_box[3], class_box[0]:class_box[2]]
    # class_img = get_img_from_box( orig, ratio, gender_and_nationality_box, padding=2)
    # plot_img(class_img)
    # cv2.imwrite("tempClass.jpg",class_img)
    # text = pytesseract.image_to_string(class_img, lang='vie', config='--psm 7') # can xoa dau :, cac ky tu la
    # print(text)

    # get time part
    time_box = info_list[3]
    time_box = get_main_text(clearedBackGround, time_box, 5)

    tmpls = list(time_box)
    tmpls[1] = int(tmpls[1] * 0.9) # Gian no len 1
    tmpls[3] = int(tmpls[3] * 0.9) # Gian no len 1
    tmpls[0] = int(tmpls[2] * 0.2) # Cat phan chu di
    time_box = tuple(tmpls)
    time_img = clearedBackGround[time_box[1]:time_box[3], time_box[0]:time_box[2]]
    # time_img2 = get_img_from_box( clearedBackGround, ratio, time_box, padding=2)
    # plot_img(time_img2)
    # plot_img(time_img)
    # cv2.imwrite("tempTime.jpg",time_img)
    # text = pytesseract.image_to_string(time_img, lang='vie', config='--psm 7') # can xoa dau :, cac ky tu la
    # print(text)
    return face, number_img, name_img, dob_img, class_img, time_img
