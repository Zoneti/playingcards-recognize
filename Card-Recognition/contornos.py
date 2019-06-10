import numpy as np
import cv2

CHOOSEM_CONTOURS_NUM = 400
CARDS_ALPHA = 0.2

dir_imagens_reconhecidas = './imagens_processadas/'


def preprocess_threshhold_background_noise(img, thresh_size):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    _, saturation, _ = cv2.split(hsv)
    lap = cv2.Laplacian(
        saturation, cv2.CV_8U, saturation, ksize=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grayscale_img = cv2.dilate(lap, kernel, iterations=1)
    grayscale_img = cv2.blur(grayscale_img, (5, 5))
    grayscale_img = cv2.adaptiveThreshold(grayscale_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, thresh_size, 2)
    return grayscale_img


def preprocess_threshhold_hsv1(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    _, saturation, _ = cv2.split(hsv)
    thresh = cv2.adaptiveThreshold(saturation, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 15, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return thresh


def preprocess_threshhold_hsv2(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    _, saturation, _ = cv2.split(hsv)
    grayscale_img = cv2.blur(saturation, (5, 5))
    thresh = cv2.adaptiveThreshold(grayscale_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 19, 2)
    kernel = np.ones((3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return thresh


def preprocess_threshhold(img):
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale_img = cv2.blur(grayscale_img, (5, 5))
    thresh = cv2.adaptiveThreshold(grayscale_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 19, 2)
    kernel = np.ones((3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return thresh


def encontre_contornos(img):
    contours = []

    height, width = img.shape[:2]
    imshape = (height, width)

    contours.append(find_contours(
        imshape, preprocess_threshhold_background_noise(img, 15)))
    contours.append(find_contours(
        imshape, preprocess_threshhold_background_noise(img, 11)))
    contours.append(find_contours(imshape, preprocess_threshhold_hsv1(img)))
    contours.append(find_contours(imshape, preprocess_threshhold_hsv2(img)))
    contours.append(find_contours(imshape, preprocess_threshhold(img)))

    filtered_contours = remove_duplicate_contours(contours, imshape)

    return filtered_contours


def remove_duplicate_contours(cnts, imshape):
    imsize = imshape[0] * imshape[1]

    contours_flatten = []
    for contours_group in cnts:
        for contour in contours_group:
            contours_flatten.append(contour)

    filtered_contours = []
    contours_map = np.ones(imshape)
    for c in contours_flatten:
        _, contour = c
        copied_map = np.copy(contours_map)
        cv2.drawContours(copied_map, [contour], -1,
                         (0, 255, 0), thickness=cv2.FILLED)

        # if difference smaller than 1% of image size => skip contour
        # it's very probably that we get contour for the same card, but different method returned slightly different contour
        if abs(np.count_nonzero(copied_map)-np.count_nonzero(contours_map)) < 0.01 * imsize:
            continue

        contours_map = copied_map

        filtered_contours.append(c)

    return filtered_contours


def find_contours(imshape, thresholding_function):
    height, width = imshape

    thresh = thresholding_function

    # detect contours
    contours_detected = cv2.Canny(thresh, 50, 250)
    contours, _ = cv2.findContours(
        contours_detected, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours by area
    choosen_contours = sorted(
        contours, key=lambda contour: cv2.contourArea(contour), reverse=True)
    # set smallest contour's area as 1/4 of biggest one's area
    smallest_area = cv2.contourArea(choosen_contours[0])/4
    choosen_contours = list(
        filter(lambda c: cv2.contourArea(c) > smallest_area, choosen_contours))

    contours_map = np.ones((height, width))
    card_contours = []
    for c in choosen_contours:

        perimeter = 0.015*cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, perimeter, True)
        # skip when it's not an rectangle
        if len(approx) != 4:
            continue
        # check if contour isn't nested in another one
        copied_map = np.copy(contours_map)
        cv2.drawContours(copied_map, [c], -1,
                         (0, 255, 0), thickness=cv2.FILLED)
        # if so, just skip it
        if np.array_equal(copied_map, contours_map):
            continue
        # if no, add to "map" of already added contours
        contours_map = copied_map

        card_contours.append((approx, c))

    return card_contours


def realce_da_deteccao(img, cards, image):
    height, width = img.shape[:2]
    fontsize = max(0.5, int(min(width, height)*0.00075))

    for (contour, match) in cards:
        # create transparent background
        overlay = img.copy()
        cv2.drawContours(overlay, [contour], -1, (255, 255, 0), -1)
        cv2.addWeighted(overlay, CARDS_ALPHA, img, 1 - CARDS_ALPHA, 0, img)
        # add notmal contour
        cv2.drawContours(img, [contour], -1, (255, 255, 0), 2)
        # calculate center if contour => get position for text
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.putText(img, match, (cX-(len(match)*int(fontsize*10)), cY),
                    cv2.FONT_HERSHEY_SIMPLEX, fontsize, (255, 0, 0), int(fontsize*5))

    cv2.imwrite(dir_imagens_reconhecidas+image, img)

