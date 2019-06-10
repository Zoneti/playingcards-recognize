
import os
import numpy as np
import cv2
from contornos import preprocess_threshhold

WIDTH = 378
HEIGHT = 534

dir_padroes = 'padroes_imagens'

dir_simbolos = 'simbolos_imagens'

def get_indentificacao_cartas(img, contours):
    h = np.float32([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])
    cards = []
    for (approx, contour) in contours:
        # transform image to rectangle
        transform = cv2.getPerspectiveTransform(np.float32(approx), h)
        warp = cv2.warpPerspective(img, transform, (WIDTH, HEIGHT))
        thresh = preprocess_threshhold(warp)
        # rotate when transformation above rotated it
        thresh = rotate_image_properly(thresh)

        # add whtite overlay in left corner in order to get rid  of border in J, Q, K
        y_off = 45
        x_off = 55
        white = (np.ones((100, 100)) * 255)
        thresh[y_off:y_off+white.shape[0], x_off:x_off+white.shape[1]] = white

        best_fit = predict_card(thresh)

        cards.append((contour, best_fit))

    return cards


def predict_card(thresh):
    value_patterns = load_patterns(dir_padroes)
    value = predict_value(thresh, value_patterns)
    symbol_patterns = load_patterns(dir_simbolos)
    symbol = predict_symbol(thresh, symbol_patterns, value)

    return '{} - {}'.format(value, symbol)


def load_patterns(dirname):
    value_patterns = {}
    # load patterns from pattern directory
    patterns = os.listdir(dirname)
    for pattern in patterns:
        pattern_img = cv2.imread(
            './'+dirname+'/'+pattern, cv2.IMREAD_GRAYSCALE)
        thresh = cv2.adaptiveThreshold(pattern_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 15, 2)

        pattern_name = pattern.split('.')[0]
        if (dirname == dir_padroes):
            
            pattern_name = pattern_name.upper()
            if pattern_name == '0':
                pattern_name = '10'

            # store that value in dictionary
            value_patterns[pattern_name] = thresh

        if (dirname == dir_simbolos):
            value_patterns[pattern_name] = pattern_img

    return value_patterns


def predict_symbol(thresh, symbol_patterns, card_value):

    # depends of card, Region Of Interests is in another place - we look for biggest symbol on card
    if card_value in ['K', 'Q', 'J']:
        roi = thresh[(HEIGHT-150):(HEIGHT-50), (WIDTH-150):(WIDTH-70)]
        # flip image - it's upside down
        roi = cv2.flip(roi, -1)
    elif card_value == 'A':
        roi = thresh[int(HEIGHT/2)-150:int(HEIGHT/2)+150,
                     int(WIDTH/2)-125:int(WIDTH/2)+125]
    elif card_value in ['2', '3']:
        roi = thresh[45:165, 145:245]
    else:
        roi = thresh[45:165, (WIDTH-155):(WIDTH-55)]
        

        # remove noise, add some blur
    roi = cv2.medianBlur(roi, 7)
    # black on white => white on black
    roi = (255 - roi)

    # find contours
    contours, _ = cv2.findContours(
        roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # filter and sort contours
    a = 0.05 * 85*70
    choosen_contours = sorted(
        contours, key=lambda c: cv2.contourArea(c), reverse=True)
    choosen_contours = list(
        filter(lambda c: cv2.contourArea(c) > a, choosen_contours))

    # in case, when no proper contour found
    if len(choosen_contours) < 1:
        return 'undefined'

    # fill found contour
    cv2.drawContours(roi, [choosen_contours[0]], -
                     1, (255, 255, 255), cv2.FILLED)

    # get bounding rect
    x, y, w, h = cv2.boundingRect(choosen_contours[0])
    # get what's inside bounding rect
    sym = roi[y:y+h, x:x+w]
    # resize it to pattern size
    sym = cv2.resize(sym, (140, 180))

    # choose symbol, that fits best
    symbol_fit = {}
    for (key, value) in symbol_patterns.items():
        symbol_fit[key] = cv2.countNonZero(cv2.absdiff(sym, value))

    # flip sym
    sym = cv2.flip(sym, -1)

    if card_value == 'A':
        for (key, value) in symbol_patterns.items():
            old_value = symbol_fit[key]
            new_value = cv2.countNonZero(cv2.absdiff(sym, value))
            if new_value < old_value:
                symbol_fit[key] = new_value

    best_fit = min(symbol_fit, key=symbol_fit.get)
    print(card_value, best_fit, symbol_fit)
    #cv2.imshow('image', sym)
    cv2.waitKey(0)

    return best_fit


def predict_value(thresh, value_patterns):
    # choose left upper corner in order to get cards value
    left = thresh[15:100, 0:70]
    # remove noise and add some blur
    left = cv2.medianBlur(left, 7)
    # black on white => white on black
    left = (255-left)
    # get contour
    contours, _ = cv2.findContours(
        left, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # filter contours, which area is bigger than 2% of whole  area
    a = 0.02 * 85*70
    choosen_contours = list(filter(lambda c: cv2.contourArea(c) > a, contours))
    choosen_contours = sorted(
        contours, key=lambda c: cv2.contourArea(c), reverse=True)
    # when no proper contours find
    if len(choosen_contours) < 1:
        return "undefined"
    # get bounding rect of biggest contour, which is probably our value
    x, y, w, h = cv2.boundingRect(choosen_contours[0])
    # get what's inside bounding rect
    num = left[y:y+h, x:x+w]
    # resize it to pattern size
    num = cv2.resize(num, (30, 60))

    # check how "similar" our num is to pattern
    value_fit = {}
    for (key, value) in value_patterns.items():
        # calculate different pixels and store that difference
        value_fit[key] = cv2.countNonZero(cv2.absdiff(num, value))

    # choose best fit = the smallest difference
    best_fit = min(value_fit, key=value_fit.get)

    # debugging => uncomment to take a look what was found
    #print(best_fit, value_fit)
    #cv2.imshow('image', num)
    cv2.waitKey(0)

    # return value we predicted
    return best_fit


def rotate_image_properly(thresh):

    # check in which corner is the symbol
    AREA1 = 140 * 40
    AREA2 = 65 * 90
    # left
    left = thresh[15:155, 0:40]
    # right - card flipped
    right = thresh[15:155, 327:367]
    # left-upper, card id detected horizontal
    upper_left = thresh[5:70, 15:105]
    # right-upper, card id detected horizontal
    upper_right = thresh[10:75, 280:370]

    # find card position according to percentage of black pixels in cut out image
    edges = [cv2.countNonZero(left)/AREA1, cv2.countNonZero(
        right)/AREA1, cv2.countNonZero(upper_left)/AREA2, cv2.countNonZero(upper_right)/AREA2]
    index = edges.index(min(edges))

    # if iamge is rotated, rotate it to desirez orientation
    if index == 1:
        thresh = cv2.flip(thresh, +1)
    elif index == 2:
        thresh = cv2.transpose(thresh)
        thresh = cv2.resize(thresh, (0, 0), fx=WIDTH /
                            HEIGHT, fy=HEIGHT/WIDTH)
        thresh = cv2.flip(thresh, -1)
    elif index == 3:
        thresh = cv2.transpose(thresh)
        thresh = cv2.resize(thresh, (0, 0), fx=WIDTH /
                            HEIGHT, fy=HEIGHT/WIDTH)
        thresh = cv2.flip(thresh, 1)

    return thresh