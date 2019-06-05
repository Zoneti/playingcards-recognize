
import numpy as np
import cv2
import os


def load_image(dirname):
    images = os.listdir(dirname)
    for image in images:
        img = cv2.imread(dirname+image)
        resized_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        contours = find_contour_wrapper(resized_img)
        
        cards = get_cards_from_image(resized_img, contours)
        highlight_detected(resized_img, cards, image)


load_image('./testes/')