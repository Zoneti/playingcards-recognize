
import numpy as np
import cv2
import os
from contornos import encontre_contornos, realce_da_deteccao
from reconhecimento_padroes import get_cards_from_image

dir_teste_name = './testes/'
imagem_teste_name = '2C.png'

img = cv2.imread(dir_teste_name+imagem_teste_name)
image_redimensionada = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
contornos_encontrados = encontre_contornos(image_redimensionada)
cartas_baralho = get_cards_from_image(image_redimensionada, contornos_encontrados)
realce_da_deteccao(image_redimensionada, cartas_baralho, imagem_teste_name)