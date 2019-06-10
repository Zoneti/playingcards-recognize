
import numpy as np
import cv2
import os
#from contornos import encontre_contornos_consolidado, realce_da_deteccao
from reconhecimento_padroes import get_indentificacao_cartas, encontre_contornos_consolidado, realce_da_deteccao

dir_teste_name = './testes/'
imagem_teste_name = '9H.png'

img = cv2.imread(dir_teste_name+imagem_teste_name)
image_redimensionada = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
contornos_encontrados = encontre_contornos_consolidado(image_redimensionada)
cartas_baralho = get_indentificacao_cartas(image_redimensionada, contornos_encontrados)
realce_da_deteccao(image_redimensionada, cartas_baralho, imagem_teste_name)