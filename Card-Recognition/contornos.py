import numpy as np
import cv2

CHOOSEM_CONTOURS_NUM = 400
CARDS_ALPHA = 0.2

dir_imagens_reconhecidas = './imagens_processadas/'

#
def preprocessamento_ruido_fundo_com_treshold(imagem, tamanho_treshold):
    img_cor_modelo_hsv = cv2.cvtColor(imagem, cv2.COLOR_RGB2HSV)
    _, saturacao, _ = cv2.split(img_cor_modelo_hsv)
    imagem_com_laplace = cv2.Laplacian(
        saturacao, cv2.CV_8U, saturacao, ksize=3)
    imagem_estruturada = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    imagem_cinza = cv2.dilate(imagem_com_laplace, imagem_estruturada, iterations=1)
    imagem_cinza = cv2.blur(imagem_cinza, (5, 5))
    imagem_cinza = cv2.adaptiveThreshold(imagem_cinza, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, tamanho_treshold, 2)
    return imagem_cinza


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

#
def preprocessar_threshhold(imagem):
    imagem_tons_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    imagem_tons_cinza = cv2.blur(imagem_tons_cinza, (5, 5))
    threshold = cv2.adaptiveThreshold(imagem_tons_cinza, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 19, 2)
    img_estrutura = np.ones((3, 3))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, img_estrutura)

    return threshold


#
def encontre_contornos_consolidado(img):
    bordas = []

    height, width = img.shape[:2]
    imshape = (height, width)

    bordas.append(encontrar_bordas(
        imshape, preprocessamento_ruido_fundo_com_treshold(img, 15)))
    bordas.append(encontrar_bordas(
        imshape, preprocessamento_ruido_fundo_com_treshold(img, 11)))
    bordas.append(encontrar_bordas(imshape, preprocess_threshhold_hsv1(img)))
    bordas.append(encontrar_bordas(imshape, preprocess_threshhold_hsv2(img)))
    bordas.append(encontrar_bordas(imshape, preprocessar_threshhold(img)))

    bordas_filtradas = remover_bordas_duplicadas(bordas, imshape)

    return bordas_filtradas

#
def remover_bordas_duplicadas(bordas, forma_imagem):
    tamanho_imagem = forma_imagem[0] * forma_imagem[1]

    bordas_niveladas = []
    for contours_group in bordas:
        for bordas in contours_group:
            bordas_niveladas.append(bordas)

    bordas_filtradas = []
    mapa_bordas = np.ones(forma_imagem)
    for c in bordas_niveladas:
        _, bordas = c
        mapa_duplicado = np.copy(mapa_bordas)
        cv2.drawContours(mapa_duplicado, [bordas], -1,
                         (0, 255, 0), thickness=cv2.FILLED)

        if abs(np.count_nonzero(mapa_duplicado)-np.count_nonzero(mapa_bordas)) < 0.01 * tamanho_imagem:
            continue

        mapa_bordas = mapa_duplicado

        bordas_filtradas.append(c)

    return bordas_filtradas


#
def encontrar_bordas(img_forma, func_threshold):
    h, w = img_forma

    threshold = func_threshold

    bordas_detectadas = cv2.Canny(threshold, 50, 250)
    bordas, _ = cv2.findContours(
        bordas_detectadas, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bordas_escolhidas = sorted(
        bordas, key=lambda contour: cv2.contourArea(contour), reverse=True)
    
    menor_area = cv2.contourArea(bordas_escolhidas[0])/4
    bordas_escolhidas = list(
        filter(lambda c: cv2.contourArea(c) > menor_area, bordas_escolhidas))

    mapa_bordas = np.ones((h, w))
    borda_cartas = []
    for c in bordas_escolhidas:

        perimetro = 0.015*cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, perimetro, True)
        
        if len(approx) != 4:
            continue
        
        mapa_duplicado = np.copy(mapa_bordas)
        cv2.drawContours(mapa_duplicado, [c], -1,
                         (0, 255, 0), thickness=cv2.FILLED)
        
        if np.array_equal(mapa_duplicado, mapa_bordas):
            continue
        
        mapa_bordas = mapa_duplicado

        borda_cartas.append((approx, c))

    return borda_cartas


def realce_da_deteccao(img, cards, image):
    altura, largura = img.shape[:2]
    tamanho_letra = max(0.5, int(min(largura, altura)*0.00075))

    for (borda, correspondencia) in cards:
        overlay = img.copy()
        cv2.drawContours(overlay, [borda], -1, (255, 255, 0), -1)
        cv2.addWeighted(overlay, CARDS_ALPHA, img, 1 - CARDS_ALPHA, 0, img)
        cv2.drawContours(img, [borda], -1, (255, 255, 0), 2)
        M = cv2.moments(borda)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.putText(img, correspondencia, (cX-(len(correspondencia)*int(tamanho_letra*10)), cY),
                    cv2.FONT_HERSHEY_SIMPLEX, tamanho_letra, (255, 0, 0), int(tamanho_letra*5))

    cv2.imwrite(dir_imagens_reconhecidas+image, img)

