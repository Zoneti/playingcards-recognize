
import os
import numpy as np
import cv2

LARGURA = 378
ALTURA = 534

dir_padroes = './padroes_imagens/'
dir_imagens_reconhecidas = './imagens_processadas/'
dir_simbolos = './simbolos_imagens/'

NUMERO_BORDAS = 400
ALPHA_CARTAS = 0.2



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


def preprocessar_threshhold_hsv1(imagem):
    escala_cor_hsv = cv2.cvtColor(imagem, cv2.COLOR_RGB2HSV)
    _, saturacao, _ = cv2.split(escala_cor_hsv)
    threshold = cv2.adaptiveThreshold(saturacao, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 15, 2)
    imagem_estruturada = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, imagem_estruturada)

    return threshold


def preprocessar_threshhold_hsv2(imagem):
    escala_cor_hsv = cv2.cvtColor(imagem, cv2.COLOR_RGB2HSV)
    _, saturação, _ = cv2.split(escala_cor_hsv)
    imagem_escala_cinza = cv2.blur(saturação, (5, 5))
    threshold = cv2.adaptiveThreshold(imagem_escala_cinza, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 19, 2)
    imagem_estruturada = np.ones((3, 3))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, imagem_estruturada)

    return threshold

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
    bordas.append(encontrar_bordas(imshape, preprocessar_threshhold_hsv1(img)))
    bordas.append(encontrar_bordas(imshape, preprocessar_threshhold_hsv2(img)))
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
        cv2.addWeighted(overlay, ALPHA_CARTAS, img, 1 - ALPHA_CARTAS, 0, img)
        cv2.drawContours(img, [borda], -1, (255, 255, 0), 2)
        M = cv2.moments(borda)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.putText(img, correspondencia, (cX-(len(correspondencia)*int(tamanho_letra*10)), cY),
                    cv2.FONT_HERSHEY_SIMPLEX, tamanho_letra, (255, 0, 0), int(tamanho_letra*5))

    cv2.imwrite(dir_imagens_reconhecidas+image, img)



#
def get_indentificacao_cartas(imagem, bordas):
    h = np.float32([[0, 0], [LARGURA, 0], [LARGURA, ALTURA], [0, ALTURA]])
    cartas = []
    for (approx, borda) in bordas:
        
        transformada_perspectiva = cv2.getPerspectiveTransform(np.float32(approx), h)
        perspectiva_consolidada = cv2.warpPerspective(imagem, transformada_perspectiva, (LARGURA, ALTURA))
        threshold = preprocessar_threshhold(perspectiva_consolidada)
        
        threshold = rotacionar_imagem(threshold)

        y_off = 45
        x_off = 55
        white = (np.ones((100, 100)) * 255)
        threshold[y_off:y_off+white.shape[0], x_off:x_off+white.shape[1]] = white

        best_fit = classificar_cartas(threshold)

        cartas.append((borda, best_fit))

    return cartas

#
def classificar_cartas(thresh):
    numero_padroes = carregar_padroes_numero(dir_padroes)
    numero = classificar_valor(thresh, numero_padroes)
    naipe_padroes = carregar_padroes_naipe(dir_simbolos)
    naipe = classificar_naipe(thresh, naipe_padroes, numero)

    return '{} - {}'.format(numero, naipe)

#
def carregar_padroes_numero(dirname):
    padroes_numero = {}
    padroes = os.listdir(dirname)
    for padrao in padroes:
        imagem_padrao = cv2.imread(dirname+padrao, cv2.IMREAD_GRAYSCALE)
        threshold = cv2.adaptiveThreshold(imagem_padrao, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 15, 2)

        nome_padrao = padrao.split('.')[0]
        nome_padrao = nome_padrao.upper()
        if nome_padrao == '0':
           nome_padrao = '10'
        padroes_numero[nome_padrao] = threshold

    return padroes_numero

#
def carregar_padroes_naipe(dirname):
    padrao_naipe = {}
    padroes = os.listdir(dirname)
    for padrao in padroes:
        imagem_padrao = cv2.imread(dirname+padrao, cv2.IMREAD_GRAYSCALE)
        threshold = cv2.adaptiveThreshold(imagem_padrao, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 15, 2)
        nome_padrao = padrao.split('.')[0]
        nome_padrao = nome_padrao.upper()
        padrao_naipe[nome_padrao] = imagem_padrao

    return padrao_naipe


def classificar_naipe(threshold, padrao_naipe, valor_numero):

    if valor_numero in ['K', 'Q', 'J']:
        roi = threshold[(ALTURA-150):(ALTURA-50), (LARGURA-150):(LARGURA-70)]
        roi = cv2.flip(roi, -1)
    elif valor_numero == 'A':
        roi = threshold[int(ALTURA/2)-150:int(ALTURA/2)+150,
                     int(LARGURA/2)-125:int(LARGURA/2)+125]
    elif valor_numero in ['2', '3']:
        roi = threshold[45:165, 145:245]
    else:
        roi = threshold[45:165, (LARGURA-155):(LARGURA-55)]
        
    roi = cv2.medianBlur(roi, 7)
    roi = (255 - roi)

    bordas, _ = cv2.findContours(
        roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    a = 0.05 * 85*70
    bordas_selecionadas = sorted(
        bordas, key=lambda c: cv2.contourArea(c), reverse=True)
    bordas_selecionadas = list(
        filter(lambda c: cv2.contourArea(c) > a, bordas_selecionadas))

    if len(bordas_selecionadas) < 1:
        return 'undefined'

    cv2.drawContours(roi, [bordas_selecionadas[0]], -
                     1, (255, 255, 255), cv2.FILLED)

    x, y, w, h = cv2.boundingRect(bordas_selecionadas[0])
    sym = roi[y:y+h, x:x+w]
    sym = cv2.resize(sym, (140, 180))

    naipe_parecido = {}
    for (chave, valor) in padrao_naipe.items():
        naipe_parecido[chave] = cv2.countNonZero(cv2.absdiff(sym, valor))

    sym = cv2.flip(sym, -1)

    if valor_numero == 'A':
        for (chave, valor) in padrao_naipe.items():
            valor_antigo = naipe_parecido[chave]
            novo_valor = cv2.countNonZero(cv2.absdiff(sym, valor))
            if novo_valor < valor_antigo:
                naipe_parecido[chave] = novo_valor

    melhor_comparacao = min(naipe_parecido, key=naipe_parecido.get)
    print(valor_numero, melhor_comparacao, naipe_parecido)
    cv2.waitKey(0)

    return melhor_comparacao


def classificar_valor(thresh, value_patterns):  
    esquerda = thresh[15:100, 0:70]
    esquerda = cv2.medianBlur(esquerda, 7)  
    esquerda = (255-esquerda)

    bordas, _ = cv2.findContours(
        esquerda, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    a = 0.02 * 85*70
    bordas_escolhidas = list(filter(lambda c: cv2.contourArea(c) > a, bordas))
    bordas_escolhidas = sorted(
        bordas, key=lambda c: cv2.contourArea(c), reverse=True)
    if len(bordas_escolhidas) < 1:
        return "undefined"
    x, y, w, h = cv2.boundingRect(bordas_escolhidas[0])
    num = esquerda[y:y+h, x:x+w]
    num = cv2.resize(num, (30, 60))

    value_fit = {}
    for (key, value) in value_patterns.items():
        value_fit[key] = cv2.countNonZero(cv2.absdiff(num, value))

    best_fit = min(value_fit, key=value_fit.get)
    cv2.waitKey(0)

    return best_fit


def rotacionar_imagem(threshold):

    AREA1 = 140 * 40
    AREA2 = 65 * 90
    esquerda = threshold[15:155, 0:40]
    direita = threshold[15:155, 327:367]
    esquerda_superior = threshold[5:70, 15:105]
    direita_superior = threshold[10:75, 280:370]

    bordas = [cv2.countNonZero(esquerda)/AREA1, cv2.countNonZero(
        direita)/AREA1, cv2.countNonZero(esquerda_superior)/AREA2, cv2.countNonZero(direita_superior)/AREA2]
    indice = bordas.index(min(bordas))

    if indice == 1:
        threshold = cv2.flip(threshold, +1)
    elif indice == 2:
        threshold = cv2.transpose(threshold)
        threshold = cv2.resize(threshold, (0, 0), fx=LARGURA /
                            ALTURA, fy=ALTURA/LARGURA)
        threshold = cv2.flip(threshold, -1)
    elif indice == 3:
        threshold = cv2.transpose(threshold)
        threshold = cv2.resize(threshold, (0, 0), fx=LARGURA /
                            ALTURA, fy=ALTURA/LARGURA)
        threshold = cv2.flip(threshold, 1)

    return threshold