
import os
import numpy as np
import cv2
from contornos import preprocessar_threshhold

LARGURA = 378
ALTURA = 534

dir_padroes = './padroes_imagens/'

dir_simbolos = './simbolos_imagens/'

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