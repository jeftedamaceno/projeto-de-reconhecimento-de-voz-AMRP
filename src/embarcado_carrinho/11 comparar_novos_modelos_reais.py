import cv2
import numpy as np
import time
import os
import csv
import matplotlib.pyplot as plt
from collections import deque

VIDEO_ENTRADA = "pontilhado2.mp4"
VIDEO_SAIDA = "resultado_comparacao_grafo2.mp4"
CSV_SAIDA = "angulos_telemetria.csv"
LARGURA_PROC, ALTURA_PROC = 128, 128

PULAR_FRAMES = 2

AREA_MINIMA = 18
AREA_MAXIMA = 350
LARGURA_MINIMA = 4
ALTURA_MINIMA = 4
LARGURA_MAXIMA = 45
ALTURA_MAXIMA = 45

DISTANCIA_MAXIMA_CONEXAO = 40.0

historico_m2 = deque(maxlen=5)
historico_m3 = deque(maxlen=5)

CLASSES_CNN = [
    "Reta", "Curva_Aberta_Esq", "Curva_Aberta_Dir",
    "Curva_Fechada_Esq", "Curva_Fechada_Dir",
    "Bifurcacao_Esq_Dir", "Cruzamento_3Pistas",
    "Cruzamento_4Pistas", "Sem_Caminho"
]

model_cnn = None
model_hibrido = None

try:
    import tensorflow as tf
    if os.path.exists("modelo_sintetico_comparacao.h5"):
        model_cnn = tf.keras.models.load_model("modelo_sintetico_comparacao.h5")
    if os.path.exists("modelo_hibrido_segmentador.h5"):
        model_hibrido = tf.keras.models.load_model("modelo_hibrido_segmentador.h5", compile=False)
except Exception:
    pass

def construir_grafo_caminho(binaria_img):
    kernel_erosao = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bin_erodida = cv2.erode(binaria_img, kernel_erosao, iterations=1)

    mascara_roi = np.zeros_like(bin_erodida)
    pts_roi = np.array([
        [int(LARGURA_PROC * 0.15), int(ALTURA_PROC * 0.30)],
        [int(LARGURA_PROC * 0.85), int(ALTURA_PROC * 0.30)],
        [LARGURA_PROC, ALTURA_PROC],
        [0, ALTURA_PROC]
    ], dtype=np.int32)
    cv2.fillPoly(mascara_roi, [pts_roi], 255)

    bin_roi = cv2.bitwise_and(bin_erodida, bin_erodida, mask=mascara_roi)

    num_labels, labels, stats, centroides = cv2.connectedComponentsWithStats(bin_roi, connectivity=8)
    
    pontos = []
    for i in range(1, num_labels):
        w, h, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
        
        if (AREA_MINIMA <= area <= AREA_MAXIMA and 
            LARGURA_MINIMA <= w <= LARGURA_MAXIMA and 
            ALTURA_MINIMA <= h <= ALTURA_MAXIMA and 
            aspect_ratio < 3.5):
            pontos.append((int(centroides[i][0]), int(centroides[i][1])))

    if len(pontos) == 0:
        return "Sem_Caminho", 0.0, bin_roi, None, [], []

    pontos = sorted(pontos, key=lambda p: p[1], reverse=True)
    pt_base = pontos[0]
    num_pontos = len(pontos)

    arestas_passo1 = []
    grau = [0] * num_pontos

    for i in range(num_pontos):
        distancias = []
        for j in range(num_pontos):
            if i != j:
                d = np.hypot(pontos[i][0] - pontos[j][0], pontos[i][1] - pontos[j][1])
                if d <= DISTANCIA_MAXIMA_CONEXAO:
                    distancias.append((d, j))
        
        distancias.sort(key=lambda x: x[0])
        for d, j in distancias[:2]:
            par = tuple(sorted((i, j)))
            if par not in arestas_passo1:
                arestas_passo1.append(par)
                grau[i] += 1
                grau[j] += 1

    triangulos = 0
    for i in range(num_pontos):
        for j in range(i + 1, num_pontos):
            for k in range(j + 1, num_pontos):
                if (tuple(sorted((i, j))) in arestas_passo1 and 
                    tuple(sorted((j, k))) in arestas_passo1 and 
                    tuple(sorted((i, k))) in arestas_passo1):
                    triangulos += 1

    eh_encruzilhada = (triangulos >= 1) or any(g >= 3 for g in grau)

    arestas_finais = []
    if eh_encruzilhada:
        for i in range(num_pontos):
            distancias = []
            for j in range(num_pontos):
                if i != j:
                    d = np.hypot(pontos[i][0] - pontos[j][0], pontos[i][1] - pontos[j][1])
                    if d <= DISTANCIA_MAXIMA_CONEXAO:
                        distancias.append((d, j))
            distancias.sort(key=lambda x: x[0])
            for d, j in distancias[:4]:
                par = tuple(sorted((i, j)))
                if par not in arestas_finais:
                    arestas_finais.append(par)
    else:
        arestas_finais = arestas_passo1

    angulos = []
    for p in pontos[1:]:
        dx = p[0] - pt_base[0]
        dy = pt_base[1] - p[1]
        if dy > 0:
            ang = np.degrees(np.arctan2(dx, dy))
            angulos.append(ang)

    if eh_encruzilhada:
        if triangulos >= 2 or sum(1 for g in grau if g >= 3) >= 2:
            classe = "Cruzamento_4Pistas"
        else:
            classe = "Cruzamento_3Pistas" if any(abs(a) <= 15 for a in angulos) else "Bifurcacao_Esq_Dir"
    elif len(pontos) >= 2:
        ang = angulos[0] if len(angulos) > 0 else 0.0
        if -12 <= ang <= 12: classe = "Reta"
        elif -40 <= ang < -12: classe = "Curva_Aberta_Esq"
        elif 12 < ang <= 40: classe = "Curva_Aberta_Dir"
        elif ang < -40: classe = "Curva_Fechada_Esq"
        else: classe = "Curva_Fechada_Dir"
    else:
        classe = "Sem_Caminho"

    angulo_principal = angulos[0] if len(angulos) > 0 else 0.0
    return classe, angulo_principal, bin_roi, pt_base, pontos, arestas_finais

def binarizar_blackhat(frame_bgr):
    cinza = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    desfocado = cv2.GaussianBlur(cinza, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    blackhat = cv2.morphologyEx(desfocado, cv2.MORPH_BLACKHAT, kernel)
    _, binaria = cv2.threshold(blackhat, 25, 255, cv2.THRESH_BINARY)
    return binaria

def executar_metodo_1(binaria_img):
    if model_cnn is None: return "N/A"
    binaria_3c = cv2.cvtColor(binaria_img, cv2.COLOR_GRAY2BGR)
    inp = np.expand_dims(cv2.cvtColor(binaria_3c, cv2.COLOR_BGR2RGB).astype(np.float32), axis=0)
    pred = model_cnn.predict(inp, verbose=0)
    return CLASSES_CNN[np.argmax(pred[0])]

def executar_metodo_2(frame_bgr):
    binaria_fita = binarizar_blackhat(frame_bgr)
    c, a, mask, p, pts, edges = construir_grafo_caminho(binaria_fita)
    historico_m2.append(a)
    return c, float(np.mean(historico_m2)) if len(historico_m2) > 0 else 0.0, mask, p, pts, edges

# def executar_metodo_3(frame_rgb):
#     if model_hibrido is None:
#         return "N/A", 0.0, np.zeros((128, 128), dtype=np.uint8), None, [], []

#     inp = np.expand_dims(frame_rgb.astype(np.float32) / 255.0, axis=0)
#     pred_mask = model_hibrido.predict(inp, verbose=0)[0, :, :, 0]
#     binaria_ia = ((pred_mask > 0.35).astype(np.uint8)) * 255
    
#     c, a, mask, p, pts, edges = construir_grafo_caminho(binaria_ia)
#     historico_m3.append(a)
#     return c, float(np.mean(historico_m3)) if len(historico_m3) > 0 else 0.0, mask, p, pts, edges
def executar_metodo_3(frame_rgb):
    if model_hibrido is None:
        return executar_metodo_2(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

  
    inp = np.expand_dims(frame_rgb.astype(np.float32) / 255.0, axis=0)
    pred_mask = model_hibrido.predict(inp, verbose=0)[0, :, :, 0]
    

    mask_gray = (pred_mask * 255.0).astype(np.uint8)


    if np.max(mask_gray) > 40:
        _, binaria_ia = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:

        _, binaria_ia = cv2.threshold(mask_gray, 20, 255, cv2.THRESH_BINARY)


    kernel_suave = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binaria_ia = cv2.dilate(binaria_ia, kernel_suave, iterations=1)

    
    if cv2.countNonZero(binaria_ia) < 12:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        binaria_ia = binarizar_blackhat(frame_bgr)

    c, a, mask, p, pts, edges = construir_grafo_caminho(binaria_ia)
    
    historico_m3.append(a)
    ang_suave = float(np.mean(historico_m3)) if len(historico_m3) > 0 else 0.0
    
    return c, ang_suave, mask, p, pts, edges


cap = cv2.VideoCapture(VIDEO_ENTRADA)
if not cap.isOpened():
    raise FileNotFoundError(f"Vídeo {VIDEO_ENTRADA} não encontrado.")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(VIDEO_SAIDA, fourcc, fps, (768, 256), True)

if not out.isOpened():
    VIDEO_SAIDA = "resultado_comparacao_grafo2.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(VIDEO_SAIDA, fourcc, fps, (768, 256), True)

arquivo_csv = open(CSV_SAIDA, mode='w', newline='')
escritor_csv = csv.writer(arquivo_csv)
escritor_csv.writerow(["frame", "tempo_ms", "classe_m1", "classe_m2", "angulo_m2", "classe_m3", "angulo_m3"])

tempos_m1, tempos_m2, tempos_m3 = [], [], []
angulos_m2, angulos_m3 = [], []

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    frame_id += 1
    if frame_id % PULAR_FRAMES != 0: continue

    frame_red = cv2.resize(frame, (LARGURA_PROC, ALTURA_PROC))
    frame_rgb = cv2.cvtColor(frame_red, cv2.COLOR_BGR2RGB)

    t0 = time.time()
    c_m2, a_m2, bin_m2, pt_base_m2, pts_m2, edges_m2 = executar_metodo_2(frame_red)
    t_m2 = (time.time() - t0) * 1000
    tempos_m2.append(t_m2)
    angulos_m2.append(a_m2)

    t0 = time.time()
    c_m1 = executar_metodo_1(bin_m2)
    t_m1 = (time.time() - t0) * 1000
    tempos_m1.append(t_m1)

    t0 = time.time()
    c_m3, a_m3, bin_m3, pt_base_m3, pts_m3, edges_m3 = executar_metodo_3(frame_rgb)
    t_m3 = (time.time() - t0) * 1000
    tempos_m3.append(t_m3)
    angulos_m3.append(a_m3)

    escritor_csv.writerow([frame_id, round(t_m3, 2), c_m1, c_m2, round(a_m2, 2), c_m3, round(a_m3, 2)])

    p1 = frame_red.copy()
    p2 = cv2.cvtColor(bin_m2, cv2.COLOR_GRAY2BGR)
    p3 = cv2.cvtColor(bin_m3, cv2.COLOR_GRAY2BGR)

    for i, j in edges_m2:
        cv2.line(p2, pts_m2[i], pts_m2[j], (0, 255, 255), 1)
    for pt in pts_m2:
        cv2.circle(p2, pt, 2, (0, 255, 0), -1)
    if pt_base_m2:
        cv2.circle(p2, pt_base_m2, 3, (0, 0, 255), -1)

    for i, j in edges_m3:
        cv2.line(p3, pts_m3[i], pts_m3[j], (0, 255, 255), 1)
    for pt in pts_m3:
        cv2.circle(p3, pt, 2, (0, 255, 0), -1)
    if pt_base_m3:
        cv2.circle(p3, pt_base_m3, 3, (0, 0, 255), -1)

    cv2.putText(p1, f"M1: {c_m1}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.putText(p2, f"M2: {c_m2} ({a_m2:.1f}deg)", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    cv2.putText(p3, f"M3: {c_m3} ({a_m3:.1f}deg)", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 100, 255), 1)
    
    cv2.putText(p3, f"ANG HIBRIDO: {a_m3:.1f}deg", (5, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    out.write(cv2.resize(np.hstack((p1, p2, p3)), (768, 256), interpolation=cv2.INTER_NEAREST))

cap.release()
out.release()
arquivo_csv.close()