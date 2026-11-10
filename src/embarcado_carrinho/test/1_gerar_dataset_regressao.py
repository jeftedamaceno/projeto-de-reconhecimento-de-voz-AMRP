import cv2
import numpy as np
import os

VIDEOS = ["pontilhado.mp4", "pontilhado2.mp4", "pontilhado3.mp4"]
ARQUIVO_SAIDA = "dataset_regressao_angulo.npz"

LARGURA_PROC, ALTURA_PROC = 128, 128
INTERVALO_FRAMES = 3

def binarizar_antireflexo(frame_bgr):
    cinza = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, mascara_clarao = cv2.threshold(cinza, 220, 255, cv2.THRESH_BINARY)
    desfocado = cv2.GaussianBlur(cinza, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    blackhat = cv2.morphologyEx(desfocado, cv2.MORPH_BLACKHAT, kernel)
    blackhat_limpo = cv2.subtract(blackhat, mascara_clarao)
    _, binaria = cv2.threshold(blackhat_limpo, 25, 255, cv2.THRESH_BINARY)
    
    mascara_roi = np.zeros_like(binaria)
    pts_roi = np.array([
        [int(LARGURA_PROC * 0.10), int(ALTURA_PROC * 0.25)],
        [int(LARGURA_PROC * 0.90), int(ALTURA_PROC * 0.25)],
        [LARGURA_PROC, ALTURA_PROC],
        [0, ALTURA_PROC]
    ], dtype=np.int32)
    cv2.fillPoly(mascara_roi, [pts_roi], 255)
    return cv2.bitwise_and(binaria, binaria, mask=mascara_roi)

def calcular_angulo_rotulo(mask_img):
    num_labels, labels, stats, centroides = cv2.connectedComponentsWithStats(mask_img, connectivity=8)
    pontos = []
    for i in range(1, num_labels):
        w, h, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if area >= 12 and w >= 2 and h >= 2:
            pontos.append((centroides[i][0], centroides[i][1]))
            
    if len(pontos) < 2:
        return None
        
    pontos = sorted(pontos, key=lambda p: p[1], reverse=True)
    pt_base = pontos[0]
    pt_topo = pontos[-1]
    
    dx = pt_topo[0] - pt_base[0]
    dy = pt_base[1] - pt_topo[1]
    
    if dy <= 0:
        return None
        
    ang = np.degrees(np.arctan2(dx, dy))
    
    ang_clamped = np.clip(ang, -60.0, 60.0)
  
    return ang_clamped / 60.0

X_mascaras, Y_angulos = [], []

for vid_path in VIDEOS:
    if not os.path.exists(vid_path): continue
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if count % INTERVALO_FRAMES == 0:
            frame_red = cv2.resize(frame, (LARGURA_PROC, ALTURA_PROC))
            mask = binarizar_antireflexo(frame_red)
            ang_norm = calcular_angulo_rotulo(mask)
            if ang_norm is not None:
                X_mascaras.append(np.expand_dims(mask.astype(np.float32) / 255.0, axis=-1))
                Y_angulos.append(ang_norm)
        count += 1
    cap.release()

X_mascaras = np.array(X_mascaras, dtype=np.float32)
Y_angulos = np.array(Y_angulos, dtype=np.float32)

np.savez_compressed(ARQUIVO_SAIDA, x=X_mascaras, y=Y_angulos)
print(f"[OK] {len(X_mascaras)} amostras salvas. Alvos normalizados em [-1.0, 1.0].")