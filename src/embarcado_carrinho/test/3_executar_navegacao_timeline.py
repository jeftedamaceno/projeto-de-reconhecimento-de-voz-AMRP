import cv2
import numpy as np
import time
import os
import json

VIDEO_ENTRADA = "pontilhado3.mp4"
VIDEO_SAIDA = "resultado_navegacao_topdown.mp4"
JSON_TIMELINE = "timeline_pista.json"

LARGURA_PROC, ALTURA_PROC = 128, 128

model_cnn = None
try:
    import tensorflow as tf
    if os.path.exists("cnn_controle_angular.h5"):
        model_cnn = tf.keras.models.load_model("cnn_controle_angular.h5")
        print("[OK] Modelo CNN de Controle Angular carregado com sucesso.")
except Exception:
    pass

def binarizar_antireflexo(frame_bgr):
    frame_red = cv2.resize(frame_bgr, (LARGURA_PROC, ALTURA_PROC))
    cinza = cv2.cvtColor(frame_red, cv2.COLOR_BGR2GRAY)
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
    return cv2.bitwise_and(binaria, binaria, mask=mascara_roi), frame_red

def desenhar_painel_topdown(angulo_deg, classe_str):
    canvas = np.zeros((ALTURA_PROC, LARGURA_PROC, 3), dtype=np.uint8)
    
    for y in range(0, ALTURA_PROC, 32):
        cv2.line(canvas, (0, y), (LARGURA_PROC, y), (30, 30, 30), 1)
    for x in range(0, LARGURA_PROC, 32):
        cv2.line(canvas, (x, 0), (x, ALTURA_PROC), (30, 30, 30), 1)

    robo_x, robo_y = LARGURA_PROC // 2, ALTURA_PROC - 15
    cv2.circle(canvas, (robo_x, robo_y), 4, (0, 0, 255), -1)

    rad = np.radians(angulo_deg)
    
    pt1_x = int(robo_x + 25 * np.sin(rad))
    pt1_y = int(robo_y - 25 * np.cos(rad))
    
    pt2_x = int(pt1_x + 30 * np.sin(rad))
    pt2_y = int(pt1_y - 30 * np.cos(rad))

    pt3_x = int(pt2_x + 35 * np.sin(rad))
    pt3_y = int(pt2_y - 35 * np.cos(rad))

    pts_trajetoria = [(robo_x, robo_y), (pt1_x, pt1_y), (pt2_x, pt2_y), (pt3_x, pt3_y)]

  
    for i in range(len(pts_trajetoria) - 1):
        cv2.line(canvas, pts_trajetoria[i], pts_trajetoria[i+1], (0, 255, 255), 2)
        cv2.circle(canvas, pts_trajetoria[i+1], 3, (0, 255, 0), -1)

    cv2.putText(canvas, "TOP-DOWN 2D", (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 255), 1)
    cv2.putText(canvas, f"ANG: {angulo_deg:.1f}d", (5, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 255, 255), 1)
    return canvas

cap = cv2.VideoCapture(VIDEO_ENTRADA)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
fourcc = cv2.VideoWriter_fourcc(*"XVID")

out = cv2.VideoWriter(VIDEO_SAIDA, fourcc, fps, (384, 128), True)

while True:
    ret, frame = cap.read()
    if not ret: break

    mask, frame_red = binarizar_antireflexo(frame)
    
    if model_cnn is not None:
        inp = np.expand_dims(mask.astype(np.float32) / 255.0, axis=(0, -1))
      
        ang_norm = float(model_cnn.predict(inp, verbose=0)[0][0])
        angulo_predito = ang_norm * 60.0
    else:
        angulo_predito = 0.0

    if abs(angulo_predito) <= 8.0:
        classe_str = "Reta"
    elif abs(angulo_predito) <= 25.0:
        classe_str = "Curva_Suave"
    else:
        classe_str = "Curva_Fechada"

    p1 = frame_red.copy()
    cv2.putText(p1, f"CAM: {classe_str}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    p2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cx, cy = LARGURA_PROC // 2, ALTURA_PROC - 10
    dx = int(30 * np.sin(np.radians(angulo_predito)))
    dy = int(30 * np.cos(np.radians(angulo_predito)))
    cv2.arrowedLine(p2, (cx, cy), (cx + dx, cy - dy), (0, 255, 0), 2)
    cv2.putText(p2, f"MASCARA", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

   
    p3 = desenhar_painel_topdown(angulo_predito, classe_str)

   
    painel_triplo = np.hstack((p1, p2, p3))
    out.write(painel_triplo)

cap.release()
out.release()
print(f"[CONCLUÍDO] Vídeo com visualização Top-Down 2D gerado: '{VIDEO_SAIDA}'")