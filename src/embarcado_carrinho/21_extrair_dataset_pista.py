import cv2
import os
import glob
import random

VIDEOS = ["pontilhado3.mp4", "pontilhado2.mp4"]
PASTA_BASE = "dataset_real_pista"
PASTA_PREVIEW = "preview_mascaras"

INTERVALO_FRAMES = 10 
RATIO_VALIDACAO = 0.20 

PASTA_TREINO_IMG = os.path.join(PASTA_BASE, "treino", "images")
PASTA_TREINO_MASK = os.path.join(PASTA_BASE, "treino", "masks")
PASTA_VAL_IMG = os.path.join(PASTA_BASE, "validacao", "images")
PASTA_VAL_MASK = os.path.join(PASTA_BASE, "validacao", "masks")

for p in [PASTA_TREINO_IMG, PASTA_TREINO_MASK, PASTA_VAL_IMG, PASTA_VAL_MASK, PASTA_PREVIEW]:
    os.makedirs(p, exist_ok=True)

def gerar_mascara_antireflexo(frame_bgr):
    h, w = frame_bgr.shape[:2]
    cinza = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    

    _, mascara_clarao = cv2.threshold(cinza, 220, 255, cv2.THRESH_BINARY)
    
    desfocado = cv2.GaussianBlur(cinza, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    blackhat = cv2.morphologyEx(desfocado, cv2.MORPH_BLACKHAT, kernel)
    

    blackhat_limpo = cv2.subtract(blackhat, mascara_clarao)
    
    _, binaria = cv2.threshold(blackhat_limpo, 30, 255, cv2.THRESH_BINARY)
    

    mascara_roi = np.zeros_like(binaria)
    pts_roi = np.array([
        [int(w * 0.10), int(h * 0.25)],
        [int(w * 0.90), int(h * 0.25)],
        [w, h],
        [0, h]
    ], dtype=np.int32)
    cv2.fillPoly(mascara_roi, [pts_roi], 255)
    
    binaria_roi = cv2.bitwise_and(binaria, binaria, mask=mascara_roi)
    

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binaria_roi, connectivity=8)
    mask_final = np.zeros_like(binaria_roi)
    for i in range(1, num_labels):
        w_comp, h_comp, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        aspect_ratio = max(w_comp, h_comp) / (min(w_comp, h_comp) + 1e-5)
        
      
        if 15 <= area <= 400 and w_comp >= 3 and h_comp >= 3 and aspect_ratio < 4.0:
            mask_final[labels == i] = 255
            
    return mask_final

import numpy as np

contador_frames = 0
for video_path in VIDEOS:
    if not os.path.exists(video_path):
        continue
    
    cap = cv2.VideoCapture(video_path)
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if count % INTERVALO_FRAMES == 0:
            frame_red = cv2.resize(frame, (128, 128))
            mask_red = gerar_mascara_antireflexo(frame_red)
            
      
            if random.random() > RATIO_VALIDACAO:
                dest_img = PASTA_TREINO_IMG
                dest_mask = PASTA_TREINO_MASK
            else:
                dest_img = PASTA_VAL_IMG
                dest_mask = PASTA_VAL_MASK
                
            nome_base = f"pista_{contador_frames:04d}"
            cv2.imwrite(os.path.join(dest_img, f"{nome_base}.jpg"), frame_red)
            cv2.imwrite(os.path.join(dest_mask, f"{nome_base}.png"), mask_red)
         
            preview = np.hstack((frame_red, cv2.cvtColor(mask_red, cv2.COLOR_GRAY2BGR)))
            cv2.imwrite(os.path.join(PASTA_PREVIEW, f"{nome_base}_preview.jpg"), preview)
            
            contador_frames += 1
        count += 1
    cap.release()

print(f"[CONCLUÍDO] Dataset salvo em '{PASTA_BASE}'.")
print(f" Imagens para Inspeção salvas na pasta: '{PASTA_PREVIEW}'")