import cv2
import numpy as np
import os
import glob

DIR_IMAGES = "dataset_real_pista/images"
DIR_MASKS = "dataset_real_pista/masks"

os.makedirs(DIR_MASKS, exist_ok=True)

lista_imagens = glob.glob(os.path.join(DIR_IMAGES, "*.jpg"))
print(f"[INFO] Gerando máscaras automáticas para {len(lista_imagens)} imagens...")

for cam_img in lista_imagens:
    nome_arquivo = os.path.basename(cam_img).replace(".jpg", ".png")
    img = cv2.imread(cam_img)
    
    # Processamento Black Hat para isolar fitas pretas reais
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    desfocado = cv2.GaussianBlur(cinza, (5, 5), 0)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    blackhat = cv2.morphologyEx(desfocado, cv2.MORPH_BLACKHAT, kernel)
    
    _, binaria = cv2.threshold(blackhat, 25, 255, cv2.THRESH_BINARY)
    
    # Filtro de tamanho para remover riscos finos do piso
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    mask_limpa = np.zeros_like(binaria)
    for i in range(1, num_labels):
        w, h, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if area >= 15 and w >= 3 and h >= 4:
            mask_limpa[labels == i] = 255
            
    cv2.imwrite(os.path.join(DIR_MASKS, nome_arquivo), mask_limpa)

print(f"[CONCLUÍDO] Máscaras salvas em '{DIR_MASKS}'.")