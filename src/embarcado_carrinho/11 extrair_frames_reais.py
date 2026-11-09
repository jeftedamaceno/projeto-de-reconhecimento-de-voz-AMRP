import cv2
import os

VIDEOS = ["pontilhado.mp4", "pontilhado2.mp4"]
PASTA_SAIDA = "dataset_real_pista/images"
INTERVALO_FRAMES = 12 # Extrai 1 frame a cada 12 (evita redundância)

os.makedirs(PASTA_SAIDA, exist_ok=True)
contador_salvos = 0

for video_path in VIDEOS:
    if not os.path.exists(video_path):
        print(f"[AVISO] Vídeo {video_path} não encontrado, pulando...")
        continue
        
    cap = cv2.VideoCapture(video_path)
    count = 0
    print(f"[INFO] Extraindo frames de {video_path}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % INTERVALO_FRAMES == 0:
            frame_red = cv2.resize(frame, (128, 128))
            caminho_img = os.path.join(PASTA_SAIDA, f"real_{contador_salvos:04d}.jpg")
            cv2.imwrite(caminho_img, frame_red)
            contador_salvos += 1
            
        count += 1
    cap.release()

print(f"[CONCLUÍDO] {contador_salvos} imagens reais salvas em '{PASTA_SAIDA}'.")