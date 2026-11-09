import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ==========================================
# 1. CONFIGURAÇÕES E PARÂMETROS
# ==========================================
IMG_SIZE = (128, 128)
CLASSES = [
    "Reta", 
    "Curva_Aberta_Esq", 
    "Curva_Aberta_Dir", 
    "Curva_Fechada_Esq", 
    "Curva_Fechada_Dir",
    "Bifurcacao_Esq_Dir",   # Y (2 saídas no topo)
    "Cruzamento_3Pistas",   # T (Pista segue reto e abre para lados)
    "Cruzamento_4Pistas",   # X (Pista segue reto e corta ambos os lados)
    "Sem_Caminho"
]

DATASET_DIR = "dataset_sintetico_multirota"
SAMPLES_PER_CLASS = 400 # Aumentado para melhor generalização
EPOCHS = 25
MARGEM_ERRO = 5.0

for c in CLASSES:
    os.makedirs(os.path.join(DATASET_DIR, c), exist_ok=True)

# ==========================================
# 2. GERADOR SINTÉTICO AVANÇADO (RUÍDO + PERSPECTIVA)
# ==========================================
def desenhar_linha_pista(img, angulo, center_x, frame_h, tom_linha, eh_continua):
    """Desenha segmento de pista (contínuo ou tracejado) com um determinado ângulo."""
    if eh_continua:
        offset_inicio = int(np.tan(np.radians(angulo)) * frame_h)
        pt_inicio = (center_x + offset_inicio, 0)
        pt_fim = (center_x, frame_h)
        cv2.line(img, pt_inicio, pt_fim, (tom_linha, tom_linha, tom_linha), 6)
    else:
        for y in range(0, frame_h, 18):
            offset = int(np.tan(np.radians(angulo)) * (frame_h - y))
            x = center_x + offset
            cv2.line(img, (x, y), (x, y + 9), (tom_linha, tom_linha, tom_linha), 6)

def adicionar_ruido_amorfo(img, frame_w, frame_h):
    """Gera manchas amorfas, linhas finas (rejunte) e pequenos ruídos aleatórios."""
    num_ruidos = np.random.randint(2, 6)
    for _ in range(num_ruidos):
        tipo = np.random.choice(["poligono", "linha_fina", "circulo"])
        tom_ruido = np.random.randint(20, 100)
        
        if tipo == "poligono":
            # Polígono amorfo
            pts = np.random.randint(0, frame_w, (np.random.randint(3, 6), 2))
            cv2.fillPoly(img, [pts], (tom_ruido, tom_ruido, tom_ruido))
        elif tipo == "linha_fina":
            # Risco/rejunte fino do piso
            p1 = (np.random.randint(0, frame_w), np.random.randint(0, frame_h))
            p2 = (np.random.randint(0, frame_w), np.random.randint(0, frame_h))
            cv2.line(img, p1, p2, (tom_ruido, tom_ruido, tom_ruido), 1)
        elif tipo == "circulo":
            center = (np.random.randint(0, frame_w), np.random.randint(0, frame_h))
            radius = np.random.randint(2, 8)
            cv2.circle(img, center, radius, (tom_ruido, tom_ruido, tom_ruido), -1)

def gerar_frame_sintetico(classe, frame_w=128, frame_h=128):
    tom_fundo = np.random.randint(180, 240)
    top_down = np.ones((frame_h, frame_w, 3), dtype=np.uint8) * tom_fundo

    if classe == "Sem_Caminho":
        adicionar_ruido_amorfo(top_down, frame_w, frame_h)
        return top_down

    center_x = (frame_w // 2) + np.random.randint(-10, 10)
    tom_linha = np.random.randint(10, 50)
    eh_continua = np.random.choice([True, False])
    tolerancia = np.random.uniform(-MARGEM_ERRO, MARGEM_ERRO)

    # Geração dos padrões de pista conforme a classe
    if classe == "Reta":
        desenhar_linha_pista(top_down, 0 + tolerancia, center_x, frame_h, tom_linha, eh_continua)
    elif classe == "Curva_Aberta_Esq":
        desenhar_linha_pista(top_down, -25 + tolerancia, center_x, frame_h, tom_linha, eh_continua)
    elif classe == "Curva_Aberta_Dir":
        desenhar_linha_pista(top_down, 25 + tolerancia, center_x, frame_h, tom_linha, eh_continua)
    elif classe == "Curva_Fechada_Esq":
        desenhar_linha_pista(top_down, -60 + tolerancia, center_x, frame_h, tom_linha, eh_continua)
    elif classe == "Curva_Fechada_Dir":
        desenhar_linha_pista(top_down, 60 + tolerancia, center_x, frame_h, tom_linha, eh_continua)
    elif classe == "Bifurcacao_Esq_Dir":
        # Rota abre em duas (Y)
        desenhar_linha_pista(top_down, -35 + tolerancia, center_x, frame_h, tom_linha, eh_continua)
        desenhar_linha_pista(top_down, 35 + tolerancia, center_x, frame_h, tom_linha, eh_continua)
    elif classe == "Cruzamento_3Pistas":
        # Rota segue reto + ramos para lados (T)
        desenhar_linha_pista(top_down, 0, center_x, frame_h, tom_linha, eh_continua)
        cv2.line(top_down, (0, frame_h//2), (frame_w, frame_h//2), (tom_linha, tom_linha, tom_linha), 5)
    elif classe == "Cruzamento_4Pistas":
        # Cruzamento completo (X)
        desenhar_linha_pista(top_down, 0, center_x, frame_h, tom_linha, eh_continua)
        desenhar_linha_pista(top_down, 45, center_x, frame_h, tom_linha, eh_continua)
        desenhar_linha_pista(top_down, -45, center_x, frame_h, tom_linha, eh_continua)

    # Adiciona ruído de fundo antes da perspectiva
    adicionar_ruido_amorfo(top_down, frame_w, frame_h)

    # Varie a perspectiva/ângulo da câmera de forma mais agressiva
    var_p = np.random.uniform(0.20, 0.40) # Ângulo de abertura
    y_horizonte = np.random.uniform(0.15, 0.30) # Variação de pitch/altura da câmera
    
    src_points = np.float32([[0, 0], [frame_w, 0], [0, frame_h], [frame_w, frame_h]])
    dst_points = np.float32([
        [int(frame_w * var_p), int(frame_h * y_horizonte)],
        [int(frame_w * (1 - var_p)), int(frame_h * y_horizonte)],
        [-int(frame_w * 0.1), frame_h],
        [int(frame_w * 1.1), frame_h]
    ])

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    frame_final = cv2.warpPerspective(top_down, M, (frame_w, frame_h), borderValue=(tom_fundo, tom_fundo, tom_fundo))
    return frame_final

print("[1/3] Gerando dataset sintético avançado (Multi-rotas, Ruídos Amorfos e Variação de Câmera)...")
for c in CLASSES:
    for i in range(SAMPLES_PER_CLASS):
        img = gerar_frame_sintetico(c, IMG_SIZE[0], IMG_SIZE[1])
        cv2.imwrite(os.path.join(DATASET_DIR, c, f"sintetico_{i:04d}.jpg"), img)

# ==========================================
# 3. PREPARAÇÃO E ARQUITETURA DA CNN
# ==========================================
print("[2/3] Carregando e dividindo dataset...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=16
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=16
)

# Arquitetura com Dropout para evitar overfit devido aos ruídos amorfos
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(128, 128, 3)),
    
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.1),
    
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.1),
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(CLASSES), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"[3/3] Treinando o modelo por {EPOCHS} épocas...")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

model.save("modelo_sintetico_comparacao.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("modelo_sintetico_comparacao.tflite", "wb") as f:
    f.write(tflite_model)

print("\n[PROCESSO CONCLUÍDO] Novo modelo treinado e exportado com sucesso!")