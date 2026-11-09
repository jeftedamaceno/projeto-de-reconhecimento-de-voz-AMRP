import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

# ==========================================
# 1. PARÂMETROS E CONFIGURAÇÕES
# ==========================================
IMG_SIZE = (128, 128)
DATASET_DIR = "dataset_hibrido_avancado"
SAMPLES = 1200
EPOCHS = 20
BATCH_SIZE = 16

MODELO_H5 = "modelo_hibrido_segmentador.h5"

os.makedirs(os.path.join(DATASET_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "masks"), exist_ok=True)

# Dice Loss para lidar com desbalanceamento entre fita e chão claro
def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# ==========================================
# 2. GERADOR SINTÉTICO COM REFLEXOS DE ILUMINAÇÃO
# ==========================================
def gerar_par_sintetico(w=128, h=128):
    tom_fundo = np.random.randint(180, 240)
    img = np.ones((h, w, 3), dtype=np.uint8) * tom_fundo
    mask = np.zeros((h, w), dtype=np.uint8)

    # Simula clarões e reflexos fortes de lâmpadas no piso
    for _ in range(np.random.randint(1, 3)):
        cx, cy = np.random.randint(20, w-20), np.random.randint(20, h-20)
        axes = (np.random.randint(20, 50), np.random.randint(10, 30))
        cv2.ellipse(img, (cx, cy), axes, np.random.randint(0, 180), 0, 360, (255, 255, 255), -1)
    img = cv2.GaussianBlur(img, (9, 9), 0)

    num_fitas = np.random.randint(1, 4)
    tom_fita = np.random.randint(10, 60)

    for _ in range(num_fitas):
        orientacao = np.random.choice(["v", "h", "diag"])
        espessura = np.random.randint(5, 14)

        if orientacao == "v":
            x = np.random.randint(15, w - 15)
            cv2.line(img, (x, 0), (x, h), (tom_fita, tom_fita, tom_fita), espessura)
            cv2.line(mask, (x, 0), (x, h), 255, espessura)
        elif orientacao == "h":
            y = np.random.randint(15, h - 15)
            cv2.line(img, (0, y), (w, y), (tom_fita, tom_fita, tom_fita), espessura)
            cv2.line(mask, (0, y), (w, y), 255, espessura)
        else:
            p1 = (np.random.randint(0, w), np.random.randint(0, h))
            p2 = (np.random.randint(0, w), np.random.randint(0, h))
            cv2.line(img, p1, p2, (tom_fita, tom_fita, tom_fita), espessura)
            cv2.line(mask, p1, p2, 255, espessura)

    return img, mask

print("[1/3] Gerando dados sintéticos com reflexos de luz...")
for i in range(SAMPLES):
    img, mask = gerar_par_sintetico()
    cv2.imwrite(os.path.join(DATASET_DIR, "images", f"{i:04d}.jpg"), img)
    cv2.imwrite(os.path.join(DATASET_DIR, "masks", f"{i:04d}.png"), mask)

# ==========================================
# 3. CONSTRUÇÃO E TREINO DA UNET LEVE
# ==========================================
def carregar_dataset():
    X, Y = [], []
    for i in range(SAMPLES):
        img = cv2.imread(os.path.join(DATASET_DIR, "images", f"{i:04d}.jpg"))
        mask = cv2.imread(os.path.join(DATASET_DIR, "masks", f"{i:04d}.png"), cv2.IMREAD_GRAYSCALE)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X.append(img_rgb / 255.0)
        Y.append(np.expand_dims(mask / 255.0, axis=-1))
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

X, Y = carregar_dataset()

inputs = layers.Input(shape=(128, 128, 3))
c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
c1 = layers.BatchNormalization()(c1)
p1 = layers.MaxPooling2D((2, 2))(c1)

c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
c2 = layers.BatchNormalization()(c2)
p2 = layers.MaxPooling2D((2, 2))(c2)

b = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)

u1 = layers.UpSampling2D((2, 2))(b)
concat1 = layers.Concatenate()([u1, c2])
d1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(concat1)

u2 = layers.UpSampling2D((2, 2))(d1)
concat2 = layers.Concatenate()([u2, c1])
d2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(concat2)

outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(d2)

model = models.Model(inputs, outputs)
model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy'])

print("[2/3] Treinando Modelo Híbrido Avançado...")
model.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

model.save(MODELO_H5)
print(f"[3/3] Modelo salvo em: {MODELO_H5}")