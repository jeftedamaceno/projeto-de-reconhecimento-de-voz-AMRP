import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

IMG_SIZE = (128, 128)
PASTA_BASE = "dataset_real_pista"
MODELO_SAIDA = "modelo_hibrido_segmentador.h5"

def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def carregar_conjunto(subpasta):
    dir_img = os.path.join(PASTA_BASE, subpasta, "images")
    dir_mask = os.path.join(PASTA_BASE, subpasta, "masks")
    
    caminhos_img = sorted(glob.glob(os.path.join(dir_img, "*.jpg")))
    caminhos_mask = sorted(glob.glob(os.path.join(dir_mask, "*.png")))
    
    X, Y = [], []
    for c_i, c_m in zip(caminhos_img, caminhos_mask):
        img = cv2.imread(c_i)
        mask = cv2.imread(c_m, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None: continue
        
        X.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0)
        Y.append(np.expand_dims(mask / 255.0, axis=-1))
        
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

print("[1/3] Carregando partições independentes de Treino e Validação...")
X_train, Y_train = carregar_conjunto("treino")
X_val, Y_val = carregar_conjunto("validacao")

print(f" Treino   : {X_train.shape[0]} amostras")
print(f" Validação: {X_val.shape[0]} amostras")

data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.2, fill_mode='reflect'),
    layers.RandomZoom(0.1, fill_mode='reflect'),
    layers.RandomContrast(0.15)
])

def construir_unet():
    inputs = layers.Input(shape=(128, 128, 3))
    aug = data_augmentation(inputs)

    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(aug)
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
    return models.Model(inputs, outputs)

model = construir_unet()
model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy'])

print("[2/3] Treinando Modelo Híbrido U-Net...")
model.fit(X_train, Y_train, epochs=30, batch_size=8, validation_data=(X_val, Y_val))

model.save(MODELO_SAIDA)
print(f"[3/3] Treinamento concluído. Modelo salvo em '{MODELO_SAIDA}'.")