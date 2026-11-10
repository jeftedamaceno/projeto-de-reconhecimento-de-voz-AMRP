import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K


IMG_SIZE = (128, 128)
DIR_IMAGES = "dataset_real/images"
DIR_MASKS = "dataset_real/masks"
MODELO_SAIDA = "modelo_hibrido_segmentador.h5"


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = 1.0 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return bce + dice

print("[1/3] Carregando dataset e aplicando auto-rotulagem do M2...")
caminhos_img = sorted(glob.glob(os.path.join(DIR_IMAGES, "*.jpg")))

X, Y = [], []
for c_img in caminhos_img:
    img = cv2.imread(c_img)
    if img is None: continue
    
    img_red = cv2.resize(img, IMG_SIZE)
    

    cinza = cv2.cvtColor(img_red, cv2.COLOR_BGR2GRAY)
    desfocado = cv2.GaussianBlur(cinza, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    blackhat = cv2.morphologyEx(desfocado, cv2.MORPH_BLACKHAT, kernel)
    _, mask_m2 = cv2.threshold(blackhat, 25, 255, cv2.THRESH_BINARY)
    
    X.append(cv2.cvtColor(img_red, cv2.COLOR_BGR2RGB) / 255.0)
    Y.append(np.expand_dims(mask_m2 / 255.0, axis=-1))

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32)


data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.3, fill_mode='reflect'),
    layers.RandomZoom(0.15, fill_mode='reflect'),
    layers.RandomContrast(0.2)
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
model.compile(optimizer='adam', loss=bce_dice_loss, metrics=['accuracy'])

print("[2/3] Treinando U-Net supervisionada pelo M2...")
model.fit(X, Y, epochs=30, batch_size=8, validation_split=0.2)

model.save(MODELO_SAIDA)
print(f"[3/3] Modelo salvo em: '{MODELO_SAIDA}'")