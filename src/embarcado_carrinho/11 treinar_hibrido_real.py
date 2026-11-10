import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

IMG_SIZE = (128, 128)
DIR_IMAGES = "dataset_real_pista/images"
DIR_MASKS = "dataset_real_pista/masks"
MODELO_SAIDA = "modelo_hibrido_segmentador.h5"


def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

print("[1/3] Carregando pares de imagens e máscaras reais...")
caminhos_img = sorted(glob.glob(os.path.join(DIR_IMAGES, "*.jpg")))
caminhos_mask = sorted(glob.glob(os.path.join(DIR_MASKS, "*.png")))

X, Y = [], []
for c_img, c_mask in zip(caminhos_img, caminhos_mask):
    img = cv2.imread(c_img)
    mask = cv2.imread(c_mask, cv2.IMREAD_GRAYSCALE)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X.append(img_rgb / 255.0)
    Y.append(np.expand_dims(mask / 255.0, axis=-1))

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32)

print(f"[INFO] Dataset carregado: {X.shape[0]} amostras reais.")


data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.5, fill_mode='reflect'), 
    layers.RandomZoom(0.2, fill_mode='reflect'),     
    layers.RandomTranslation(0.1, 0.1, fill_mode='reflect')
])

def construir_unet_real(input_shape=(128, 128, 3)):
    inputs = layers.Input(shape=input_shape)
   
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

model = construir_unet_real()
model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy'])

print("[2/3] Treinando U-Net Híbrida com imagens reais e augmentação de inclinação...")
model.fit(X, Y, epochs=25, batch_size=8, validation_split=0.2)

model.save(MODELO_SAIDA)
print(f"[3/3] Modelo Híbrido atualizado e salvo em '{MODELO_SAIDA}'!")