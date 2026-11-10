import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

ARQUIVO_DATASET = "dataset_regressao_angulo.npz"
MODELO_SAIDA = "cnn_controle_angular.h5"

dados = np.load(ARQUIVO_DATASET)
X, Y = dados['x'], dados['y']

def criar_cnn_angular():
    inputs = layers.Input(shape=(128, 128, 1))
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32, activation='relu')(x)

    outputs = layers.Dense(1, activation='tanh')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

model = criar_cnn_angular()
print("[INFO] Treinando CNN de Controle Angular com ativação tanh e loss estroncada...")
model.fit(X, Y, epochs=40, batch_size=16, validation_split=0.2)
model.save(MODELO_SAIDA)
print(f"[CONCLUÍDO] Modelo salvo em '{MODELO_SAIDA}'. A Loss esperada agora deve ser < 0.05!")