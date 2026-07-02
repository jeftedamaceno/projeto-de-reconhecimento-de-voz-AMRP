# import os
# import json
# import wave
# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import (
#     Input, Conv1D, MaxPooling1D, BatchNormalization,
#     Bidirectional, LSTM, Dense, Dropout, Layer
# )
# from tensorflow.keras.losses import CategoricalCrossentropy
# from sklearn.metrics import confusion_matrix, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ==========================================
# # 1. CONFIGURAÇÕES E CAMINHOS
# # ==========================================
# AUDIO_ORIGINAL = "dataset_final"
# AUDIO_RUIDO = "dataset_ruido2"

# SAMPLE_RATE = 16000
# DURATION = 1  
# MAX_SAMPLES = SAMPLE_RATE * DURATION # 16000 pontos por áudio

# labels = sorted(os.listdir(AUDIO_ORIGINAL))
# label_map = {label: i for i, label in enumerate(labels)}
# num_classes = len(labels)

# # Salva o mapa de classes em disco para uso futuro na inferência
# with open("label_map_hibrido.json", "w") as f:
#     json.dump(label_map, f, indent=4)

# # ==========================================
# # 2. CAMADA DE ATENÇÃO PERSONALIZADA (SERIALIZÁVEL)
# # ==========================================
# @tf.keras.utils.register_keras_serializable(package="Custom")
# class AttentionLayer(Layer):
#     """Mecanismo de Atenção Temporal para focar no momento exato da fala"""
#     def __init__(self, **kwargs):
#         super(AttentionLayer, self).__init__(**kwargs)

#     def build(self, input_shape):
#         self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
#         self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
#         super(AttentionLayer, self).build(input_shape)

#     def call(self, inputs):
#         e = tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W) + self.b)
#         a = tf.keras.backend.softmax(e, axis=1)
#         output = inputs * a
#         return tf.keras.backend.sum(output, axis=1)

#     def get_config(self):
#         return super(AttentionLayer, self).get_config()

# # ==========================================
# # 3. LEITOR DE ÁUDIO MANUAL (SEM LIBROSA)
# # ==========================================
# def carregar_wav_manual(file_path):
#     with wave.open(file_path, 'rb') as wav_file:
#         n_channels = wav_file.getnchannels()
#         sampwidth = wav_file.getsampwidth()
#         n_frames = wav_file.getnframes()
#         raw_data = wav_file.readframes(n_frames)
        
#         if sampwidth == 2:
#             audio_data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
#         else:
#             raise ValueError("Suporta apenas arquivos WAV de 16-bit PCM.")
            
#         if n_channels > 1:
#             audio_data = audio_data.reshape(-1, n_channels).mean(axis=1)
#         return audio_data

# def audio_to_sequence(file_path):
#     audio = carregar_wav_manual(file_path)
#     if len(audio) < MAX_SAMPLES:
#         audio = np.pad(audio, (0, int(MAX_SAMPLES - len(audio))), 'constant')
#     else:
#         audio = audio[:int(MAX_SAMPLES)]
        
#     if np.std(audio) > 0:
#         audio = (audio - np.mean(audio)) / np.std(audio)
#     return audio

# # ==========================================
# # 4. CARREGAMENTO E PROCESSAMENTO DO DATASET
# # ==========================================
# X_raw, y_raw = [], []

# print("Carregando dataset original...")
# for label in labels:
#     path = os.path.join(AUDIO_ORIGINAL, label)
#     if not os.path.isdir(path): continue
#     for file in os.listdir(path):
#         if file.endswith(".wav"):
#             try:
#                 X_raw.append(audio_to_sequence(os.path.join(path, file)))
#                 y_raw.append(label_map[label])
#             except: pass

# X_raw = np.array(X_raw)
# y_raw = np.array(y_raw)

# X_train_orig, X_val, y_train_orig, y_val = train_test_split(
#     X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=42
# )

# X_aug, y_aug = [], []
# print("Carregando dataset com ruído para balanceamento...")
# for label in labels:
#     path = os.path.join(AUDIO_RUIDO, label)
#     if not os.path.isdir(path): continue
#     for file in os.listdir(path):
#         if file.endswith(".wav"):
#             try:
#                 X_aug.append(audio_to_sequence(os.path.join(path, file)))
#                 y_aug.append(label_map[label])
#             except: pass

# X_aug = np.array(X_aug)
# y_aug = np.array(y_aug)

# MAX_AUG_PER_CLASS = 800
# X_aug_bal, y_aug_bal = [], []
# for label in range(len(labels)):
#     idx = np.where(y_aug == label)[0]
#     if len(idx) > 0:
#         np.random.shuffle(idx)
#         idx = idx[:MAX_AUG_PER_CLASS]
#         X_aug_bal.append(X_aug[idx])
#         y_aug_bal.append(y_aug[idx])

# X_aug_bal = np.concatenate(X_aug_bal)
# y_aug_bal = np.concatenate(y_aug_bal)

# X_train = np.concatenate([X_train_orig, X_aug_bal])
# y_train = np.concatenate([y_train_orig, y_aug_bal])

# idx = np.random.permutation(len(X_train))
# X_train = X_train[idx]
# y_train = y_train[idx]

# y_train_cat = to_categorical(y_train, num_classes=num_classes)
# y_val_cat = to_categorical(y_val, num_classes=num_classes)

# # Formato final para entrada 1D: (N, 16000, 1)
# X_train = np.expand_dims(X_train, axis=-1)
# X_val = np.expand_dims(X_val, axis=-1)

# # ==========================================
# # 5. ARQUITETURA HÍBRIDA (CONV1D + BI-LSTM + ATTENTION)
# # ==========================================
# inputs = Input(shape=(16000, 1))

# # Bloco Extrator Convolucional 1D (Limpeza e compactação)
# x = Conv1D(16, kernel_size=16, strides=4, padding='same', activation='relu')(inputs)
# x = MaxPooling1D(pool_size=4)(x)
# x = BatchNormalization()(x)

# x = Conv1D(32, kernel_size=8, strides=2, padding='same', activation='relu')(x)
# x = MaxPooling1D(pool_size=4)(x)
# x = BatchNormalization()(x)

# # Bloco Recorrente Bidirecional (Contexto temporal da fala)
# x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(x)
# x = BatchNormalization()(x)

# # Bloco de Foco Seletivo (Mecanismo de Atenção)
# x = AttentionLayer()(x)

# # Classificador Final
# x = Dense(64, activation='relu')(x)
# x = Dropout(0.4)(x)
# outputs = Dense(num_classes, activation='softmax')(x)

# model = Model(inputs, outputs, name="Hybrid_CNN_BiLSTM_Attention")
# model.compile(
#     optimizer='adam',
#     loss=CategoricalCrossentropy(label_smoothing=0.05),
#     metrics=['accuracy']
# )

# # ==========================================
# # 6. TREINAMENTO COLABORATIVO (FEDAVG)
# # ==========================================
# print(f"\n[Collaborative Learning] Treinando o Modelo Híbrido...")
# epochs_federadas = 35
# batch_size = 32

# history_loss, history_val_loss = [], []
# history_acc, history_val_acc = [], []

# for epoch in range(epochs_federadas):
#     chunks_X = np.array_split(X_train, 3)
#     chunks_y = np.array_split(y_train_cat, 3)
    
#     local_weights = []
#     for node in range(3):
#         model.fit(chunks_X[node], chunks_y[node], epochs=1, batch_size=batch_size, verbose=0)
#         local_weights.append(model.get_weights())
        
#     avg_weights = [np.mean([w[i] for w in local_weights], axis=0) for i in range(len(local_weights[0]))]
#     model.set_weights(avg_weights)
    
#     # Avaliação por época
#     train_scores = model.evaluate(X_train, y_train_cat, verbose=0)
#     val_scores = model.evaluate(X_val, y_val_cat, verbose=0)
    
#     history_loss.append(train_scores[0])
#     history_val_loss.append(val_scores[0])
#     history_acc.append(train_scores[1])
#     history_val_acc.append(val_scores[1])
    
#     print(f"Época Federada Global [{epoch+1}/{epochs_federadas}] -> Loss Val: {val_scores[0]:.4f} | Acc Val: {val_scores[1]*100:.2f}%")

# # ==========================================
# # 7. SALVAR MODELO EM ARQUIVO
# # ==========================================
# model.save("modelo_hibrido_com_atencao.h5")
# print("\n[SALVO] 'modelo_hibrido_com_atencao.h5' gravado com sucesso!")

# # ==========================================
# # 8. GERAÇÃO E SALVAMENTO DE VISUALIZAÇÕES
# # ==========================================
# # Gráfico 1: Curvas de Aprendizado (Loss e Acurácia)
# epochs_range = range(1, epochs_federadas + 1)
# plt.figure(figsize=(14, 5))

# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, history_acc, label='Acurácia Treino')
# plt.plot(epochs_range, history_val_acc, label='Acurácia Validação')
# plt.title('Histórico de Acurácia - Modelo Híbrido')
# plt.xlabel('Épocas Globais')
# plt.ylabel('Acurácia')
# plt.legend()
# plt.grid(True)

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, history_loss, label='Loss Treino')
# plt.plot(epochs_range, history_val_loss, label='Loss Validação')
# plt.title('Histórico de Erro (Loss) - Modelo Híbrido')
# plt.xlabel('Épocas Globais')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.savefig("hibrido_curvas_aprendizado.png", dpi=300)
# plt.show()
# print("[GRÁFICO SALVO] 'hibrido_curvas_aprendizado.png'")

# # Gráfico 2: Matriz de Confusão
# preds = model.predict(X_val)
# y_pred = np.argmax(preds, axis=1)

# cm = confusion_matrix(y_val, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='viridis')
# plt.title('Matriz de Confusão - Modelo Híbrido (CNN + BiLSTM + Atenção)')
# plt.xlabel('Predito')
# plt.ylabel('Real')
# plt.tight_layout()
# plt.savefig("hibrido_matriz_confusao.png", dpi=300)
# plt.show()
# print("[GRÁFICO SALVO] 'hibrido_matriz_confusao.png'")

# # Relatório impresso de métricas adicionais
# print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ===")
# print(classification_report(y_val, y_pred, target_names=labels))

import os
import json
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, BatchNormalization,
    LSTM, Dense, Dropout, Bidirectional
)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="Custom")
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = inputs * a
        return tf.keras.backend.sum(output, axis=1)

    def get_config(self):
        return super(AttentionLayer, self).get_config()

BASE_DIR = r"C:\Users\jefte\projetos em python\ufc 2025 a 2026\aprendizado de maquina\projeto de reconhecimento de voz AMRP"
AUDIO_ORIGINAL = os.path.join(BASE_DIR, "dataset_vozes_old")

SAMPLE_RATE = 16000
DURATION = 1.5
TOTAL_SAMPLES = int(SAMPLE_RATE * DURATION)

CLASSES = ["direita", "esquerda", "siga", "pare", "voltar"]
label_map = {classe: idx for idx, classe in enumerate(CLASSES)}

with open("labels_1s5_atencao.json", "w") as f:
    json.dump(label_map, f)

def carregar_e_processar_dataset():
    X = []
    y = []
    
    for raiz, diretorios, arquivos in os.walk(AUDIO_ORIGINAL):
        for arquivo in arquivos:
            if arquivo.lower().endswith((".wav", ".flac", ".ogg", ".m4a", ".mp3")):
                caminho_completo = os.path.join(raiz, arquivo)
                nome_classe = None
                
                for c in CLASSES:
                    if c in caminho_completo.lower() or c in arquivo.lower():
                        nome_classe = c
                        break
                        
                if nome_classe is None:
                    continue
                    
                try:
                    audio, sr = librosa.load(caminho_completo, sr=SAMPLE_RATE)
                    audio, _ = librosa.effects.trim(audio, top_db=25)
                    
                    if len(audio) > TOTAL_SAMPLES:
                        audio = audio[:TOTAL_SAMPLES]
                    else:
                        audio = np.pad(audio, (0, TOTAL_SAMPLES - len(audio)), 'constant')
                        
                    audio_int16 = (audio * 32767.0).astype(np.float32)
                    
                    if np.std(audio_int16) > 0:
                        audio_preprocessed = (audio_int16 - np.mean(audio_int16)) / np.std(audio_int16)
                    else:
                        audio_preprocessed = audio_int16
                        
                    X.append(audio_preprocessed)
                    y.append(label_map[nome_classe])
                except:
                    continue
                    
    return np.array(X), np.array(y)

X, y = carregar_e_processar_dataset()

X = np.expand_dims(X, axis=-1)
y = to_categorical(y, num_classes=len(CLASSES))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def criar_modelo_hibrido_1s5(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=4)(x)
    
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=4)(x)
    
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=4)(x)
    
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    
    x = AttentionLayer()(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.4)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

input_shape = (TOTAL_SAMPLES, 1)
model = criar_modelo_hibrido_1s5(input_shape, len(CLASSES))

model.compile(
    optimizer='adam',
    loss=CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32,
    callbacks=callbacks
)

model.save("modelo_hibrido_1s5_atencao.h5")