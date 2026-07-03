import os
import json
import wave
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, BatchNormalization,
    Bidirectional, LSTM, Dense, Dropout
)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping

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

OUTPUT_DIR = "pasta_de_teste_ruido"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_DIR = r"C:\Users\jefte\projetos em python\ufc 2025 a 2026\aprendizado de maquina\projeto de reconhecimento de voz AMRP"
AUDIO_ORIGINAL = os.path.join(BASE_DIR, "dataset_vozes_old")

SAMPLE_RATE = 16000
DURATION = 1.5
TOTAL_SAMPLES = int(SAMPLE_RATE * DURATION)
CLASSES = ["direita", "esquerda", "siga", "pare", "voltar"]
label_map = {classe: idx for idx, classe in enumerate(CLASSES)}

def carregar_wav_manual(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        n_channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        n_frames = wav_file.getnframes()
        raw_data = wav_file.readframes(n_frames)
        if sampwidth == 2:
            audio_data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
        else:
            raise ValueError("Suporta apenas arquivos WAV de 16-bit PCM.")
        if n_channels > 1:
            audio_data = audio_data.reshape(-1, n_channels).mean(axis=1)
        return audio_data

def carregar_e_processar_dataset():
    X = []
    y = []
    contagem_carregados = {c: 0 for c in CLASSES}
    
    for raiz, diretorios, arquivos in os.walk(AUDIO_ORIGINAL):
        pasta_atual = os.path.basename(raiz).lower()
        if pasta_atual in label_map:
            nome_classe = pasta_atual
            for arquivo in arquivos:
                if arquivo.lower().endswith(".wav"):
                    caminho_completo = os.path.join(raiz, arquivo)
                    try:
                        audio = carregar_wav_manual(caminho_completo)
                        if len(audio) > TOTAL_SAMPLES:
                            audio = audio[:TOTAL_SAMPLES]
                        else:
                            audio = np.pad(audio, (0, TOTAL_SAMPLES - len(audio)), 'constant')
                        
                        if np.std(audio) > 0:
                            audio_preprocessed = (audio - np.mean(audio)) / np.std(audio)
                        else:
                            audio_preprocessed = audio
                        X.append(audio_preprocessed)
                        y.append(label_map[nome_classe])
                        contagem_carregados[nome_classe] += 1
                    except:
                        continue
                        
    print("--- TOTAL DE AUDIOS ENCONTRADOS E CARREGADOS POR CLASSE ---")
    for c, total in contagem_carregados.items():
        print(f"Classe {c.upper()}: {total} audios carregados com sucesso.")
    print("---------------------------------------------------------")
    
    return np.array(X), np.array(y)

X, y = carregar_e_processar_dataset()

if len(X) == 0:
    raise ValueError("Nenhum arquivo de audio foi carregado. Verifique os caminhos das pastas.")

X = np.expand_dims(X, axis=-1)
y_cat = to_categorical(y, num_classes=len(CLASSES))

# Divisão 70/15/15
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_cat, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=np.argmax(y_temp, axis=1)
)
y_val_numeric = np.argmax(y_val, axis=1)
y_test_numeric = np.argmax(y_test, axis=1)

print("\n--- DIVISAO DOS MODELOS DE TESTE E TREINO ---")
print(f"Total Geral de Amostras para Treino: {X_train.shape[0]}")
print(f"Total Geral de Amostras para Validacao/Teste: {X_test.shape[0]}")
for c in CLASSES:
    idx_classe = label_map[c]
    total_treino_c = np.sum(np.argmax(y_train, axis=1) == idx_classe)
    total_val_c = np.sum(y_val_numeric == idx_classe)
    print(f"Classe {c.upper()} -> Treino: {total_treino_c} | Validacao (Amostras na Matriz): {total_val_c}")
print("---------------------------------------------\n")

def criar_modelo_hibrido(input_shape, num_classes):
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
    return Model(inputs=inputs, outputs=outputs)

model = criar_modelo_hibrido((TOTAL_SAMPLES, 1), len(CLASSES))
model.compile(
    optimizer='adam',
    loss=CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)

callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=callbacks
)

model.save(os.path.join(OUTPUT_DIR, "modelo_hibrido_voz_estavel.h5"))

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validacao')
plt.title('Historico de Acuracia')
plt.xlabel('Epoca')
plt.ylabel('Acuracia')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validacao')
plt.title('Historico de Erro Loss')
plt.xlabel('Epoca')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "grafico_acuracia_x_loss.png"), dpi=300)
plt.close()

ruidos = {
    "ruido_baixo": 0.05,
    "ruido_medio": 0.15,
    "ruido_intenso": 0.30
}
preds_por_nivel = {}

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
metricas=[]


for nome_ruido, desvio in ruidos.items():
    noise = np.random.normal(0, desvio, X_test.shape)
    X_val_ruidoso = X_test + noise
    
    preds = model.predict(X_val_ruidoso, verbose=0)
    preds_por_nivel[nome_ruido] = preds
    y_pred_numeric = np.argmax(preds, axis=1)
    
    cm = confusion_matrix(y_test_numeric, y_pred_numeric)
    acc = accuracy_score(y_test_numeric, y_pred_numeric)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test_numeric, y_pred_numeric, average='macro', zero_division=0)
    metricas.append([nome_ruido, desvio, acc, prec, rec, f1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES, cmap='viridis')
    plt.title(f"Matriz de Confusao - {nome_ruido.upper()}")
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"matriz_confusao_{nome_ruido}.png"), dpi=300)
    plt.close()

plt.figure(figsize=(18, 5))
colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red']
posicoes = [131, 132, 133]

for idx, (nome_ruido, preds) in enumerate(preds_por_nivel.items()):
    plt.subplot(posicoes[idx])
    for i in range(len(CLASSES)):
        fpr, tpr, _ = roc_curve(y_test[:, i], preds[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{CLASSES[i]} ({roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Falsos Positivos')
    plt.ylabel('Verdadeiros Positivos')
    plt.title(f'ROC - {nome_ruido.upper()}')
    plt.legend(loc="lower right", fontsize='small')
    plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparativo_curvas_roc.png"), dpi=300)
plt.close()

plt.figure(figsize=(10, 5))
for nome_ruido, preds in preds_por_nivel.items():
    confiancas = np.max(preds, axis=1)
    sns.kdeplot(confiancas, label=nome_ruido.upper(), fill=True, alpha=0.3)

plt.title("Impacto do Ruido na Distribuicao de Confianca do Softmax")
plt.xlabel("Probabilidade da Classe Predita")
plt.ylabel("Densidade")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "distribuicao_confianca_ruidos.png"), dpi=300)
plt.close()

import pandas as pd
df_metricas = pd.DataFrame(metricas, columns=[
    "Nivel_Ruido","Sigma","Accuracy","Precision","Recall","F1"
])
df_metricas.to_csv(os.path.join(OUTPUT_DIR,"metricas_ruido.csv"),index=False)

plt.figure(figsize=(8,5))
plt.plot(df_metricas["Sigma"], df_metricas["Accuracy"], marker='o')
plt.grid(True)
plt.xlabel("Sigma")
plt.ylabel("Accuracy")
plt.title("Degradação da acurácia")
plt.savefig(os.path.join(OUTPUT_DIR,"degradacao_acuracia.png"), dpi=300)
plt.close()
