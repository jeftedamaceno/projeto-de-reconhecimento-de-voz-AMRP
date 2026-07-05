import os
import json
import wave
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, BatchNormalization,
    LSTM, Dense, Dropout, Bidirectional
)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


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
AUDIO_ORIGINAL = os.path.join(BASE_DIR, "dataset_final")

SAMPLE_RATE = 16000
DURATION = 1.5
TOTAL_SAMPLES = int(SAMPLE_RATE * DURATION)

CLASSES = ["direita", "esquerda", "siga", "pare", "voltar"]
label_map = {classe: idx for idx, classe in enumerate(CLASSES)}

with open("labels_1s5_atencao_sem_ruido.json", "w") as f:
    json.dump(label_map, f, indent=4)


def carregar_wav_manual(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        n_channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        sr = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        raw_data = wav_file.readframes(n_frames)
        
        if sampwidth == 2:
            audio_data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 1:
            audio_data = (np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError("Suporta apenas arquivos WAV de 8-bit ou 16-bit PCM.")
            
        if n_channels > 1:
            audio_data = audio_data.reshape(-1, n_channels).mean(axis=1)
            
        if sr != SAMPLE_RATE:
            num_provisorio = int(len(audio_data) * SAMPLE_RATE / sr)
            audio_data = np.interp(np.linspace(0, len(audio_data), num_provisorio), np.arange(len(audio_data)), audio_data)
            
        return audio_data

def trim_manual(audio, top_db=25, frame_length=2048, hop_length=512):
    if len(audio) == 0:
        return audio
        
    num_frames = 1 + int((len(audio) - frame_length) / hop_length)
    if num_frames <= 0:
        return audio
        
    rms = np.zeros(num_frames)
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = audio[start:end]
        rms[i] = np.sqrt(np.mean(frame**2) + 1e-10)
        
    rms_db = 20 * np.log10(rms / (np.max(rms) + 1e-10))
    intervalos_validos = np.where(rms_db > -top_db)[0]
    
    if len(intervalos_validos) == 0:
        return audio
        
    start_sample = intervalos_validos[0] * hop_length
    end_sample = min(len(audio), intervalos_validos[-1] * hop_length + frame_length)
    
    return audio[start_sample:end_sample]

def carregar_e_processar_dataset():
    X = []
    y = []
    
    for nome_classe in CLASSES:
        caminho_classe = os.path.join(AUDIO_ORIGINAL, nome_classe)
        
        if not os.path.isdir(caminho_classe):
            print(f"[AVISO] Pasta não encontrada para a classe: {caminho_classe}")
            continue
            
        print(f"Lendo arquivos da classe: '{nome_classe}'...")
        arquivos = os.listdir(caminho_classe)
        
        for arquivo in arquivos:
            if arquivo.lower().endswith(".wav"):
                caminho_completo = os.path.join(caminho_classe, arquivo)
                
                try:
                    audio = carregar_wav_manual(caminho_completo)
                    audio = trim_manual(audio, top_db=25)
                    
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
                    
                except Exception as e:
                    continue
                    
    return np.array(X), np.array(y)


X, y = carregar_e_processar_dataset()

if len(X) == 0:
    raise ValueError("Nenhum dado foi carregado. Verifique os caminhos e as pastas das classes.")

X = np.expand_dims(X, axis=-1)
y_cat = to_categorical(y, num_classes=len(CLASSES))

X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.3, random_state=42, stratify=y)


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

model.save("modelo_hibrido_1s5_atencao_sem_ruido.h5")


preds = model.predict(X_val)
y_pred_labels = np.argmax(preds, axis=1)
y_real_labels = np.argmax(y_val, axis=1)

print("\n" + "="*50)
print("CONFIANÇA MÉDIA POR CLASSE (SAÍDA SOFTMAX)")
print("="*50)
for classe, idx in label_map.items():
    indices_da_classe = np.where(y_real_labels == idx)[0]
    if len(indices_da_classe) > 0:
        confianca_media = np.mean(preds[indices_da_classe, idx]) * 100
        print(f"Classe: {classe:<10} -> Confiança Média do Modelo: {confianca_media:.2f}%")
print("="*50 + "\n")

print("=== RELATÓRIO DE CLASSIFICAÇÃO ===")
print(classification_report(y_real_labels, y_pred_labels, target_names=CLASSES))


plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Histórico de Acurácia')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Histórico de Erro (Loss)')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("hibrido_curvas_aprendizado_sem_ruido.png", dpi=300)
plt.show()

cm = confusion_matrix(y_real_labels, y_pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES, cmap='viridis')
plt.title('Matriz de Confusão - Sem Classe Ruído')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.tight_layout()
plt.savefig("hibrido_matriz_confusao_sem_ruido.png", dpi=300)
plt.show()

plt.figure(figsize=(9, 7))
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(CLASSES)):
    fpr[i], tpr[i], _ = roc_curve(y_val[:, i], preds[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f'Curva ROC de {CLASSES[i]} (Área = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', linestyle='--', color='red')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curva ROC Multiclasse (Um-Contra-Todos)')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("hibrido_curva_roc_sem_ruido.png", dpi=300)
plt.show()

print("\n[PROCESSO CONCLUÍDO] Dataset treinado sem a classe de ruído!")