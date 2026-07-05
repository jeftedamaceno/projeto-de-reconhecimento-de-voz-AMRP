import os
import json
import wave
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_recall_fscore_support
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

def gerar_ruido_rosa(shape):
    ruido_branco = np.random.normal(0, 1, shape)
    valores_fft = np.fft.rfft(ruido_branco, axis=1)
    frequencias = np.fft.rfftfreq(shape[1])
    frequencias[0] = frequencias[1]
    
    filtro = 1.0 / np.sqrt(frequencias)
    filtro = filtro / np.max(filtro)
    filtro = filtro.reshape(1, -1, 1)
    
    valores_fft_filtrados = valores_fft * filtro
    ruido_rosa = np.fft.irfft(valores_fft_filtrados, n=shape[1], axis=1)
    
    _mean = np.mean(ruido_rosa, axis=1, keepdims=True)
    _std = np.std(ruido_rosa, axis=1, keepdims=True) + 1e-8
    ruido_rosa = (ruido_rosa - _mean) / _std
    return ruido_rosa

def aplicar_cortes_audio(X, num_cortes=5, tamanho_corte=400):
    X_cortado = X.copy()
    for i in range(X_cortado.shape[0]):
        for _ in range(num_cortes):
            inicio = np.random.randint(0, X_cortado.shape[1] - tamanho_corte)
            X_cortado[i, inicio:inicio+tamanho_corte, 0] = 0.0
    return X_cortado

def aplicar_deslocamento_temporal(X, max_shift=1600):
    X_deslocado = np.zeros_like(X)
    for i in range(X.shape[0]):
        shift = np.random.randint(-max_shift, max_shift)
        if shift > 0:
            X_deslocado[i, shift:, 0] = X[i, :-shift, 0]
        elif shift < 0:
            X_deslocado[i, :shift, 0] = X[i, -shift:, 0]
        else:
            X_deslocado[i, :, 0] = X[i, :, 0]
    return X_deslocado

OUTPUT_DIR = "pasta_de_teste_ruido"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_DIR = r"C:\Users\jefte\projetos em python\ufc 2025 a 2026\aprendizado de maquina\projeto de reconhecimento de voz AMRP"
AUDIO_ORIGINAL = os.path.join(BASE_DIR, "dataset_final")

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
    
    if not os.path.isdir(AUDIO_ORIGINAL):
        raise ValueError(f"Diretorio raiz nao encontrado: {AUDIO_ORIGINAL}")
        
    for nome_classe in CLASSES:
        caminho_classe = os.path.join(AUDIO_ORIGINAL, nome_classe)
        if os.path.isdir(caminho_classe):
            for entrada_arquivo in os.scandir(caminho_classe):
                if entrada_arquivo.is_file() and entrada_arquivo.name.lower().endswith(".wav"):
                    caminho_completo = entrada_arquivo.path
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

X_train, X_temp, y_train, y_temp = train_test_split(X, y_cat, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=np.argmax(y_temp, axis=1))
y_test_numeric = np.argmax(y_test, axis=1)

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
model.compile(optimizer='adam', loss=CategoricalCrossentropy(label_smoothing=0.05), metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32, callbacks=callbacks, verbose=1)

model.save(os.path.join(OUTPUT_DIR, "modelo_teste_ruido.h5"))

cenarios_teste = {
    "limpo": X_test,
    "ruido_gaussiano": X_test + np.random.normal(0, 0.15, X_test.shape),
    "ruido_rosa": X_test + (gerar_ruido_rosa(X_test.shape) * 0.15),
    "cortes_audio": aplicar_cortes_audio(X_test, num_cortes=6, tamanho_corte=500),
    "deslocamento_temporal": aplicar_deslocamento_temporal(X_test, max_shift=2400)
}

preds_por_cenario = {}
metricas_cenarios = []

for nome_cenario, dados_teste in cenarios_teste.items():
    preds = model.predict(dados_teste, verbose=0)
    preds_por_cenario[nome_cenario] = preds
    y_pred_numeric = np.argmax(preds, axis=1)
    
    cm = confusion_matrix(y_test_numeric, y_pred_numeric)
    acc = accuracy_score(y_test_numeric, y_pred_numeric)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test_numeric, y_pred_numeric, average='macro', zero_division=0)
    metricas_cenarios.append([nome_cenario, acc, prec, rec, f1])
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES, cmap='plasma')
    plt.title(f"Matriz de Confusao - {nome_cenario.upper()}")
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"matriz_{nome_cenario}.png"), dpi=300)
    plt.close()

df_cenarios = pd.DataFrame(metricas_cenarios, columns=["Cenario", "Accuracy", "Precision", "Recall", "F1"])
df_cenarios.to_csv(os.path.join(OUTPUT_DIR, "metricas_cenarios_isolados.csv"), index=False)

plt.figure(figsize=(10, 5))
sns.barplot(x="Cenario", y="Accuracy", data=df_cenarios, palette="viridis")
plt.title("Comparativo de Acuracia do Modelo por Tipo de Degradacao Isolada")
plt.ylabel("Acuracia")
plt.xlabel("Cenario de Teste")
plt.ylim(0, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparativo_acuracia_cenarios.png"), dpi=300)
plt.close()

plt.figure(figsize=(10, 5))
for nome_cenario, preds in preds_por_cenario.items():
    confiancas = np.max(preds, axis=1)
    sns.kdeplot(confiancas, label=nome_cenario.upper(), fill=True, alpha=0.2)
plt.title("Impacto das Degradacoes Isoladas na Confianca do Softmax")
plt.xlabel("Probabilidade Maxima (Softmax)")
plt.ylabel("Densidade")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "distribuicao_confianca_cenarios.png"), dpi=300)
plt.close()

# =====================================================================
# GERAÇÃO DA CURVA ROC COMPARATIVA ENTRE CENÁRIOS (MÉDIA MACRO)
# =====================================================================
plt.figure(figsize=(10, 8))
cores_roc = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6", "#f1c40f"]

for idx, (nome_cenario, preds) in enumerate(preds_por_cenario.items()):
    fpr_macro = []
    tpr_macro = []
    
    todas_fpras = np.unique(np.concatenate([roc_curve(y_test[:, i], preds[:, i])[0] for i in range(len(CLASSES))]))
    interp_tpras = np.zeros_like(todas_fpras)
    
    for i in range(len(CLASSES)):
        fpr, tpr, _ = roc_curve(y_test[:, i], preds[:, i])
        interp_tpras += np.interp(todas_fpras, fpr, tpr)
        
    interp_tpras /= len(CLASSES)
    area_auc = auc(todas_fpras, interp_tpras)
    
    plt.plot(todas_fpras, interp_tpras, color=cores_roc[idx], lw=2.5,
             label=f'{nome_cenario.upper()} (AUC Médio = {area_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, color='grey')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curva ROC Comparativa: Impacto de Cenários Isolados no Modelo', fontsize=12, fontweight='bold')
plt.legend(loc="lower right", frameon=True)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparativo_curvas_roc_cenarios.png"), dpi=300)
plt.close()