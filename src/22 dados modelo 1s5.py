import os
import wave
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, BatchNormalization,
    Bidirectional, LSTM, Dense, Dropout
)

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
        return tf.keras.backend.sum(output, axis=1), a

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1], 1)]

    def get_config(self):
        return super(AttentionLayer, self).get_config()

OUTPUT_DIR = "visualizacoes_apresentacao"
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
    X, y = [], []
    for nome_classe in CLASSES:
        caminho_classe = os.path.join(AUDIO_ORIGINAL, nome_classe)
        if os.path.isdir(caminho_classe):
            for entrada_arquivo in os.scandir(caminho_classe):
                if entrada_arquivo.is_file() and entrada_arquivo.name.lower().endswith(".wav"):
                    try:
                        audio = carregar_wav_manual(entrada_arquivo.path)
                        if len(audio) > TOTAL_SAMPLES:
                            audio = audio[:TOTAL_SAMPLES]
                        else:
                            audio = np.pad(audio, (0, TOTAL_SAMPLES - len(audio)), 'constant')
                        audio_preprocessed = (audio - np.mean(audio)) / (np.std(audio) + 1e-8)
                        X.append(audio_preprocessed)
                        y.append(label_map[nome_classe])
                    except:
                        continue
    return np.array(X), np.array(y)

X, y = carregar_e_processar_dataset()
X = np.expand_dims(X, axis=-1)
y_cat = to_categorical(y, num_classes=len(CLASSES))

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.20, random_state=42, stratify=y)
y_test_numeric = np.argmax(y_test, axis=1)

def criar_modelo_treinavel(input_shape, num_classes):
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
    saida_atencao, pesos_atencao = AttentionLayer(name="atencao")(x)
    x = Dense(64, activation='relu')(saida_atencao)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax', name="classificacao")(x)
    return Model(inputs=inputs, outputs=[outputs, pesos_atencao])

modelo_treino = criar_modelo_treinavel((TOTAL_SAMPLES, 1), len(CLASSES))

modelo_treino.compile(
    optimizer='adam',
    loss={
        'classificacao': 'categorical_crossentropy',
        'atencao': None
    },
    metrics={'classificacao': ['accuracy']}
)

modelo_treino.fit(
    X_train, 
    {
        'classificacao': y_train,
        'atencao': np.zeros((len(X_train), 1))
    }, 
    epochs=5, 
    batch_size=32, 
    verbose=1
)

preds, pesos = modelo_treino.predict(X_test, verbose=0)
pesos_squeezed = np.squeeze(pesos, axis=-1)
y_pred_numeric = np.argmax(preds, axis=1)
confiancas = np.max(preds, axis=1)

acertos_idx = np.where(y_test_numeric == y_pred_numeric)[0]
erros_idx = np.where(y_test_numeric != y_pred_numeric)[0]

fig, axes = plt.subplots(len(CLASSES), 1, figsize=(12, 10), sharex=True)
for idx, nome_classe in enumerate(CLASSES):
    idx_acerto_classe = np.where((y_test_numeric == idx) & (y_pred_numeric == idx))[0]
    if len(idx_acerto_classe) > 0:
        escolhido = idx_acerto_classe[0]
        audio_bruto = X_test[escolhido, :, 0]
        pesos_audio = pesos_squeezed[escolhido]
        eixo_tempo_pesos = np.linspace(0, len(audio_bruto), len(pesos_audio))
        
        axes[idx].plot(audio_bruto, color='#95a5a6', alpha=0.4)
        axes[idx].set_ylabel("Amplitude", color='#7f8c8d')
        
        ax_twin = axes[idx].twinx()
        ax_twin.fill_between(eixo_tempo_pesos, pesos_audio, color='#2ecc71', alpha=0.3)
        ax_twin.plot(eixo_tempo_pesos, pesos_audio, color='#27ae60', lw=1.5)
        ax_twin.set_ylabel("Atenção", color='#27ae60')
        axes[idx].set_title(f"Classe: {nome_classe.upper()} (Predição Correta | Confiança: {confiancas[escolhido]*100:.1f}%)", fontsize=10, fontweight='bold', loc='left')
    else:
        axes[idx].text(0.5, 0.5, f"Sem amostra de acerto para a classe {nome_classe}", transform=axes[idx].transAxes, ha='center')

plt.xlabel('Amostras Temporais do Áudio (Linha do Tempo Real)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "v1_pesagem_parametros_acertos.png"), dpi=300)
plt.close()

df_analise = pd.DataFrame({
    'Confiança': confiancas,
    'Classe_Real_Idx': y_test_numeric,
    'Classe_Real': [CLASSES[i] for i in y_test_numeric],
    'Resultado': ['Acerto' if i in acertos_idx else 'Erro' for i in range(len(X_test))]
})

plt.figure(figsize=(10, 5))
cores = {'Acerto': '#2ecc71', 'Erro': '#e74c3c'}
sns.histplot(data=df_analise, x='Confiança', hue='Resultado', element='step', stat='count', palette=cores, alpha=0.6, multiple='stack')
plt.title("Validação de Performance: Distribuição Volumétrica de Certeza do Softmax", fontsize=12, fontweight='bold')
plt.xlabel("Grau de Confiança do Modelo (0.0 a 1.0)")
plt.ylabel("Quantidade Total de Áudios Analisados")
plt.grid(True, linestyle='--', alpha=0.4)
plt.xlim(0, 1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "v3_distribuicao_contagem_sucesso.png"), dpi=300)
plt.close()

plt.figure(figsize=(10, 5))
sns.kdeplot(data=df_analise, x='Confiança', hue='Resultado', palette=cores, fill=True, common_norm=False, alpha=0.4)
plt.title("Análise de Confiabilidade: Distribuição Correta de Incerteza do Modelo", fontsize=12, fontweight='bold')
plt.xlabel("Probabilidade da Classe Predita (Saída Softmax)")
plt.ylabel("Densidade de Ocorrências")
plt.xlim(0, 1.0)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "v3_densidade_confianca.png"), dpi=300)
plt.close()

fig, axes = plt.subplots(len(CLASSES), 1, figsize=(12, 12), sharex=True)
for idx, nome_classe in enumerate(CLASSES):
    df_classe = df_analise[df_analise['Classe_Real'] == nome_classe]
    if not df_classe.empty:
        sns.histplot(
            data=df_classe, 
            x='Confiança', 
            hue='Resultado', 
            element='step', 
            stat='count', 
            palette=cores, 
            alpha=0.6, 
            multiple='stack',
            ax=axes[idx],
            binwidth=0.05
        )
    axes[idx].set_title(f"Distribuição de Confiança - Classe: {nome_classe.upper()}", fontsize=10, fontweight='bold', loc='left')
    axes[idx].set_ylabel("Quantidade")
    axes[idx].grid(True, linestyle='--', alpha=0.3)
    axes[idx].set_xlim(0, 1.01)

plt.xlabel("Grau de Confiança do Modelo (0.0 a 1.0)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "v4_distribuicao_confianca_por_classe.png"), dpi=300)
plt.close()

print(f"Visualizações geradas com sucesso na pasta: '{OUTPUT_DIR}'")