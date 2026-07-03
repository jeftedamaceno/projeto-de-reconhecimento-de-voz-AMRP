import os
import wave
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

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
    X, y = [], []
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
                    except:
                        continue
    return np.array(X), np.array(y)

X, y = carregar_e_processar_dataset()
X = np.expand_dims(X, axis=-1)

model = load_model("modelo_hibrido_1s5_atencao.h5", custom_objects={"AttentionLayer": AttentionLayer})

camada_cnn = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.MaxPooling1D)][-1]
camada_lstm = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Bidirectional)][0]
camada_atencao = [layer for layer in model.layers if "attention" in layer.name.lower() or "AttentionLayer" in str(type(layer))][0]

extrator_cnn = Model(inputs=model.input, outputs=camada_cnn.output)
extrator_lstm = Model(inputs=model.input, outputs=camada_lstm.output)
extrator_atencao = Model(inputs=model.input, outputs=camada_atencao.output)

feat_cnn = extrator_cnn.predict(X, verbose=0)
feat_lstm = extrator_lstm.predict(X, verbose=0)
feat_atencao = extrator_atencao.predict(X, verbose=0)

feat_cnn_flat = np.mean(feat_cnn, axis=-1)
feat_lstm_flat = np.mean(feat_lstm, axis=-1)

pca = PCA(n_components=2, random_state=42)

fig, axes = plt.subplots(1, 3, figsize=(22, 6))

data_layers = [
    ("PCA - Saida Convolucional (CNN)", feat_cnn_flat),
    ("PCA - Saida Temporal (Bi-LSTM)", feat_lstm_flat),
    ("PCA - Vetor de Atencao (Final)", feat_atencao)
]

for idx, (titulo, dados) in enumerate(data_layers):
    dados_projetados = pca.fit_transform(dados)
    ax = axes[idx]
    
    for c_idx, classe in enumerate(CLASSES):
        indices = np.where(y == c_idx)
        ax.scatter(dados_projetados[indices, 0], dados_projetados[indices, 1], label=classe.upper(), alpha=0.6, edgecolors='k', s=35)
        
    ax.set_title(titulo, fontweight='bold')
    ax.grid(True, alpha=0.2)
    if idx == 0:
        ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparativo_evolucao_pca.png"), dpi=300)
plt.close()

print("Grafico comparativo salvo com sucesso na pasta de testes.")