import os
import wave
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
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
CAMINHO_MODELO = os.path.join(BASE_DIR, "modelo_hibrido_1s5_atencao.h5")

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
                    except:
                        continue
    return np.array(X), np.array(y)

X, y = carregar_e_processar_dataset()
X = np.expand_dims(X, axis=-1)
y_cat = to_categorical(y, num_classes=len(CLASSES))

_, X_temp, _, y_temp = train_test_split(X, y_cat, test_size=0.30, random_state=42, stratify=y)
_, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=np.argmax(y_temp, axis=1))
y_test_numeric = np.argmax(y_test, axis=1)

def recriar_estrutura_modelo(input_shape, num_classes):
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
    saida_atencao, pesos_atencao = AttentionLayer()(x)
    x = Dense(64, activation='relu')(saida_atencao)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=[outputs, pesos_atencao])

print(f"Instanciando extrator e injetando pesos originais de: {CAMINHO_MODELO}")
extrator_pesos_model = recriar_estrutura_modelo((TOTAL_SAMPLES, 1), len(CLASSES))
extrator_pesos_model.load_weights(CAMINHO_MODELO, by_name=True, skip_mismatch=True)

cenarios_teste = {
    "limpo": X_test,
    "ruido_gaussiano": X_test + np.random.normal(0, 0.15, X_test.shape),
    "ruido_rosa": X_test + (gerar_ruido_rosa(X_test.shape) * 0.15),
    "cortes_audio": aplicar_cortes_audio(X_test, num_cortes=6, tamanho_corte=500),
    "deslocamento_temporal": aplicar_deslocamento_temporal(X_test, max_shift=2400)
}

preds_por_cenario = {}
pesos_por_cenario = {}
metricas_cenarios = []

for nome_cenario, dados_teste in cenarios_teste.items():
    preds, pesos = extrator_pesos_model.predict(dados_teste, verbose=0)
    preds_por_cenario[nome_cenario] = preds
    pesos_por_cenario[nome_cenario] = np.squeeze(pesos, axis=-1)
    
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

dados_estatisticos_pesos = []

for nome_cenario in cenarios_teste.keys():
    pesos_cenario = pesos_por_cenario[nome_cenario]
    preds_cenario = np.argmax(preds_por_cenario[nome_cenario], axis=1)
    
    plt.figure(figsize=(12, 6))
    
    for idx_classe, nome_classe in enumerate(CLASSES):
        indices_classe = np.where((y_test_numeric == idx_classe) & (preds_cenario == idx_classe))[0]
        
        if len(indices_classe) > 0:
            pesos_filtrados = pesos_cenario[indices_classe]
            pesos_medios_temporais = np.mean(pesos_filtrados, axis=0)
            
            media_global = np.mean(pesos_filtrados)
            mediana_global = np.median(pesos_filtrados)
            std_global = np.std(pesos_filtrados)
            
            dados_estatisticos_pesos.append({
                "Cenario": nome_cenario,
                "Classe": nome_classe,
                "Status_Predicao": "Correto",
                "Media_Pesos": media_global,
                "Mediana_Pesos": mediana_global,
                "Desvio_Padrao_Pesos": std_global
            })
            
            plt.plot(pesos_medios_temporais, label=f"{nome_classe.upper()} (Acertos)", lw=2)
            
        indices_erros = np.where((y_test_numeric == idx_classe) & (preds_cenario != idx_classe))[0]
        if len(indices_erros) > 0:
            pesos_erros = pesos_cenario[indices_erros]
            media_gl_err = np.mean(pesos_erros)
            mediana_gl_err = np.median(pesos_erros)
            std_gl_err = np.std(pesos_erros)
            
            dados_estatisticos_pesos.append({
                "Cenario": nome_cenario,
                "Classe": nome_classe,
                "Status_Predicao": "Incorreto",
                "Media_Pesos": media_gl_err,
                "Mediana_Pesos": mediana_gl_err,
                "Desvio_Padrao_Pesos": std_gl_err
            })

    plt.title(f"Distribuicao Temporal Media dos Pesos de Atencao - {nome_cenario.upper()}")
    plt.xlabel("Indices de Frames Reduzidos (Eixo Temporal do Modelo)")
    plt.ylabel("Peso de Importancia (Attention Score)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"distribuicao_pesos_atencao_{nome_cenario}.png"), dpi=300)
    plt.close()

df_pesos_stats = pd.DataFrame(dados_estatisticos_pesos)
df_pesos_stats.to_csv(os.path.join(OUTPUT_DIR, "estatisticas_atributos_pesos_classe.csv"), index=False)

df_resumo_cenario = df_pesos_stats.groupby("Cenario").agg({"Desvio_Padrao_Pesos": "mean"}).reset_index()
df_resumo_cenario = df_resumo_cenario.merge(df_cenarios[["Cenario", "Accuracy"]], on="Cenario")

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_resumo_cenario, x="Desvio_Padrao_Pesos", y="Accuracy", hue="Cenario", s=150, palette="deep")
plt.title("Como a Variabilidade dos Pesos Influencia a Acuracia Global")
plt.xlabel("Desvio Padrao Medio dos Pesos da Atencao (Foco Seletivo)")
plt.ylabel("Acuracia Global")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "influencia_pesos_na_acuracia.png"), dpi=300)
plt.close()

print(f"Estudo concluido com sucesso. Os relatorios e graficos foram salvos em: {OUTPUT_DIR}")