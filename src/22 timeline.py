import os
import wave
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model

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

SAMPLE_RATE = 16000
DURATION = 1.5
TOTAL_SAMPLES = int(SAMPLE_RATE * DURATION)
DATASET_DIR = "dataset_final"
OUTPUT_DIR = "visualizacoes_timeline"
CLASSES = ["direita", "esquerda", "siga", "pare", "voltar"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = load_model("modelo_hibrido_1s5_atencao.h5", custom_objects={"AttentionLayer": AttentionLayer})

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

def processar_audio_individual(caminho_audio):
    audio = carregar_wav_manual(caminho_audio)
    
    if len(audio) < TOTAL_SAMPLES:
        audio_final = np.pad(audio, (0, int(TOTAL_SAMPLES - len(audio))), 'constant')
    else:
        audio_final = audio[:int(TOTAL_SAMPLES)]
        
    return audio_final

audios_selecionados = {}

print("Buscando exemplos e extraindo caracteristicas...")

for classe in CLASSES:
    pasta_classe = os.path.join(DATASET_DIR, classe)
    if not os.path.exists(pasta_classe):
        continue
        
    arquivos = [f for f in os.listdir(pasta_classe) if f.lower().endswith(".wav")]
    
    for arquivo in arquivos:
        caminho_completo = os.path.join(pasta_classe, arquivo)
        try:
            audio_bruto = processar_audio_individual(caminho_completo)
            
            if np.std(audio_bruto) > 0:
                audio_norm = (audio_bruto - np.mean(audio_bruto)) / (np.std(audio_bruto) + 1e-8)
            else:
                audio_norm = audio_bruto
                
            input_data = np.expand_dims(np.expand_dims(audio_norm, axis=0), axis=-1)
            
            preds = model.predict(input_data, verbose=0)[0]
            idx_predito = np.argmax(preds)
            classe_predita = CLASSES[idx_predito]
            confianca = preds[idx_predito]
            
            if classe_predita == classe and confianca >= 0.75:
                audios_selecionados[classe] = {
                    "dados_norm": audio_norm,
                    "arquivo": arquivo,
                    "confianca": confianca,
                    "input_tensor": input_data
                }
                print(f"Classe '{classe}': {arquivo} selecionado ({confianca*100:.2f}%)")
                break
        except Exception as e:
            continue

camada_cnn = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.MaxPooling1D)][-1]
camada_lstm = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Bidirectional)][0]
camada_atencao = [layer for layer in model.layers if "attention" in layer.name.lower() or "AttentionLayer" in str(type(layer))][0]

sub_model_cnn = Model(inputs=model.input, outputs=camada_cnn.output)
sub_model_lstm = Model(inputs=model.input, outputs=camada_lstm.output)
sub_model_att = Model(inputs=model.input, outputs=camada_atencao.output)

for classe, obj in audios_selecionados.items():
    tensor = obj["input_tensor"]
    
    features_cnn = sub_model_cnn.predict(tensor, verbose=0)
    features_lstm = sub_model_lstm.predict(tensor, verbose=0)
    features_att = sub_model_att.predict(tensor, verbose=0)
    previsao_final = model.predict(tensor, verbose=0)
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 14))
    fig.suptitle(f"Linha do Tempo de Mutacao 1D - Classe: {classe.upper()}\nArquivo original: {obj['arquivo']}", fontsize=14, fontweight='bold')
    
    axes[0].plot(obj["dados_norm"], color='royalblue', alpha=0.8)
    axes[0].set_title(f"1. Entrada Normalizada Z-score | Shape: {obj['dados_norm'].shape}", fontsize=11, loc='left')
    axes[0].grid(True, alpha=0.2)
    
    sinal_medio_cnn = np.mean(features_cnn[0], axis=-1)
    axes[1].plot(sinal_medio_cnn, color='crimson')
    axes[1].set_title(f"2. Saida do Bloco Convolucional (Media dos Canais) | Shape: {features_cnn.shape[1:]}", fontsize=11, loc='left')
    axes[1].grid(True, alpha=0.2)
    
    sinal_medio_lstm = np.mean(features_lstm[0], axis=-1)
    axes[2].plot(sinal_medio_lstm, color='darkorange')
    axes[2].set_title(f"3. Saida da Camada Bi-LSTM (Media dos Canais) | Shape: {features_lstm.shape[1:]}", fontsize=11, loc='left')
    axes[2].grid(True, alpha=0.2)
    
    axes[3].bar(range(len(features_att[0])), features_att[0], color='forestgreen', alpha=0.7)
    axes[3].set_title(f"4. Vetor Estatico Achatado da Atencao | Shape: {features_att.shape[1:]}", fontsize=11, loc='left')
    axes[3].grid(True, alpha=0.2)
    
    axes[4].bar(CLASSES, previsao_final[0], color='purple', alpha=0.8)
    axes[4].set_title(f"5. Softmax Final | Predito: {CLASSES[np.argmax(previsao_final[0])]} ({np.max(previsao_final[0])*100:.2f}%)", fontsize=11, loc='left')
    axes[4].set_ylim(0, 1.05)
    for idx, v in enumerate(previsao_final[0]):
        axes[4].text(idx, v + 0.02, f"{v*100:.1f}%", ha='center', fontweight='bold')
        
    plt.tight_layout()
    
    nome_arquivo_salvar = os.path.join(OUTPUT_DIR, f"timeline_{classe}.png")
    plt.savefig(nome_arquivo_salvar, dpi=300)
    plt.close()
    
    print(f"Grafico salvo em: {nome_arquivo_salvar}")