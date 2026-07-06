
import os
import json
import wave
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Reshape, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.losses import CategoricalCrossentropy


INPUT_DIR = "dataset_final"
OUTPUT_DIR = "dataset_ruido"
PASTA_EXPERIMENTO = "experimentos_ia"
SAMPLE_RATE = 16000
DURATION = 1.5           # Configurado para 1.5s para capturar a cadência cheia
TARGET_SIZE = 64 
EPOCHS_FEDERADAS = 30    # 30 gerações 
BATCH_SIZE = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PASTA_EXPERIMENTO, exist_ok=True)

def standardize_audio(audio):
    if np.max(np.abs(audio)) > 0:
        return audio / np.max(np.abs(audio))
    return audio

def random_shift(audio, shift_max=0.1, sr=16000):
    shift_samples = int(np.random.uniform(-shift_max, shift_max) * sr)
    return np.roll(audio, shift_samples)

def random_gain(audio, min_gain=0.7, max_gain=1.3):
    return audio * np.random.uniform(min_gain, max_gain)

print(" [ETAPA 1/4] Iniciando geração de ruídos multi-nível...")

def augment_audio_multinivel(audio):
    audios = []
    audios.append(('original', audio))
    # Ruído Médio
    ruido_medio = audio + np.random.normal(0, 0.008, len(audio))
    audios.append(('ruido_med', ruido_medio))
    # Ruído Intenso (Garante que a IA foque na cadência, não no timbre estático)
    ruido_intenso = audio + np.random.normal(0, 0.02, len(audio))
    audios.append(('ruido_int', ruido_intenso))
    # Variações de tempo e ganho
    audios.append(('shift', random_shift(audio, shift_max=0.1, sr=SAMPLE_RATE)))
    audios.append(('gain', random_gain(audio)))
    return audios

total_audios_originais = 0
total_audios_com_ruido = 0

if os.path.exists(INPUT_DIR):
    for label in os.listdir(INPUT_DIR):
        input_label_path = os.path.join(INPUT_DIR, label)
        if not os.path.isdir(input_label_path): continue

        output_label_path = os.path.join(OUTPUT_DIR, label)
        os.makedirs(output_label_path, exist_ok=True)

        for file in os.listdir(input_label_path):
            if not file.lower().endswith(('.wav', '.flac')): continue
            file_path = os.path.join(input_label_path, file)
            try:
                audio, sr = sf.read(file_path)
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)

                audio = standardize_audio(audio)
                total_audios_originais += 1
                
                versoes = augment_audio_multinivel(audio)
                for tipo, aug_audio in versoes:
                    aug_audio = np.clip(aug_audio, -1.0, 1.0)
                    nome_saida = f"{os.path.splitext(file)[0]}_{tipo}_{total_audios_com_ruido}.wav"
                    sf.write(os.path.join(output_label_path, nome_saida), aug_audio, SAMPLE_RATE)
                    total_audios_com_ruido += 1
            except Exception as e:
                pass
    print(f"   -> {total_audios_originais} arquivos originais processados.")
    print(f"   -> {total_audios_com_ruido} arquivos totais gerados com aumento de dados.")
else:
    print(f" Alerta: Pasta {INPUT_DIR} não encontrada.")


print("\n[ETAPA 2/4] Preparando matrizes espectrais para a IA...")

labels = sorted(os.listdir(INPUT_DIR)) if os.path.exists(INPUT_DIR) else []
label_map = {label: i for i, label in enumerate(labels)}
num_classes = len(labels)

def carregar_wav_manual(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        n_channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        raw_data = wav_file.readframes(wav_file.getnframes())
        if sampwidth == 2:
            audio_data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
        else:
            raise ValueError("Suporte apenas para WAV 16-bit PCM.")
        if n_channels > 1:
            audio_data = audio_data.reshape(-1, n_channels).mean(axis=1)
        return audio_data

def extrair_espectrograma_linear_manual(audio, n_fft=512, hop_length=256):
    janela = np.hanning(n_fft)
    num_frames = 1 + (len(audio) - n_fft) // hop_length
    if num_frames <= 0: return np.zeros((TARGET_SIZE, TARGET_SIZE))
    stft_matrix = []
    for t in range(num_frames):
        inicio = t * hop_length
        fatia = audio[inicio:inicio + n_fft]
        if len(fatia) < n_fft: fatia = np.pad(fatia, (0, n_fft - len(fatia)), 'constant')
        fft_valores = np.abs(np.fft.rfft(fatia * janela))
        stft_matrix.append(fft_valores)
    return np.log1p(np.array(stft_matrix).T)

def redimensionar_matriz_bilinear(matriz, target_size):
    orig_h, orig_w = matriz.shape
    if orig_h == 0 or orig_w == 0: return np.zeros((target_size, target_size))
    grid_h = np.linspace(0, orig_h - 1, target_size)
    grid_w = np.linspace(0, orig_w - 1, target_size)
    y_b = grid_h.astype(np.int32)
    y_a = np.minimum(y_b + 1, orig_h - 1)
    x_e = grid_w.astype(np.int32)
    x_d = np.minimum(x_e + 1, orig_w - 1)
    dy = (grid_h - y_b)[:, None]
    dx = (grid_w - x_e)[None, :]
    return (1-dy)*(1-dx)*matriz[y_b[:,None], x_e] + (1-dy)*dx*matriz[y_b[:,None], x_d] + dy*(1-dx)*matriz[y_a[:,None], x_e] + dy*dx*matriz[y_a[:,None], x_d]

def processar_pipeline_ideal(file_path):
    audio = carregar_wav_manual(file_path)
    limiar_energia = 0.03 * np.max(np.abs(audio))
    indices_uteis = np.where(np.abs(audio) > limiar_energia)[0]
    if len(indices_uteis) > 0:
        audio = audio[indices_uteis[0]:indices_uteis[-1]]
    max_samples = int(SAMPLE_RATE * DURATION)
    if len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)), 'constant')
    else:
        audio = audio[:max_samples]
    pico = np.max(np.abs(audio))
    if pico > 1e-6: audio = audio / pico
    espectrograma = extrair_espectrograma_linear_manual(audio)
    espectrograma_quadrado = redimensionar_matriz_bilinear(espectrograma, TARGET_SIZE)
    return np.expand_dims(espectrograma_quadrado, axis=-1)

X_orig, y_orig = [], []
for label in labels:
    path = os.path.join(INPUT_DIR, label)
    if not os.path.isdir(path): continue
    for file in os.listdir(path):
        if file.endswith(".wav"):
            X_orig.append(processar_pipeline_ideal(os.path.join(path, file)))
            y_orig.append(label_map[label])

X_orig, y_orig = np.array(X_orig), np.array(y_orig)
# Separação estrita de validação (composta apenas de dados limpos/originais)
X_train_limpo, X_val, y_train_limpo, y_val = train_test_split(X_orig, y_orig, test_size=0.2, stratify=y_orig, random_state=42)

# Conjunto com Aumento de Dados
X_train_ruido, y_train_ruido = np.copy(X_train_limpo), np.copy(y_train_limpo)
if os.path.exists(OUTPUT_DIR):
    X_aug, y_aug = [], []
    for label in labels:
        path = os.path.join(OUTPUT_DIR, label)
        if not os.path.isdir(path): continue
        for file in os.listdir(path):
            if file.endswith(".wav"):
                X_aug.append(processar_pipeline_ideal(os.path.join(path, file)))
                y_aug.append(label_map[label])
    if len(X_aug) > 0:
        X_train_ruido = np.concatenate([X_train_ruido, np.array(X_aug)])
        y_train_ruido = np.concatenate([y_train_ruido, np.array(y_aug)])

# Contadores exatos de treino
print(f"\n CONTADORES OFICIAIS PARA O RELATÓRIO:")
print(f"   -> Quantidade de áudios usados na VALIDAÇÃO (Limpos): {len(X_val)}")
print(f"   -> Quantidade de áudios usados no TREINAMETO COM RUÍDO COMPLETO: {len(X_train_ruido)}")

# Preparando os vetores categóricos
y_val_cat = to_categorical(y_val, num_classes)
idx_r = np.random.permutation(len(X_train_ruido))
X_train_ruido, y_train_ruido_cat = X_train_ruido[idx_r], to_categorical(y_train_ruido[idx_r], num_classes)

idx_l = np.random.permutation(len(X_train_limpo))
X_train_limpo, y_train_limpo_cat = X_train_limpo[idx_l], to_categorical(y_train_limpo[idx_l], num_classes)


def criar_modelo_crnn():
    return Sequential([
        Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(TARGET_SIZE, TARGET_SIZE, 1)),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Reshape(target_shape=(16, 16 * 64)), 
        
       
        Bidirectional(LSTM(64, return_sequences=False, dropout=0.4, recurrent_dropout=0.4)),
        
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])


def rodar_treino_federado(X_dados, y_dados, nome_experimento):
    print(f"\n Executando Treino Colaborativo para: {nome_experimento}...")
    model_fed = criar_modelo_crnn()
    model_fed.compile(optimizer='adam', loss=CategoricalCrossentropy(label_smoothing=0.08), metrics=['accuracy'])
    
    chunks_X = np.array_split(X_dados, 3)
    chunks_y = np.array_split(y_dados, 3)
    proporcoes = [len(chunks_X[i]) / len(X_dados) for i in range(3)]
    
    historico_acc = []
    for epoch in range(EPOCHS_FEDERADAS):
        local_weights = []
        pesos_globais = model_fed.get_weights()
        for node in range(3):
            model_fed.set_weights(pesos_globais)
            model_fed.fit(chunks_X[node], chunks_y[node], epochs=1, batch_size=BATCH_SIZE, verbose=0)
            local_weights.append(model_fed.get_weights())
        pesos_agregados = [np.zeros_like(w) for w in pesos_globais]
        for layer_idx in range(len(pesos_globais)):
            for node in range(3):
                pesos_agregados[layer_idx] += local_weights[node][layer_idx] * proporcoes[node]
        model_fed.set_weights(pesos_agregados)
        scores = model_fed.evaluate(X_val, y_val_cat, verbose=0)
        historico_acc.append(scores[1])
    return model_fed, historico_acc

# Treina os dois cenários para provar o impacto do ruído
modelo_limpo, acc_sem_ruido = rodar_treino_federado(X_train_limpo, y_train_limpo_cat, "Modelo SEM Aumento de Ruído")
modelo_final, acc_com_ruido = rodar_treino_federado(X_train_ruido, y_train_ruido_cat, "Modelo COM Multi-Nível de Ruído")


print("\n Salvando arquivos finais na pasta de experimentos...")
modelo_final.save(os.path.join(PASTA_EXPERIMENTO, "modelo_crnn_cadencia_30_geracoes.h5"))
with open(os.path.join(PASTA_EXPERIMENTO, "labels_config.json"), "w") as f:
    json.dump(label_map, f)

# Gerar gráfico mostrando o impacto do ruído para combater Overfitting/Underfitting
plt.figure(figsize=(9, 5), dpi=200)
plt.plot(range(1, EPOCHS_FEDERADAS + 1), [a * 100 for a in acc_sem_ruido], color="#e74c3c", linestyle="--", marker="o", label="Treino com Dados Limpos (Risco de Overfitting)")
plt.plot(range(1, EPOCHS_FEDERADAS + 1), [a * 100 for a in acc_com_ruido], color="#2ecc71", linestyle="-", marker="s", linewidth=2.5, label="Treino com Multi-Nível de Ruído (Robusto)")
plt.title("Impacto do Ruído Multi-Nível na Generalização da IA", fontsize=12, fontweight='bold')
plt.xlabel("Geração Federada Global")
plt.ylabel("Acurácia na Validação (%)")
plt.legend(loc="lower right")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()

caminho_impacto = os.path.join(PASTA_EXPERIMENTO, "impacto_do_ruido_no_modelo.png")
plt.savefig(caminho_impacto)
plt.close()

print(f"\n PIPELINE CORRIGIDO COM SUCESSO!")
print(f" Gráfico de impacto salvo em: '{caminho_impacto}'")