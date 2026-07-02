import os
import json
import wave
import numpy as np
import sounddevice as sd
from tensorflow.keras.models import load_model
import keyboard

# --- CONFIGURAÇÕES DO EXPERIMENTO NOVO ---
PASTA_EXPERIMENTO = "experimentos_ia"
# MODEL_PATH = os.path.join(PASTA_EXPERIMENTO, "modelo_crnn_cadencia_30_geracoes.h5")
# MODEL_PATH = os.path.join(PASTA_EXPERIMENTO, "modelo_crnn_cadencia_30_geracoes_heterogeneo_validado.h5")
# LABELS_PATH = os.path.join(PASTA_EXPERIMENTO, "labels_config.json")
MODEL_PATH = os.path.join(PASTA_EXPERIMENTO, "modelo kj.h5")
LABELS_PATH = os.path.join(PASTA_EXPERIMENTO, "labels_config.json")

SAMPLE_RATE = 16000
DURATION = 1.25         # Atualizado para 1.5s igual ao treino da CRNN
TARGET_SIZE = 64       # Tamanho da matriz quadrada (64x64)

THRESHOLD = 0.55       # Limiar mínimo de confiança
ENTROPY_LIMIT = 1.3    # Limiar máximo de incerteza (entropia)

# =====================================================================
# 1. FUNÇÕES MATEMÁTICAS MANUAIS (IGUAIS ÀS DO PIPELINE DE TREINO)
# =====================================================================
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

def processar_audio_microfone(audio):
    # Truncamento de silêncio inicial/final por energia (VAD manual)
    limiar_energia = 0.03 * np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else 0
    indices_uteis = np.where(np.abs(audio) > limiar_energia)[0]
    if len(indices_uteis) > 0:
        audio = audio[indices_uteis[0]:indices_uteis[-1]]
        
    # Garante tamanho exato de 1.5s
    max_samples = int(SAMPLE_RATE * DURATION)
    if len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)), 'constant')
    else:
        audio = audio[:max_samples]
        
    # Normalização de ganho de pico
    pico = np.max(np.abs(audio))
    if pico > 1e-6: audio = audio / pico
        
    # Gera a matriz 64x64 idêntica à que a CRNN espera
    espectrograma = extrair_espectrograma_linear_manual(audio)
    espectrograma_quadrado = redimensionar_matriz_bilinear(espectrograma, TARGET_SIZE)
    return espectrograma_quadrado

def calcular_entropia(probs):
    probs = probs + 1e-10
    return -np.sum(probs * np.log(probs))

# =====================================================================
# 2. CARREGAMENTO DO MODELO E LABELS
# =====================================================================
print("🔄 Carregando modelo CRNN e configurações...")
if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    raise FileNotFoundError("Modelo ou Labels não encontrados na pasta 'experimentos_ia'. Execute o treino primeiro.")

model = load_model(MODEL_PATH)
with open(LABELS_PATH, "r") as f:
    label_map = json.load(f)

inv_map = {v: k for k, v in label_map.items()}

# =====================================================================
# 3. PIPELINE DE GRAVAÇÃO E CLASSIFICAÇÃO
# =====================================================================
def gravar_audio():
    # Captura em 16kHz Mono
    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1,
                   dtype='float32')
    sd.wait()
    return audio.flatten()

def classificar():
    print("\n🎤 Ouvindo... Fale agora!")
    audio = gravar_audio()
    print("🧠 Processando cadência temporal...")

    # Processamento acústico estruturado
    matriz_input = processar_audio_microfone(audio)

    # Ajusta as dimensões para a CRNN: (Batch, Altura, Largura, Canais) -> (1, 64, 64, 1)
    matriz_input = np.expand_dims(matriz_input, axis=-1)
    matriz_input = np.expand_dims(matriz_input, axis=0)

    # Predição da IA
    pred = model.predict(matriz_input, verbose=0)[0]

    idx = np.argmax(pred)
    conf = pred[idx]
    ent = calcular_entropia(pred)
    classe = inv_map[idx]

    # Regras de segurança (Filtros de ruído ou incerteza baseada em entropia)
    if conf < THRESHOLD or ent > ENTROPY_LIMIT:
        classe = "Comando desconhecido ou muito incerto"

    print("-" * 30)
    print(f"▶️  Resultado da IA: {classe.upper()}")
    print(f"📊 Confiança: {conf * 100:.2f}%")
    print(f"📉 Entropia Vetorial: {ent:.3f}")
    print("-" * 30)
    print("Pressione 'G' para gravar novamente ou 'Q' para sair.")

def loop():
    print("\n=== SISTEMA PRONTO PARA AGUARDAR COMANDOS ===")
    print("Pressione [ G ] para iniciar a gravação de 1.5s.")
    print("Pressione [ Q ] a qualquer momento para fechar o programa.")
    print("=============================================")

    while True:
        if keyboard.is_pressed('q'):
            print("\nEncerrando o testador de microfone.")
            break
        if keyboard.is_pressed('g'):
            classificar()
            # Pequeno delay para evitar múltiplas ativações acidentais pelo clique
            sd.sleep(300)

if __name__ == "__main__":
    loop()

# import os
# import json
# import wave
# import numpy as np
# import sounddevice as sd
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model
# import keyboard

# # --- CONFIGURAÇÕES DO EXPERIMENTO NOVO ---
# PASTA_EXPERIMENTO = "experimentos_ia"
# # MODEL_PATH = os.path.join(PASTA_EXPERIMENTO, "modelo_crnn_cadencia_30_geracoes_geral.h5")
# # LABELS_PATH = os.path.join(PASTA_EXPERIMENTO, "labels_config.json")

# MODEL_PATH = os.path.join(PASTA_EXPERIMENTO, "modelo_crnn_cadencia_log.h5")
# LABELS_PATH = os.path.join(PASTA_EXPERIMENTO, "labels_config_cadencia.json")

# SAMPLE_RATE = 16000
# DURATION = 1.5         # 1.5s igual ao treino da CRNN
# TARGET_SIZE = 64       # Tamanho da matriz quadrada (64x64)

# THRESHOLD = 0.55       # Limiar mínimo de confiança
# ENTROPY_LIMIT = 1.3    # Limiar máximo de incerteza (entropia)

# # =====================================================================
# # 1. FUNÇÕES MATEMÁTICAS MANUAIS 
# # =====================================================================
# def extrair_espectrograma_linear_manual(audio, n_fft=512, hop_length=256):
#     janela = np.hanning(n_fft)
#     num_frames = 1 + (len(audio) - n_fft) // hop_length
#     if num_frames <= 0: return np.zeros((TARGET_SIZE, TARGET_SIZE))
#     stft_matrix = []
#     for t in range(num_frames):
#         inicio = t * hop_length
#         fatia = audio[inicio:inicio + n_fft]
#         if len(fatia) < n_fft: fatia = np.pad(fatia, (0, n_fft - len(fatia)), 'constant')
#         fft_valores = np.abs(np.fft.rfft(fatia * janela))
#         stft_matrix.append(fft_valores)
#     return np.log1p(np.array(stft_matrix).T)

# def redimensionar_matriz_bilinear(matriz, target_size):
#     orig_h, orig_w = matriz.shape
#     if orig_h == 0 or orig_w == 0: return np.zeros((target_size, target_size))
#     grid_h = np.linspace(0, orig_h - 1, target_size)
#     grid_w = np.linspace(0, orig_w - 1, target_size)
#     y_b = grid_h.astype(np.int32)
#     y_a = np.minimum(y_b + 1, orig_h - 1)
#     x_e = grid_w.astype(np.int32)
#     x_d = np.minimum(x_e + 1, orig_w - 1)
#     dy = (grid_h - y_b)[:, None]
#     dx = (grid_w - x_e)[None, :]
#     return (1-dy)*(1-dx)*matriz[y_b[:,None], x_e] + (1-dy)*dx*matriz[y_b[:,None], x_d] + dy*(1-dx)*matriz[y_a[:,None], x_e] + dy*dx*matriz[y_a[:,None], x_d]

# def processar_audio_microfone(audio):
#     # DIAGNÓSTICO DO VAD: Reduzido o limiar de 0.03 para 0.015 para não engolir o "S" de "Siga"
#     limiar_energia = 0.015 * np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else 0
#     indices_uteis = np.where(np.abs(audio) > limiar_energia)[0]
    
#     if len(indices_uteis) > 0:
#         audio_util = audio[indices_uteis[0]:indices_uteis[-1]]
#     else:
#         audio_util = audio
        
#     # Centralização fixa ao invés de corte abrupto à direita para manter a cadência temporal
#     max_samples = int(SAMPLE_RATE * DURATION)
#     if len(audio_util) < max_samples:
#         pad_total = max_samples - len(audio_util)
#         pad_esquerdo = pad_total // 2
#         pad_direito = pad_total - pad_esquerdo
#         audio_final = np.pad(audio_util, (pad_esquerdo, pad_direito), 'constant')
#     else:
#         audio_final = audio_util[:max_samples]
        
#     pico = np.max(np.abs(audio_final))
#     if pico > 1e-6: 
#         audio_final = audio_final / pico
        
#     espectrograma = extrair_espectrograma_linear_manual(audio_final)
#     espectrograma_quadrado = redimensionar_matriz_bilinear(espectrograma, TARGET_SIZE)
#     return espectrograma_quadrado, audio_final

# def calcular_entropia(probs):
#     probs = probs + 1e-10
#     return -np.sum(probs * np.log(probs))

# # =====================================================================
# # 2. CARREGAMENTO DO MODELO E LABELS
# # =====================================================================
# print("🔄 Carregando modelo CRNN e configurações...")
# if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
#     raise FileNotFoundError("Modelo ou Labels não encontrados na pasta 'experimentos_ia'.")

# model = load_model(MODEL_PATH)
# with open(LABELS_PATH, "r") as f:
#     label_map = json.load(f)

# inv_map = {v: k for k, v in label_map.items()}

# # =====================================================================
# # 3. PIPELINE DE GRAVAÇÃO COM DIAGNÓSTICO VISUAL E AUDITIVO
# # =====================================================================
# def gravar_audio():
#     audio = sd.rec(int(DURATION * SAMPLE_RATE),
#                    samplerate=SAMPLE_RATE,
#                    channels=1,
#                    dtype='float32')
#     sd.wait()
#     return audio.flatten()

# def classificar():
#     print("\n🎤 Ouvindo... Fale agora!")
#     audio_bruto = gravar_audio()
    
#     # 1. RETORNO DE ÁUDIO: Toca exatamente o que foi capturado para você ouvir se há ruído excessivo ou cortes
#     print("🔊 Reproduzindo áudio capturado para verificação auditiva...")
#     sd.play(audio_bruto, SAMPLE_RATE)
#     sd.wait()

#     print("🧠 Processando e gerando assinatura visual dos dados...")
#     matriz_input, audio_processado = processar_audio_microfone(audio_bruto)

#     # 2. VISUALIZAÇÃO EM TEMPO REAL: Abre o gráfico mostrando como a IA enxerga o dado
#     plt.figure(figsize=(5, 4))
#     plt.imshow(matriz_input, aspect='auto', origin='lower', cmap='viridis')
#     plt.title("Assinatura Acústica Enviada ao Modelo (64x64)")
#     plt.colorbar(label="Intensidade Log")
#     plt.xlabel("Tempo (Frames Redimensionados)")
#     plt.ylabel("Frequência (Bins Lineares)")
#     plt.show(block=False) # block=False permite que o script continue sem travar o terminal
#     plt.pause(2.0)        # Mantém a janela aberta por 2 segundos antes de fechar automaticamente
#     plt.close()

#     # Prepara dimensão para entrada do modelo (1, 64, 64, 1)
#     matriz_tensor = np.expand_dims(matriz_input, axis=-1)
#     matriz_tensor = np.expand_dims(matriz_tensor, axis=0)

#     # Predição da IA
#     pred = model.predict(matriz_tensor, verbose=0)[0]

#     idx = np.argmax(pred)
#     conf = pred[idx]
#     ent = calcular_entropia(pred)
#     classe = inv_map[idx]

#     if conf < THRESHOLD or ent > ENTROPY_LIMIT:
#         classe = "Comando desconhecido ou muito incerto"

#     print("-" * 40)
#     print(f"▶️  Resultado da IA: {classe.upper()}")
#     print(f"📊 Confiança: {conf * 100:.2f}%")
#     print(f"📉 Entropia Vetorial: {ent:.3f}")
#     print("-" * 40)
#     print("\n[Distribuição de Probabilidade por classe]:")
#     for i, p in enumerate(pred):
#         print(f"   -> {inv_map[i]}: {p*100:.2f}%")
#     print("-" * 40)
#     print("Pressione 'G' para gravar novamente ou 'Q' para sair.")

# def loop():
#     print("\n=== SISTEMA DE DIAGNÓSTICO PRONTO ===")
#     print("Pressione [ G ] para iniciar a gravação de 1.5s.")
#     print("Pressione [ Q ] a qualquer momento para fechar o programa.")
#     print("=====================================")

#     while True:
#         if keyboard.is_pressed('q'):
#             print("\nEncerrando o testador de microfone.")
#             break
#         if keyboard.is_pressed('g'):
#             classificar()
#             sd.sleep(300)

# if __name__ == "__main__":
#     loop()