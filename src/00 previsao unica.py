import os
import json
import wave
import numpy as np
import sounddevice as sd
from tensorflow.keras.models import load_model
import keyboard

# --- CONFIGURAÇÕES DO EXPERIMENTO NOVO ---
PASTA_EXPERIMENTO = "experimentos_ia"
MODEL_PATH = os.path.join(PASTA_EXPERIMENTO, "modelo_crnn_cadencia_30_geracoes.h5")
LABELS_PATH = os.path.join(PASTA_EXPERIMENTO, "labels_config.json")

SAMPLE_RATE = 16000
DURATION = 1.5         # Atualizado para 1.5s igual ao treino da CRNN
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