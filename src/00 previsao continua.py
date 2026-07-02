import os
import json
import wave
import numpy as np
import sounddevice as sd
from tensorflow.keras.models import load_model
import keyboard

# --- CONFIGURAÇÕES DO EXPERIMENTO ---
PASTA_EXPERIMENTO = "experimentos_ia"
MODEL_PATH = os.path.join(PASTA_EXPERIMENTO, "modelo_crnn_cadencia_30_geracoes.h5")
LABELS_PATH = os.path.join(PASTA_EXPERIMENTO, "labels_config.json")

SAMPLE_RATE = 16000
DURATION = 1.5         # Duração ideal do modelo focado em cadência
TARGET_SIZE = 64       # Matriz 64x64 esperada pela CRNN

THRESHOLD = 0.60       # Limiar mínimo de confiança da IA
ENTROPY_LIMIT = 1.2    # Limiar máximo de incerteza (entropia)

# --- PARÂMETROS DO VAD CONTÍNUO (DETECÇÃO DE VOZ) ---
CHUNK_DURATION = 0.100  # Analisa o microfone a cada 100 milissegundos
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
LIMIAR_ENERGIA_VAD = 0.025  # Sensibilidade para disparo (ajuste se seu microfone for muito sensível)
SILENCE_TIMEOUT_CHUNKS = 3  # Quantos blocos de silêncio seguidos determinam o FIM da palavra (~300ms)

# =====================================================================
# 1. FUNÇÕES MATEMÁTICAS MANUAIS (IGUAIS AO TREINO DA CRNN)
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

def processar_palavra_inteira(audio_completo):
    """
    Remove silêncios residuais e centraliza a palavra INTEIRA 
    dentro de uma janela exata de 1.5s para preservar a cadência.
    """
    # Encontra trecho com voz ativa real por energia
    limiar_interno = 0.03 * np.max(np.abs(audio_completo)) if np.max(np.abs(audio_completo)) > 0 else 0
    indices = np.where(np.abs(audio_completo) > limiar_interno)[0]
    if len(indices) > 0:
        audio_util = audio_completo[indices[0]:indices[-1]]
    else:
        audio_util = audio_completo

    max_samples = int(SAMPLE_RATE * DURATION) # 24000 amostras para 1.5s
    
    # Centralização perfeita do bloco utilitário (impede fragmentação)
    if len(audio_util) < max_samples:
        pad_total = max_samples - len(audio_util)
        pad_esquerdo = pad_total // 2
        pad_direito = pad_total - pad_esquerdo
        audio_final = np.pad(audio_util, (pad_esquerdo, pad_direito), 'constant')
    else:
        audio_final = audio_util[:max_samples]

    # Normalização de ganho
    pico = np.max(np.abs(audio_final))
    if pico > 1e-6: 
        audio_final = audio_final / pico

    espectrograma = extrair_espectrograma_linear_manual(audio_final)
    espectrograma_quadrado = redimensionar_matriz_bilinear(espectrograma, TARGET_SIZE)
    return espectrograma_quadrado

def calcular_entropia(probs):
    probs = probs + 1e-10
    return -np.sum(probs * np.log(probs))

# =====================================================================
# 2. CARREGAMENTO DA INTELIGÊNCIA ARTIFICIAL
# =====================================================================
print("🔄 Carregando modelo CRNN e dicionário de classes...")
if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    raise FileNotFoundError("Certifique-se de que o modelo heterogêneo foi gerado na pasta 'experimentos_ia'.")

model = load_model(MODEL_PATH)
with open(LABELS_PATH, "r") as f:
    label_map = json.load(f)
inv_map = {v: k for k, v in label_map.items()}

# =====================================================================
# 3. STREAM DE ÁUDIO E MONITORAÇÃO CONTÍNUA DO VAD
# =====================================================================
def executar_captura_continua():
    print("\n=============================================")
    print("🎙️  SISTEMA DE ESCUTA ATIVA AUTOMÁTICO INICIADO")
    print("-> Fale um comando e faça silêncio ao terminar.")
    print("-> Pressione [ Q ] a qualquer momento para fechar.")
    print("=============================================\n")

    # Inicialização de buffers auxiliares
    em_gravaçao = False
    buffer_palavra = []
    chunks_de_silencio_contador = 0
    
    # Buffer circular de pre-roll para não cortar o primeiríssimo milissegundo fonético
    buffer_pre_roll = []

    # Inicializa o stream contínuo de entrada do sounddevice
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=CHUNK_SAMPLES, dtype='float32') as stream:
        while True:
            if keyboard.is_pressed('q'):
                print("\nStream de áudio contínuo finalizado.")
                break

            # Leitura do bloco de 100ms atual do microfone
            chunk, overflowed = stream.read(CHUNK_SAMPLES)
            chunk = chunk.flatten()
            
            # Cálculo de energia RMS do bloco atual
            energia_rms = np.sqrt(np.mean(chunk**2))

            if not em_gravaçao:
                # Mantém o pre-roll atualizado com o último bloco de silêncio estável
                buffer_pre_roll = list(chunk)
                
                # Se a energia subiu acima do limiar, o usuário começou a falar!
                if energia_rms > LIMIAR_ENERGIA_VAD:
                    em_gravaçao = True
                    buffer_palavra = buffer_pre_roll + list(chunk)
                    chunks_de_silencio_contador = 0
                    print("🔊 [VAD]: Fala detectada! Gravando estrutura completa da palavra...", end="\r")
            else:
                # Se já está gravando, adiciona o bloco atual à palavra em construção
                buffer_palavra.extend(list(chunk))
                
                # Se o bloco atual voltou a ficar abaixo do limiar, conta como silêncio
                if energia_rms < LIMIAR_ENERGIA_VAD:
                    chunks_de_silencio_contador += 1
                else:
                    chunks_de_silencio_contador = 0 # Reseta se voltou a emitir som

                # Condição de encerramento: O usuário parou de falar por tempo suficiente
                if chunks_de_silencio_contador >= SILENCE_TIMEOUT_CHUNKS:
                    em_gravaçao = False
                    audio_final_completado = np.array(buffer_palavra, dtype=np.float32)
                    
                    # Filtra gravações fantasmas (ex: cliques de mouse rápidos < 300ms)
                    if len(audio_final_completado) > int(SAMPLE_RATE * 0.35):
                        print("\n🛑 [VAD]: Fim da palavra detectado. Enviando para análise da CRNN...")
                        
                        # Processamento espacial (Gera a matriz quadrada balanceada na cadência)
                        matriz_input = processar_palavra_inteira(audio_final_completado)
                        matriz_input = np.expand_dims(matriz_input, axis=-1)
                        matriz_input = np.expand_dims(matriz_input, axis=0)
                        
                        # Predição da Rede Neural
                        pred = model.predict(matriz_input, verbose=0)[0]
                        idx = np.argmax(pred)
                        conf = pred[idx]
                        ent = calcular_entropia(pred)
                        classe = inv_map[idx]
                        
                        # Filtros de exclusão por incerteza vetorial ou ruído de ambiente espúrio
                        if conf < THRESHOLD or ent > ENTROPY_LIMIT:
                            classe = "Incerteza acústica / Comando inválido"
                        
                        print("-" * 40)
                        print(f"▶️  Comando Identificado: {classe.upper()}")
                        print(f"📊 Confiança Estatística: {conf * 100:.2f}%")
                        print(f"📉 Entropia da Decisão: {ent:.3f}")
                        print("-" * 40 + "\n🎙️  Aguardando próximo comando...")
                    else:
                        print("🔈 [VAD]: Ruído rápido descartado automaticamente.          ", end="\r")
                    
                    # Limpa buffers para o próximo ciclo
                    buffer_palavra = []
                    chunks_de_silencio_contador = 0

if __name__ == "__main__":
    executar_captura_continua()