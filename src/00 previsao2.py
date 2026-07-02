import os
import json
import wave
import numpy as np
import sounddevice as sd
from tensorflow.keras.models import load_model
import keyboard

# --- CONFIGURAÇÕES DO EXPERIMENTO ---
PASTA_EXPERIMENTO = "experimentos_ia"
LABELS_PATH = os.path.join(PASTA_EXPERIMENTO, "labels_config.json")

# Caminhos dos 3 modelos salvos no script anterior
MODELOS_CONFIG = {
    "Modelo 1 (CNN + Espectrograma)": {
        "arquivo": "modelo_1_cnn_espectrograma.h5",
        "modo": "cnn_espectrograma",
        "expandir_dim": True
    },
    "Modelo 2 (Filtros Temporais Puros)": {
        "arquivo": "modelo_2_temporal_puro.h5",
        "modo": "temporal_filtro_puro",
        "expandir_dim": False
    },
    "Modelo 3 (Banco de Filtros Mel)": {
        "arquivo": "modelo_3_banco_mel_recorrente.h5",
        "modo": "classico_mel_hmm",
        "expandir_dim": False
    }
}

SAMPLE_RATE = 16000
DURATION = 1.5         # 1.5 segundos igual ao pipeline de treino

# =====================================================================
# GERADOR MANUAL DE BANCO DE FILTROS MEL (SEM LIBROSA)
# =====================================================================
def criar_banco_filtros_mel(n_fft, n_mels, sample_rate):
    f_min = 0.0
    f_max = sample_rate / 2.0
    mel_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + f_max / 700.0)
    
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    freq_pts = 700.0 * (10.0**(mel_pts / 2595.0) - 1.0)
    
    bins = np.floor((n_fft + 1) * freq_pts / sample_rate).astype(int)
    fb = np.zeros((n_fft // 2 + 1, n_mels))
    
    for m in range(1, n_mels + 1):
        for k in range(bins[m - 1], bins[m]):
            fb[k, m - 1] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        for k in range(bins[m], bins[m + 1]):
            fb[k, m - 1] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
    return fb

MATRIZ_MEL_FILTRO = criar_banco_filtros_mel(n_fft=512, n_mels=40, sample_rate=SAMPLE_RATE)

# =====================================================================
# EXTRATORES DE RECURSOS IDENTICOS AO TREINO
# =====================================================================
def extrair_stft_pura(audio, n_fft=512, hop_length=256):
    janela = np.hanning(n_fft)
    num_frames = 1 + (len(audio) - n_fft) // hop_length
    if num_frames <= 0: return np.zeros((1, n_fft // 2 + 1))
    stft_matrix = []
    for t in range(num_frames):
        inicio = t * hop_length
        fatia = audio[inicio:inicio + n_fft]
        if len(fatia) < n_fft: 
            fatia = np.pad(fatia, (0, n_fft - len(fatia)), 'constant')
        stft_matrix.append(np.abs(np.fft.rfft(fatia * janela)))
    return np.array(stft_matrix)

def redimensionar_bilinear_tempo_fixo(matriz, target_h, target_w):
    orig_h, orig_w = matriz.shape
    grid_h = np.linspace(0, orig_h - 1, target_h)
    grid_w = np.linspace(0, orig_w - 1, target_w)
    y_b = grid_h.astype(np.int32)
    y_a = np.minimum(y_b + 1, orig_h - 1)
    x_e = grid_w.astype(np.int32)
    x_d = np.minimum(x_e + 1, orig_w - 1)
    dy = (grid_h - y_b)[:, None]
    dx = (grid_w - x_e)[None, :]
    return (1-dy)*(1-dx)*matriz[y_b[:,None], x_e] + (1-dy)*dx*matriz[y_b[:,None], x_d] + dy*(1-dx)*matriz[y_a[:,None], x_e] + dy*dx*matriz[y_a[:,None], x_d]

def adaptar_audio_para_entrada(audio, modo):
    """Garante o processamento idêntico ao esperado por cada IA individual."""
    stft = extrair_stft_pura(audio)
    
    if modo == 'cnn_espectrograma':
        espectrograma_log = np.log1p(stft * 15.0)
        delta = np.diff(espectrograma_log, axis=0)
        delta = np.pad(delta, ((1, 0), (0, 0)), 'constant')
        img_fusao = espectrograma_log + delta
        img_res = redimensionar_bilinear_tempo_fixo(img_fusao.T, 64, 64)
        return (img_res - np.mean(img_res)) / (np.std(img_res) + 1e-9)
        
    elif modo == 'temporal_filtro_puro':
        img_lin = redimensionar_bilinear_tempo_fixo(stft, stft.shape[0], 40)
        img_lin = np.log1p(img_lin * 10.0)
        return (img_lin - np.mean(img_lin)) / (np.std(img_lin) + 1e-9)
        
    elif modo == 'classico_mel_hmm':
        mel_espectrograma = np.dot(stft, MATRIZ_MEL_FILTRO)
        mel_log = np.log1p(mel_espectrograma * 20.0)
        return (mel_log - np.mean(mel_log)) / (np.std(mel_log) + 1e-9)

# =====================================================================
# CARREGAMENTO DO ECOSSISTEMA DE MODELOS
# =====================================================================
print("🔄 Inicializando testador e carregando as 3 Redes Neurais...")
if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError("Mapeamento 'labels_config.json' não encontrado. Rode o treino primeiro.")

with open(LABELS_PATH, "r") as f:
    label_map = json.load(f)
inv_map = {v: k for k, v in label_map.items()}

modelos_carregados = {}
for nome_ia, cfg in MODELOS_CONFIG.items():
    caminho_completo = os.path.join(PASTA_EXPERIMENTO, cfg["arquivo"])
    if os.path.exists(caminho_completo):
        print(f"   -> {nome_ia} carregado.")
        modelos_carregados[nome_ia] = {
            "modelo_objeto": load_model(caminho_completo),
            "modo": cfg["modo"],
            "expandir_dim": cfg["expandir_dim"]
        }
    else:
        print(f"   ⚠️  Aviso: {cfg['arquivo']} não encontrado na pasta de experimentos.")

# =====================================================================
# PIPELINE DE CAPTURA E INFERÊNCIA COMPARATIVA
# =====================================================================
def executar_teste_comparativo():
    print("\n🎤 [OUVINDO]... Fale o comando de voz AGORA!")
    
    # Grava áudio bruto do microfone
    audio_raw = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio = audio_raw.flatten()
    
    # Padronização de ganho de pico inicial
    pico = np.max(np.abs(audio))
    if pico > 1e-6: audio = audio / pico
    
    print("🧠 Áudio capturado. Ramificando entradas e calculando predições...")
    
    resultados = {}
    
    # Loop que alimenta as 3 IAs de forma independente usando o mesmo áudio base
    for nome_ia, estrutura in modelos_carregados.items():
        # Gera o vetor matemático sob medida exigido por este modelo específico
        matriz_input = adaptar_audio_para_entrada(audio, modo=estrutura["modo"])
        
        # Ajusta as dimensões do Tensor do Keras
        tensor_input = np.expand_dims(matriz_input, axis=0) # Adiciona dimensão de Batch (1, ...)
        if estrutura["expandir_dim"]:
            tensor_input = np.expand_dims(tensor_input, axis=-1) # Adiciona canal de cor para CNN (1, 64, 64, 1)
            
        # Executa a predição
        predicoes_vetor = estrutura["modelo_objeto"].predict(tensor_input, verbose=0)[0]
        
        idx_vencedor = np.argmax(predicoes_vetor)
        resultados[nome_ia] = {
            "classe": inv_map[idx_vencedor].upper(),
            "confianca": predicoes_vetor[idx_vencedor] * 100,
            "vetor_completo": predicoes_vetor
        }

    # --- IMPRESSÃO DO PAINEL DE RESULTADOS COMPARATIVOS ---
    print("\n" + "🏁 " + "═"*70 + " 🏁")
    print("                      DIAGNÓSTICO COMPARATIVO EM TEMPO REAL")
    print("═"*75)
    print(f"{'Modelo Testado':<36} | {'Classe Eleita':<15} | {'Confiança':<12}")
    print("─"*75)
    for nome_ia, res in resultados.items():
        print(f"{nome_ia:<36} | {res['classe']:<15} | {res['confianca']:.2f}%")
        
    print("─"*75)
    print("🔍 DETALHAMENTO DA DISTRIBUIÇÃO DE PROBABILIDADES POR CLASSE:")
    for nome_ia, res in resultados.items():
        print(f"\n🔹 {nome_ia}:")
        for i, prob in enumerate(res["vetor_completo"]):
            # Cria uma pequena barra visual de preenchimento para facilitar a leitura no terminal
            barra = "█" * int(prob * 15)
            print(f"   [{inv_map[i].upper():<10}]: {prob*100:6.2f}% {barra}")
            
    print("\n" + "═"*75)
    print("Pressione 'G' para gravar um novo teste ou 'Q' para encerrar.")

def loop_escuta():
    print("\n" + "="*50)
    print("SISTEMA SIMULTÂNEO DE PREVISÃO MULTI-IA")
    print("Pressione [ G ] para iniciar a gravação de 1.5 segundos.")
    print("Pressione [ Q ] para fechar o programa.")
    print("="*50)

    while True:
        if keyboard.is_pressed('q'):
            print("\nEncerrando o testador simultâneo.")
            break
        if keyboard.is_pressed('g'):
            executar_teste_comparativo()
            sd.sleep(400) # Evita múltiplos gatilhos com um único clique rápido

if __name__ == "__main__":
    if not modelos_carregados:
        print("❌ Erro: Nenhum arquivo de modelo foi encontrado na pasta 'experimentos_ia'. Certifique-se de rodar o script de treino primeiro.")
    else:
        loop_escuta()