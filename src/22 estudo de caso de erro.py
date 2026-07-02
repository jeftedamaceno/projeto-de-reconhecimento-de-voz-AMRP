import os
import json
import time
import csv
import numpy as np
import sounddevice as sd
from tensorflow.keras.models import load_model
import keyboard

# --- CONFIGURAÇÕES DE INFRAESTRUTURA ---
PASTA_EXPERIMENTO = "experimentos_ia"
MODEL_PATH = os.path.join(PASTA_EXPERIMENTO, "modelo_crnn_otimizado_fonemas.h5")
LABELS_PATH = os.path.join(PASTA_EXPERIMENTO, "labels_config.json")

# Pastas de Auditoria para o Aprendizado Ativo
PASTA_AUDITORIA_ERROS = "auditoria_erros_reais"
ARQUIVO_LOG_CSV = os.path.join(PASTA_AUDITORIA_ERROS, "relatorio_auditoria.csv")

os.makedirs(PASTA_AUDITORIA_ERROS, exist_ok=True)

SAMPLE_RATE = 16000
DURATION = 1.5         
TARGET_SIZE = 128      
THRESHOLD_CONFIAVEL = 0.75  # 75% conforme solicitado para o filtro estrito

def calcular_entropia_vetorial(probabilidades):
    return -np.sum(probabilidades * np.log2(probabilidades + 1e-9))

# =====================================================================
# EXTRAÇÃO E PROCESSAMENTO DE SINAL (CADÊNCIA RÍTMICA)
# =====================================================================
def extrair_espectrograma_cadencia(audio, n_fft=512, hop_length=128):
    janela = np.hanning(n_fft)
    num_frames = 1 + (len(audio) - n_fft) // hop_length
    if num_frames <= 0: return np.zeros((TARGET_SIZE, TARGET_SIZE))
    
    stft_matrix = []
    for t in range(num_frames):
        inicio = t * hop_length
        fatia = audio[inicio:inicio + n_fft]
        if len(fatia) < n_fft: 
            fatia = np.pad(fatia, (0, n_fft - len(fatia)), 'constant')
        stft_matrix.append(np.abs(np.fft.rfft(fatia * janela)))
    
    stft_matrix = np.array(stft_matrix).T  
    envelope_modulacao = np.abs(np.diff(stft_matrix, axis=1))
    envelope_modulacao = np.pad(envelope_modulacao, ((0, 0), (1, 0)), 'constant')
    return np.log1p(envelope_modulacao * 8.0)

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
    max_samples = int(SAMPLE_RATE * DURATION)
    if len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)), 'constant')
    else:
        audio = audio[:max_samples]
    pico = np.max(np.abs(audio))
    if pico > 1e-6: audio = audio / pico
    
    matriz_cadencia = extrair_espectrograma_cadencia(audio)
    img = redimensionar_matriz_bilinear(matriz_cadencia, TARGET_SIZE)
    return (img - np.mean(img)) / (np.std(img) + 1e-9)

# =====================================================================
# GERENCIADOR DO RELATÓRIO METROLÓGICO (CSV)
# =====================================================================
def inicializar_csv():
    if not os.path.exists(ARQUIVO_LOG_CSV):
        with open(ARQUIVO_LOG_CSV, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Nome_Arquivo", "Classe_Alvo_Humana", 
                "Classe_Predita_IA", "Confianca_IA", "Entropia_Incerteza", 
                "Motivo_Falha", "Energia_Sinal_RMS"
            ])

def registrar_falha_csv(nome_arq, alvo, predito, conf, entropia, motivo, rms):
    with open(ARQUIVO_LOG_CSV, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"), nome_arq, alvo, 
            predito, f"{conf*100:.2f}%", f"{entropia:.4f}", motivo, f"{rms:.5f}"
        ])

# =====================================================================
# INICIALIZAÇÃO DO MODELO E LABELS
# =====================================================================
print("🔄 Carregando ecossistema de Cadência (.keras)...")
if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    raise FileNotFoundError("Modelo ou arquivo de labels ausente em 'experimentos_ia'.")

model = load_model(MODEL_PATH)
with open(LABELS_PATH, "r") as f:
    label_map = json.load(f)
inv_map = {v: k for k, v in label_map.items()}
inicializar_csv()

# =====================================================================
# LOOP DE CAPTURA COM HUMAN-IN-THE-LOOP
# =====================================================================
def executar_pipeline_coleta():
    print("\n" + "═"*50)
    print("📋 CLASSES DISPONÍVEIS:", ", ".join([c.upper() for c in label_map.keys()]))
    classe_alvo = input("✍️ Digite qual classe você vai falar agora: ").strip().lower()
    
    if classe_alvo not in label_map:
        print("❌ Classe inválida! Verifique a grafia correta.")
        return

    print(f"🎯 Pronto para falar: '{classe_alvo.upper()}'. Pressione [ G ] para iniciar a gravação...")
    
    while True:
        if keyboard.is_pressed('g'):
            break
            
    print("🎤 [GRAVANDO...] Fale agora!")
    audio_raw = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio = audio_raw.flatten()
    
    # Métrica extra que julguei importante: Energia RMS do sinal capturado (detecta se o usuário falou muito baixo ou soprou)
    energia_rms = np.sqrt(np.mean(audio**2))
    
    print("🧠 Processando e inferindo textura...")
    matriz_input = processar_audio_microfone(audio)
    tensor_input = np.expand_dims(np.expand_dims(matriz_input, axis=-1), axis=0)
    
    predicoes = model.predict(tensor_input, verbose=0)[0]
    idx_vencedor = np.argmax(predicoes)
    confianca = predicoes[idx_vencedor]
    entropia = calcular_entropia_vetorial(predicoes)
    classe_predita = inv_map[idx_vencedor]
    
    # Critérios de Auditoria Estrita
    errou_classe = (classe_predita != classe_alvo)
    baixa_confianca = (confianca < THRESHOLD_CONFIAVEL)
    
    print("\n" + "📊 RESULTADO DA INFERÊNCIA " + "═"*33)
    print(f"   • Intenção Humana: {classe_alvo.upper()}")
    print(f"   • Resposta da IA:  {classe_predita.upper()} ({confianca*100:.2f}% de certeza)")
    print(f"   • Entropia Vetorial: {entropia:.3f}")
    
    if errou_classe or baixa_confianca:
        motivo = "Erro de Classificacao" if errou_classe else "Baixa Confianca (<75%)"
        print(f"🚨 FALHA DETECTADA -> Motivo: {motivo}")
        
        # Nomenclatura descritiva solicitada: ajuda a identificar instantaneamente o arquivo na pasta
        timestamp = int(time.time())
        nome_arquivo = f"ALVO_{classe_alvo}_PREDITO_{classe_predita}_CONF_{int(confianca*100)}_{timestamp}.wav"
        caminho_wav = os.path.join(PASTA_AUDITORIA_ERROS, nome_arquivo)
        
        # Como o sounddevice grava normalizado em float32, usamos a biblioteca para salvar direto sem wave overhead
        import soundfile as sf
        sf.write(caminho_wav, audio, SAMPLE_RATE)
        
        # Registra no livro de bordo CSV
        registrar_falha_csv(nome_arquivo, classe_alvo, classe_predita, confianca, entropia, motivo, energia_rms)
        print(f"💾 Áudio e métricas arquivados com sucesso em '{PASTA_AUDITORIA_ERROS}'!")
    else:
        print("✅ COMANDO PERFEITO! A IA acertou com alta confiança. (Nenhum arquivo foi armazenado)")
        
    print("═"*60)
    print("\nPressione [ ENTER ] para nova captura ou [ Q ] para encerrar.")

def main():
    print("==========================================================")
    print(" SISTEMA COLETOR DE AUDITORIA ATIVA - CADÊNCIA SILÁBICA ")
    print("==========================================================")
    while True:
        executar_pipeline_coleta()
        opcao = input().strip().lower()
        if opcao == 'q' or keyboard.is_pressed('q'):
            print("Encerrando coletor.")
            break

if __name__ == "__main__":
    main()