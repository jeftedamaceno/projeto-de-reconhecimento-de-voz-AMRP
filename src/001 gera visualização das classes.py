import os
import wave
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# CONFIGURAÇÕES E DIRETÓRIOS
# =====================================================================
DIR_ORIGINAL = "dataset_final"   # Seus áudios puros por classe
DIR_AUGMENTED = "dataset_ruido"  # Onde o seu modelo salva as misturas
OUTPUT_VIS_DIR = "visualizacoes_dataset"
SAMPLE_RATE = 16000
DURATION = 1.5
N_INTERVALOS = 90  

os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

def carregar_e_normalizar_wav(file_path):
    try:
        with wave.open(file_path, 'rb') as wav_file:
            n_channels = wav_file.getnchannels()
            sampwidth = wav_file.getsampwidth()
            raw_data = wav_file.readframes(wav_file.getnframes())
            if sampwidth == 2:
                audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
            else:
                return None
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels).mean(axis=1)
        
        max_samples = int(SAMPLE_RATE * DURATION)
        if len(audio) < max_samples:
            audio = np.pad(audio, (0, max_samples - len(audio)), 'constant')
        else:
            audio = audio[:max_samples]
            
        pico = np.max(np.abs(audio))
        if pico > 1e-6: audio = audio / pico
        return audio
    except:
        return None

def extrair_perfil_espectral_90_bins(audio):
    fatia_tamanho = len(audio) // N_INTERVALOS
    perfil = []
    for i in range(N_INTERVALOS):
        inicio = i * fatia_tamanho
        fim = inicio + fatia_tamanho
        fatia = audio[inicio:fim]
        fft_valores = np.abs(np.fft.rfft(fatia))
        energia_log = np.log1p(np.mean(fft_valores) * 10.0)
        perfil.append(energia_log)
    return np.array(perfil)

def coletar_perfis_pasta(pasta_classe):
    if not os.path.exists(pasta_classe): return None
    arquivos = [os.path.join(pasta_classe, f) for f in os.listdir(pasta_classe) if f.endswith('.wav')]
    perfis = []
    for f in arquivos:
        audio = carregar_e_normalizar_wav(f)
        if audio is not None:
            perfis.append(extrair_perfil_espectral_90_bins(audio))
    return np.array(perfis) if len(perfis) > 0 else None

def gerar_perfil_ruido_gaussiano_puro():
    perfis_ruido = []
    for _ in range(50):
        ruido_puro = np.random.normal(0, 0.015, int(SAMPLE_RATE * DURATION))
        pico = np.max(np.abs(ruido_puro))
        if pico > 0: ruido_puro = ruido_puro / pico
        perfis_ruido.append(extrair_perfil_espectral_90_bins(ruido_puro))
    return np.array(perfis_ruido)

# =====================================================================
# PROCESSAMENTO LOOP AUTOMÁTICO PARA TODAS AS CLASSES
# =====================================================================
if not os.path.exists(DIR_ORIGINAL):
    print(f"⚠️ Erro: A pasta '{DIR_ORIGINAL}' não existe.")
    exit()

# Detecta automaticamente as classes do seu projeto
classes = [d for d in os.listdir(DIR_ORIGINAL) if os.path.isdir(os.path.join(DIR_ORIGINAL, d))]
print(f"🔍 Classes detectadas para análise: {classes}")

perfis_ruido_puro = gerar_perfil_ruido_gaussiano_puro()
media_ruido = np.mean(perfis_ruido_puro, axis=0)
std_ruido  = np.std(perfis_ruido_puro, axis=0)

for classe in classes:
    print(f"📊 Processando cenário acústico para a classe: '{classe.upper()}'...")
    
    pasta_orig = os.path.join(DIR_ORIGINAL, classe)
    pasta_aug = os.path.join(DIR_AUGMENTED, classe)
    
    perfis_originais = coletar_perfis_pasta(pasta_orig)
    perfis_augmented = coletar_perfis_pasta(pasta_aug)
    
    if perfis_originais is None:
        print(f"   ⚠️ Sem áudios válidos na pasta original de '{classe}'. Pulando...")
        continue
        
    media_orig = np.mean(perfis_originais, axis=0)
    std_orig  = np.std(perfis_originais, axis=0)
    
    # Montagem do Gráfico Individual da Classe
    plt.figure(figsize=(14, 6), dpi=150)
    
    # 1. Ruído de Referência (Vermelho)
    plt.plot(range(N_INTERVALOS), media_ruido, color='#c0392b', linewidth=1.5, linestyle=':', label='Referência: Ruído Gaussiano Puro')
    plt.fill_between(range(N_INTERVALOS), media_ruido - std_ruido, media_ruido + std_ruido, color='#e74c3c', alpha=0.08)
    
    # 2. Dados Originais Puros (Azul)
    plt.plot(range(N_INTERVALOS), media_orig, color='#2980b9', linewidth=3.0, label=f'Cenário A: Dados Originais Puros ("{classe.upper()}")')
    plt.fill_between(range(N_INTERVALOS), media_orig - std_orig, media_orig + std_orig, color='#3498db', alpha=0.15)
    
    # 3. Dados Aumentados com Mix (Verde)
    if perfis_augmented is not None:
        media_aug = np.mean(perfis_augmented, axis=0)
        std_aug  = np.std(perfis_augmented, axis=0)
        plt.plot(range(N_INTERVALOS), media_aug, color='#27ae60', linewidth=2.5, linestyle='--', label='Cenário C: Com Mix de Augmentation (Treino Real)')
        plt.fill_between(range(N_INTERVALOS), media_aug - std_aug, media_aug + std_aug, color='#2ecc71', alpha=0.12)
    else:
        print(f"   ℹ️ Nota: Sem dados aumentados em '{pasta_aug}'. Plotando apenas o original.")
        
    plt.title(f"Análise Comparativa de Cenários Acústicos: Classe '{classe.upper()}'", fontsize=12, fontweight='bold')
    plt.xlabel("Linha do Tempo Modular (90 Intervalos de Resolução)", fontsize=10)
    plt.ylabel("Assinatura de Energia Espectral (Log-Scale)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="upper right", framealpha=0.95)
    plt.tight_layout()
    
    nome_arquivo = f"confronto_acustico_total_{classe}.png"
    plt.savefig(os.path.join(OUTPUT_VIS_DIR, nome_arquivo))
    plt.close()
    print(f"   🎉 Gráfico salvo: '{OUTPUT_VIS_DIR}/{nome_arquivo}'")

print("\n✔️ Processamento concluído! Verifique a pasta 'visualizacoes_dataset'.")