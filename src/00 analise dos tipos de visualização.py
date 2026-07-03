import os
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples

# --- CONFIGURAÇÕES DE CAMINHO ---
BASE_DIR = r"C:\Users\jefte\projetos em python\ufc 2025 a 2026\aprendizado de maquina\projeto de reconhecimento de voz AMRP"
DATASET_FINAL = os.path.join(BASE_DIR, "dataset_final")
PASTA_SAIDA = os.path.join(BASE_DIR, "visualizacoes_audio")
CLASSES = ["direita", "esquerda", "siga", "pare", "voltar"]

os.makedirs(PASTA_SAIDA, exist_ok=True)


print("🎬 Selecionando um áudio de exemplo para gerar os gráficos visuais...")
audio_exemplo_path = None
classe_exemplo = ""

for classe in CLASSES:
    caminho_classe = os.path.join(DATASET_FINAL, classe)
    if os.path.isdir(caminho_classe):
        arquivos = [f for f in os.listdir(caminho_classe) if f.endswith(".wav")]
        if arquivos:
            audio_exemplo_path = os.path.join(caminho_classe, arquivos[0])
            classe_exemplo = classe
            break

if not audio_exemplo_path:
    raise FileNotFoundError("Gere os dados no dataset_final primeiro para rodar esta análise.")

# Carregar o áudio de exemplo
audio_ex, sr_ex = sf.read(audio_exemplo_path)
tempo_ex = np.linspace(0, len(audio_ex)/sr_ex, len(audio_ex))

# Visualização 1: Waveform
plt.figure(figsize=(10, 4))
plt.plot(tempo_ex, audio_ex, color='#1f77b4', alpha=0.8)
plt.title(f"1. Domínio do Tempo (Waveform) - Classe: {classe_exemplo.capitalize()}", fontsize=12, fontweight='bold')
plt.xlabel("Tempo (segundos)")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, "1_waveform.png"), dpi=200)
plt.close()

# Visualização 2: Espectrograma Linear
stft_linear = np.abs(librosa.stft(audio_ex))
stft_db = librosa.amplitude_to_db(stft_linear, ref=np.max)
plt.figure(figsize=(10, 4))
librosa.display.specshow(stft_db, sr=sr_ex, x_axis='time', y_axis='linear', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title("2. Domínio da Frequência (Espectrograma Linear Exato)", fontsize=12, fontweight='bold')
plt.xlabel("Tempo (segundos)")
plt.ylabel("Frequência (Hz)")
plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, "2_espectrograma_linear.png"), dpi=200)
plt.close()

# Visualização 3: Espectrograma Mel
mel_spec_ex = librosa.feature.melspectrogram(y=audio_ex, sr=sr_ex, n_mels=128)
mel_db_ex = librosa.power_to_db(mel_spec_ex, ref=np.max)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_db_ex, sr=sr_ex, x_axis='time', y_axis='mel', cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title("3. Espectrograma Mel (Frequências Mapeadas ao Ouvido Humano)", fontsize=12, fontweight='bold')
plt.xlabel("Tempo (segundos)")
plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, "3_espectrograma_mel.png"), dpi=200)
plt.close()

# Visualização 4: MFCCs
mfccs_ex = librosa.feature.mfcc(y=audio_ex, sr=sr_ex, n_mfcc=13)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs_ex, sr=sr_ex, x_axis='time', cmap='coolwarm')
plt.colorbar()
plt.title("4. MFCCs (Filtro do Trato Vocal)", fontsize=12, fontweight='bold')
plt.xlabel("Tempo (segundos)")
plt.ylabel("Coeficientes MFCC")
plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, "4_mfccs.png"), dpi=200)
plt.close()

# Visualização 5: Cadência e Prosódia
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
rms_cadencia = librosa.feature.rms(y=audio_ex)[0]
frames_tempo = librosa.frames_to_time(range(len(rms_cadencia)), sr=sr_ex)
ax1.plot(frames_tempo, rms_cadencia, color='#d62728', linewidth=2.5)
ax1.fill_between(frames_tempo, rms_cadencia, alpha=0.2, color='#d62728')
ax1.set_title("5. Análise de Cadência: Envelope de Volume (Acentuação e Ritmo)", fontsize=11, fontweight='bold')
ax1.set_ylabel("Energia RMS")
ax1.grid(True, alpha=0.3)

pitches, magnitudes = librosa.core.piptrack(y=audio_ex, sr=sr_ex)
pitch_contour = [pitches[magnitudes[:, t].argmax(), t] if 50 < pitches[magnitudes[:, t].argmax(), t] < 400 else np.nan for t in range(pitches.shape[1])]
ax2.plot(frames_tempo, pitch_contour, color='#2ca02c', marker='o', markersize=3)
ax2.set_title("Contorno de Pitch (F0 - Curva de Entonação da Fala)", fontsize=11, fontweight='bold')
ax2.set_xlabel("Tempo (segundos)")
ax2.set_ylabel("Frequência F0 (Hz)")
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, "5_cadencia_e_prosodia.png"), dpi=200)
plt.close()

print("✅ Gráficos individuais salvos com sucesso!")


print("\n🔄 Iniciando análise em lote de todo o dataset_final para gerar as estatísticas do relatório...")

features_dict = {"Waveform": [], "Espectrograma Mel": [], "MFCCs": [], "Cadência e Ritmo": []}
labels = []

for label_idx, classe in enumerate(CLASSES):
    caminho_classe = os.path.join(DATASET_FINAL, classe)
    if not os.path.exists(caminho_classe): continue
    
    for arquivo in os.listdir(caminho_classe):
        if not arquivo.endswith(".wav"): continue
        try:
            audio, sr = sf.read(os.path.join(caminho_classe, arquivo))
            
            # Extraindo vetores padronizados para cálculo do Silhouette
            wave_feat = librosa.resample(audio, orig_sr=sr, target_sr=2000)[:3000]
            
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
            mel_feat = np.mean(librosa.power_to_db(mel_spec, ref=np.max), axis=1)
            
            mfcc_spec = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc_feat = np.mean(mfcc_spec, axis=1)
            
            rms_feat = librosa.feature.rms(y=audio)[0]
            
            features_dict["Waveform"].append(wave_feat)
            features_dict["Espectrograma Mel"].append(mel_feat)
            features_dict["MFCCs"].append(mfcc_feat)
            features_dict["Cadência e Ritmo"].append(rms_feat)
            labels.append(label_idx)
        except Exception:
            pass

y = np.array(labels)
scores_metodos = {}
scores_por_classe_mel = {classe: 0.0 for classe in CLASSES}

for metodo, lista_feat in features_dict.items():
    X = np.array(lista_feat)
    score_geral = silhouette_score(X, y)
    scores_metodos[metodo] = score_geral
    
    if metodo == "Espectrograma Mel":
        valores_individuais = silhouette_samples(X, y)
        for i, classe in enumerate(CLASSES):
            scores_por_classe_mel[classe] = np.mean(valores_individuais[y == i])

# Gerar Gráfico 6: Comparativo Final
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=200)

metodos_nomes = list(scores_metodos.keys())
metodos_valores = list(scores_metodos.values())
bars1 = ax1.bar(metodos_nomes, metodos_valores, color=['#7f8c8d', '#2ecc71', '#2980b9', '#d35400'], edgecolor='black')
ax1.set_title("Qual forma de ver os dados é MELHOR para IA?\n(Silhouette Score Geral - Quanto maior, melhor)", fontsize=11, fontweight='bold')
ax1.set_ylabel("Silhouette Score")
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.set_ylim(min(metodos_valores) - 0.05, max(metodos_valores) + 0.1)
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.01, f'{bar.get_height():.3f}', ha='center', va='bottom', fontweight='bold')

classes_nomes = [c.capitalize() for c in scores_por_classe_mel.keys()]
classes_valores = list(scores_por_classe_mel.values())
bars2 = ax2.bar(classes_nomes, classes_valores, color='#9b59b6', edgecolor='black')
ax2.set_title("Quais classes são mais DIFERENCIÁVEIS entre si?\n(Análise baseada no Espectrograma Mel)", fontsize=11, fontweight='bold')
ax2.set_ylabel("Silhouette Score por Classe")
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.set_ylim(min(classes_valores) - 0.05, max(classes_valores) + 0.1)
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.01, f'{bar.get_height():.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
caminho_grafico_final = os.path.join(PASTA_SAIDA, "6_comparativo_relatorio_ia.png")
plt.savefig(caminho_grafico_final)
plt.close()

print(f"\n🎉 COMPLETO! Todas as 6 imagens foram salvas em:\n➡️ {PASTA_SAIDA}")