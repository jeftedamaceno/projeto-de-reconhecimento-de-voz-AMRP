# import os
# import numpy as np
# import soundfile as sf
# import matplotlib.pyplot as plt

# # --- CONFIGURAÇÕES DO SEU PROJETO (IGUAIS AO SEU PIPELINE) ---
# BASE_DIR = r"C:\Users\jefte\projetos em python\ufc 2025 a 2026\aprendizado de maquina\projeto de reconhecimento de voz AMRP"
# ORIGEM = os.path.join(BASE_DIR, "dataset_vozes_old")
# CLASSES = ["direita", "esquerda", "siga", "pare", "voltar"]
# EXTENSOES = (".wav", ".flac", ".ogg", ".m4a", ".mp3")
# SAMPLE_RATE = 16000

# def calcular_duracao_util_rms(audio, sr, frame_length=256, hop_length=128, threshold_db=-35):
#     """
#     Calcula o tempo de áudio útil usando a energia RMS em janelas móveis (frames),
#     sendo muito mais robusto contra pequenos ruídos isolados ou estalos.
#     """
#     if len(audio) == 0:
#         return 0
        
#     # Normaliza o áudio
#     audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
#     # Divisão em frames para cálculo de energia RMS
#     num_frames = int(np.floor((len(audio) - frame_length) / hop_length) + 1)
#     if num_frames <= 0:
#         return len(audio) / sr
        
#     rms = np.zeros(num_frames)
#     for t in range(num_frames):
#         start = t * hop_length
#         end = start + frame_length
#         rms[t] = np.sqrt(np.mean(audio[start:end]**2) + 1e-8)
        
#     # Converte o RMS para Decibéis (dB) em relação ao máximo
#     rms_db = 20 * np.log10(rms / (np.max(rms) + 1e-8))
    
#     # Encontra os frames que superam o limiar de silêncio (threshold)
#     frames_uteis = np.where(rms_db > threshold_db)[0]
    
#     if len(frames_uteis) == 0:
#         return 0
        
#     frame_inicio = frames_uteis[0]
#     frame_fim = frames_uteis[-1]
    
#     # Converte o índice do frame de volta para tempo em segundos
#     tempo_inicio = (frame_inicio * hop_length) / sr
#     tempo_fim = ((frame_fim * hop_length) + frame_length) / sr
    
#     return max(0.1, tempo_fim - tempo_inicio)

# # Dicionário para armazenar os tempos extraídos de cada arquivo
# duracoes_por_classe = {classe: [] for classe in CLASSES}

# print("🔄 Iniciando a varredura e análise estatística do seu dataset...")

# # Varre a pasta de origem refletindo a estrutura de pastas do seu projeto
# for aluno in os.listdir(ORIGEM):
#     caminho_aluno = os.path.join(ORIGEM, aluno)
#     if not os.path.isdir(caminho_aluno):
#         continue

#     for classe in CLASSES:
#         caminho_classe = os.path.join(caminho_aluno, classe)
#         if not os.path.exists(caminho_classe):
#             continue

#         for arquivo in os.listdir(caminho_classe):
#             if not arquivo.lower().endswith(EXTENSOES):
#                 continue
                
#             caminho_arquivo = os.path.join(caminho_classe, arquivo)
#             try:
#                 # Leitura do arquivo
#                 audio, sr = sf.read(caminho_arquivo)
                
#                 # Conversão para mono se necessário
#                 if len(audio.shape) > 1:
#                     audio = np.mean(audio, axis=1)
                
#                 # Calcula a duração útil real da palavra falada
#                 t_util = calcular_duracao_util_rms(audio, sr)
                
#                 if t_util > 0.15:  # Filtra arquivos inválidos ou curtíssimos (ruídos)
#                     duracoes_por_classe[classe].append(t_util)
#             except Exception as e:
#                 print(f"⚠️ Erro ao processar {arquivo}: {e}")

# # --- ETAPA DE GERAÇÃO E EXIBIÇÃO DA VISUALIZAÇÃO ---
# todas_duracoes = []
# for tempos in duracoes_por_classe.values():
#     todas_duracoes.extend(tempos)

# if len(todas_duracoes) == 0:
#     print("❌ Nenhum áudio válido foi analisado. Verifique a estrutura das suas pastas.")
# else:
#     # Estatísticas Chave
#     percentil_95 = np.percentile(todas_duracoes, 95)
#     percentil_99 = np.percentile(todas_duracoes, 99)
#     media_geral = np.mean(todas_duracoes)
    
#     # Configuração visual do gráfico
#     plt.figure(figsize=(11, 6))
    
#     # Cores distintas para diferenciar as 5 classes
#     cores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
#     for i, (classe, tempos) in enumerate(duracoes_por_classe.items()):
#         if tempos:
#             plt.hist(tempos, bins=15, alpha=0.6, label=f'{classe} (Média: {np.mean(tempos):.2f}s)', 
#                      color=cores[i], edgecolor='black', linewidth=0.5)

#     # Adicionando referências visuais de tempo no gráfico
#     plt.axvline(percentil_95, color='#e74c3c', linestyle='--', linewidth=2.5, 
#                 label=f'Percentil 95 (Sugestão Ideal: {percentil_95:.2f}s)')
#     plt.axvline(1.2, color='#7f8c8d', linestyle=':', linewidth=2.5, 
#                 label='Seu Cenário Atual (1.20s)')
    
#     # Ajustes finos de layout
#     plt.title("Análise Estatística Real: Tempo de Informação Útil por Classe", fontsize=14, fontweight='bold', pad=12)
#     plt.xlabel("Tempo (segundos)", fontsize=11)
#     plt.ylabel("Frequência (Quantidade de Áudios)", fontsize=11)
#     plt.grid(True, linestyle='--', alpha=0.4)
#     plt.xlim(0.1, max(1.3, np.max(todas_duracoes) + 0.1))
    
#     # Caixa informativa flutuante
#     texto_resumo = f"Média Geral: {media_geral:.2f}s\nP95 (Ideal): {percentil_95:.2f}s\nP99 (Seguro): {percentil_99:.2f}s"
#     plt.gca().text(0.05, 0.95, texto_resumo, transform=plt.gca().transAxes, fontsize=10,
#                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.3))
    
#     plt.legend(loc='upper right', frameon=True, fontsize=10)
#     plt.tight_layout()
    
#     # Exibe a janela gráfica na tela
#     plt.show()
    
#     print("\n📊 --- RELATÓRIO FINAL DA ANÁLISE COMPUTAÇÃO ---")
#     print(f"-> Tempo médio em que os alunos concluem as palavras: {media_geral:.2f} segundos.")
#     print(f"-> 95% das amostras duram MENOS que: {percentil_95:.2f} segundos. (Ponto ideal de fatiamento)")
#     print(f"-> 99% das amostras duram MENOS que: {percentil_99:.2f} segundos.")
#     if percentil_95 < 1.1:
#         print(f"-> 💡 Confirmado: Reduzindo a DURACAO para {max(1.0, round(percentil_95, 1))}s você poupará processamento!")

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

BASE_DIR = r"C:\Users\jefte\projetos em python\ufc 2025 a 2026\aprendizado de maquina\projeto de reconhecimento de voz AMRP"
ORIGEM = os.path.join(BASE_DIR, "dataset_vozes_old")
CLASSES = ["direita", "esquerda", "siga", "pare", "voltar"]
EXTENSOES = (".wav", ".flac", ".ogg", ".m4a", ".mp3")
SAMPLE_RATE = 16000

def calcular_duracao_util_rms(audio, sr, frame_length=256, hop_length=128, threshold_db=-35):
    if len(audio) == 0:
        return 0
        
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
    num_frames = int(np.floor((len(audio) - frame_length) / hop_length) + 1)
    if num_frames <= 0:
        return len(audio) / sr
        
    frames_energia = []
    for f in range(num_frames):
        inicio = f * hop_length
        fim = inicio + frame_length
        quadrado = audio[inicio:fim] ** 2
        rms = np.sqrt(np.mean(quadrado))
        frames_energia.append(rms)
        
    frames_energia = np.array(frames_energia)
    frames_db = 20 * np.log10(frames_energia + 1e-8)
    
    indices_uteis = np.where(frames_db > threshold_db)[0]
    if len(indices_uteis) == 0:
        return 0
        
    frame_inicio = indices_uteis[0]
    frame_fim = indices_uteis[-1]
    
    amostra_inicio = frame_inicio * hop_length
    amostra_fim = (frame_fim * hop_length) + frame_length
    
    duracao = (amostra_fim - amostra_inicio) / sr
    return max(0.1, duracao)

duracoes_por_classe = {c: [] for c in CLASSES}
todas_duracoes = []

for raiz, diretorios, arquivos in os.walk(ORIGEM):
    for arquivo in arquivos:
        if arquivo.lower().endswith(EXTENSOES):
            caminho_completo = os.path.join(raiz, arquivo)
            nome_classe = None
            
            for c in CLASSES:
                if c in caminho_completo.lower() or c in arquivo.lower():
                    nome_classe = c
                    break
                    
            if nome_classe is None:
                continue
                
            try:
                audio, sr = librosa.load(caminho_completo, sr=SAMPLE_RATE)
                
                duracao_u = calcular_duracao_util_rms(audio, sr)
                if duracao_u > 0:
                    duracoes_por_classe[nome_classe].append(duracao_u)
                    todas_duracoes.append(duracao_u)
            except Exception as e:
                print(f"Erro ao processar {arquivo}: {e}")

if len(todas_duracoes) == 0:
    print("Nenhum audio foi processado.")
    exit()

media_geral = np.mean(todas_duracoes)
percentil_95 = np.percentile(todas_duracoes, 95)
percentil_99 = np.percentile(todas_duracoes, 99)

print("\n" + "="*45)
print(" RESULTADOS ESTATISTICOS DO DATASET")
print("="*45)
print(f"Total de audios analisados: {len(todas_duracoes)}")
print(f"Tempo medio de fala util: {media_geral:.2f}s")
print(f"Percentil 95 (Recomendado): {percentil_95:.2f}s")
print(f"Percentil 99 (Seguranca maxima): {percentil_99:.2f}s")
print("="*45 + "\n")

plt.figure(figsize=(10, 6))

cores = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6']
for idx, c in enumerate(CLASSES):
    if len(duracoes_por_classe[c]) > 0:
        plt.hist(duracoes_por_classe[c], bins=15, alpha=0.5, 
                 label=f'{c.upper()} (n={len(duracoes_por_classe[c])})', color=cores[idx])

plt.axvline(media_geral, color='#2c3e50', linestyle='--', linewidth=2, 
            label=f'Media Geral ({media_geral:.2f}s)')
plt.axvline(percentil_95, color='#e67e22', linestyle='-', linewidth=2.5, 
            label=f'Corte P95 ({percentil_95:.2f}s)')
plt.axvline(percentil_99, color='#c0392b', linestyle='-.', linewidth=2.5, 
            label=f'Corte P99 ({percentil_99:.2f}s)')
plt.axvline(1.20, color='#7f8c8d', linestyle=':', linewidth=2.5, 
            label='Seu Cenario Atual (1.20s)')

plt.title("Analise Estatistica Real: Tempo de Informacao Util por Classe", fontsize=14, fontweight='bold', pad=12)
plt.xlabel("Tempo (segundos)", fontsize=11)
plt.ylabel("Frequencia (Quantidade de Audios)", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.4)
plt.xlim(0.1, max(1.3, np.max(todas_duracoes) + 0.1))

texto_resumo = f"Media Geral: {media_geral:.2f}s\nP95 (Ideal): {percentil_95:.2f}s\nP99 (Seguro): {percentil_99:.2f}s"
plt.gca().text(0.05, 0.95, texto_resumo, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.3))

plt.legend(loc='upper right', frameon=True, fontsize=10)
plt.tight_layout()

plt.savefig("analise_limites_audio.png", dpi=300)
plt.show()