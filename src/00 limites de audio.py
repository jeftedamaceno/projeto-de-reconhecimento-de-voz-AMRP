import os
import wave
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuração de caminhos usando Path (evita problemas com acentos)
BASE_DIR = Path(r"C:\Users\jefte\projetos em python\ufc 2025 a 2026\aprendizado de maquina\projeto de reconhecimento de voz AMRP")
ORIGEM = BASE_DIR / "dataset_vozes_old"

CLASSES = ["direita", "esquerda", "siga", "pare", "voltar"]
EXTENSOES = (".wav", ".flac", ".ogg", ".m4a", ".mp3")
SAMPLE_RATE = 16000

# Caminho para o FFmpeg idêntico ao seu script de tratamento
FFMPEG_PATH = BASE_DIR / "ffmpeg-8.1-essentials_build" / "bin" / "ffmpeg.exe"

def converter_para_wav_temp(caminho_entrada):
    """Converte formatos m4a, mp3, ogg para um WAV temporário em 16kHz Mono usando FFmpeg"""
    caminho_saida = caminho_entrada.with_name(caminho_entrada.name + "_temp.wav")
    comando = [
        str(FFMPEG_PATH), "-y",
        "-i", str(caminho_entrada),
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-loglevel", "error",
        str(caminho_saida)
    ]
    result = subprocess.run(comando, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return caminho_saida

def carregar_wav_manual(file_path):
    with wave.open(str(file_path), 'rb') as wav_file:
        n_channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        sr = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        raw_data = wav_file.readframes(n_frames)
        
        if sampwidth == 2:
            audio_data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 1:
            audio_data = (np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError("Suporta apenas arquivos WAV de 8-bit ou 16-bit PCM.")
            
        if n_channels > 1:
            audio_data = audio_data.reshape(-1, n_channels).mean(axis=1)
            
        if sr != SAMPLE_RATE:
            num_provisorio = int(len(audio_data) * SAMPLE_RATE / sr)
            audio_data = np.interp(np.linspace(0, len(audio_data), num_provisorio), np.arange(len(audio_data)), audio_data)
            
        return audio_data

def calcular_duracao_util_rms(audio, sr, frame_length=256, hop_length=128, threshold_db=-35):
    if len(audio) == 0:
        return 0
        
    max_val = np.max(np.abs(audio))
    audio = audio / (max_val + 1e-8)
    
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

if not ORIGEM.is_dir():
    print(f"Diretorio raiz nao encontrado: {ORIGEM}")
    exit()

# Varre a pasta de origem usando Path.iterdir() que preserva acentuações perfeitamente
for entrada_aluno in ORIGEM.iterdir():
    if entrada_aluno.is_dir():
        
        for nome_classe in CLASSES:
            caminho_classe = entrada_aluno / nome_classe
            
            if caminho_classe.is_dir():
                for entrada_arquivo in caminho_classe.iterdir():
                    if entrada_arquivo.is_file() and entrada_arquivo.suffix.lower() in EXTENSOES:
                        
                        caminho_processar = entrada_arquivo
                        arquivo_temporario = None
                        
                        try:
                            # Se não for WAV nativo, converte via FFmpeg antes de analisar
                            if entrada_arquivo.suffix.lower() != ".wav":
                                arquivo_temporario = converter_para_wav_temp(entrada_arquivo)
                                if arquivo_temporario is None:
                                    continue
                                caminho_processar = arquivo_temporario
                            
                            audio = carregar_wav_manual(caminho_processar)
                            duracao_u = calcular_duracao_util_rms(audio, SAMPLE_RATE)
                            
                            if duracao_u > 0:
                                duracoes_por_classe[nome_classe].append(duracao_u)
                                todas_duracoes.append(duracao_u)
                                
                        except Exception as e:
                            continue
                        finally:
                            # Garante a exclusão do arquivo temporário convertido
                            if arquivo_temporario and arquivo_temporario.exists():
                                try:
                                    arquivo_temporario.unlink()
                                except:
                                    pass

if len(todas_duracoes) == 0:
    print("Nenhum audio foi processado. Verifique a estrutura das pastas.")
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

plt.title("Análise Estatística Real: Tempo de Informação Útil por Classe", fontsize=14, fontweight='bold', pad=12)
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