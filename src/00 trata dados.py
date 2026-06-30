# import os
# import numpy as np
# import soundfile as sf
# import subprocess

# # Importando o novo pipeline do seu utils
# from utils import preprocess_audio

# BASE_DIR = r"C:\Users\jefte\projetos em python\ufc 2025 a 2026\aprendizado de maquina\projeto de reconhecimento de voz AMRP"
# ORIGEM = os.path.join(BASE_DIR, "dataset_vozes_old")
# DESTINO = os.path.join(BASE_DIR, "dataset_final")

# CLASSES = ["direita", "esquerda", "siga", "pare", "voltar"]
# EXTENSOES = (".wav", ".flac", ".ogg", ".m4a", ".mp3")

# SAMPLE_RATE = 16000
# DURACAO = 1.2
# # Calculando o número de amostras exato para o target_length (1.2s * 16000Hz = 19200 amostras)
# TARGET_LENGTH = int(DURACAO * SAMPLE_RATE) 

# FFMPEG_PATH = r"ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"

# def converter_audio(entrada):
#     saida = entrada + "_temp.wav"
#     comando = [
#         FFMPEG_PATH, "-y",
#         "-i", entrada,
#         "-ac", "1",
#         "-ar", str(SAMPLE_RATE),
#         "-loglevel", "error",
#         saida
#     ]
#     result = subprocess.run(comando, capture_output=True, text=True)
#     if result.returncode != 0:
#         print("Erro FFmpeg:", result.stderr)
#         return None
#     return saida

# def reamostrar(audio, sr_origem, sr_destino):
#     if sr_origem == sr_destino:
#         return audio
#     duracao = len(audio) / sr_origem
#     novo_tamanho = int(duracao * sr_destino)
#     indices = np.linspace(0, len(audio) - 1, novo_tamanho)
#     indices_int = np.floor(indices).astype(int)
#     frac = indices - indices_int
#     indices_int2 = np.clip(indices_int + 1, 0, len(audio) - 1)
#     return (1 - frac) * audio[indices_int] + frac * audio[indices_int2]


# # Criar pastas de destino
# os.makedirs(DESTINO, exist_ok=True)
# for classe in CLASSES:
#     os.makedirs(os.path.join(DESTINO, classe), exist_ok=True)

# contador = {classe: 0 for classe in CLASSES}

# # Varre a pasta de origem (Alunos)
# for aluno in os.listdir(ORIGEM):
#     caminho_aluno = os.path.join(ORIGEM, aluno)

#     if not os.path.isdir(caminho_aluno):
#         continue

#     # Varre as classes dentro do aluno
#     for classe in CLASSES:
#         caminho_classe = os.path.join(caminho_aluno, classe)

#         if not os.path.exists(caminho_classe):
#             continue

#         for arquivo in os.listdir(caminho_classe):
#             if not arquivo.lower().endswith(EXTENSOES):
#                 continue

#             caminho_arquivo = os.path.join(caminho_classe, arquivo)
#             temp_file = None

#             try:
#                 # 1. Conversão se não for WAV
#                 if not arquivo.lower().endswith(".wav"):
#                     temp_file = converter_audio(caminho_arquivo)
#                     if temp_file is None:
#                         continue
#                     caminho_arquivo = temp_file

#                 # 2. Leitura do arquivo
#                 audio, sr = sf.read(caminho_arquivo)

#                 # 3. Transforma em Mono se for Estéreo
#                 if len(audio.shape) > 1:
#                     audio = np.mean(audio, axis=1)

#                 # 4. Garante a taxa de amostragem correta (16kHz) antes do preprocessamento
#                 audio = reamostrar(audio, sr, SAMPLE_RATE)

#                 # 5. Todo o processamento novo centralizado aqui!
#                 # Mude training=True se quiser ativar a simulação de microfone no dataset
#                 audio = preprocess_audio(audio, target_length=TARGET_LENGTH, training=False)

#                 # 6. Salvar arquivo final
#                 nome = f"{classe}_{contador[classe]}.wav"
#                 destino_final = os.path.join(DESTINO, classe, nome)
                
#                 sf.write(destino_final, audio, SAMPLE_RATE)
#                 print(f"✅ Processado com sucesso: {nome}")
                
#                 contador[classe] += 1

#             except Exception as e:
#                 print(f"❌ Erro em {caminho_arquivo}: {e}")

#             finally:
#                 if temp_file and os.path.exists(temp_file):
#                     os.remove(temp_file)

# print("\n🎉 dataset_final criado e padronizado com sucesso!")

import os
import numpy as np
import soundfile as sf
import subprocess

# Importando o novo pipeline do seu utils
from utils import preprocess_audio

BASE_DIR = r"C:\Users\jefte\projetos em python\ufc 2025 a 2026\aprendizado de maquina\projeto de reconhecimento de voz AMRP"
ORIGEM = os.path.join(BASE_DIR, "dataset_vozes_old")
DESTINO = os.path.join(BASE_DIR, "dataset_final")

CLASSES = ["direita", "esquerda", "siga", "pare", "voltar"]
EXTENSOES = (".wav", ".flac", ".ogg", ".m4a", ".mp3")

SAMPLE_RATE = 16000
# --- ALTERAÇÃO REALIZADA AQUI: Ajustado para o sweet spot ideal do seu dataset ---
DURACAO = 1.5
# Calculando o número de amostras exato para o target_length (1.5s * 16000Hz = 24000 amostras)
TARGET_LENGTH = int(DURACAO * SAMPLE_RATE) 

FFMPEG_PATH = r"ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"

def converter_audio(entrada):
    saida = entrada + "_temp.wav"
    comando = [
        FFMPEG_PATH, "-y",
        "-i", entrada,
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-loglevel", "error",
        saida
    ]
    result = subprocess.run(comando, capture_output=True, text=True)
    if result.returncode != 0:
        print("Erro FFmpeg:", result.stderr)
        return None
    return saida

def reamostrar(audio, sr_origem, sr_destino):
    if sr_origem == sr_destino:
        return audio
    duracao = len(audio) / sr_origem
    novo_tamanho = int(duracao * sr_destino)
    indices = np.linspace(0, len(audio) - 1, novo_tamanho)
    indices_int = np.floor(indices).astype(int)
    frac = indices - indices_int
    indices_int2 = np.clip(indices_int + 1, 0, len(audio) - 1)
    return (1 - frac) * audio[indices_int] + frac * audio[indices_int2]


# Criar pastas de destino
os.makedirs(DESTINO, exist_ok=True)
for classe in CLASSES:
    os.makedirs(os.path.join(DESTINO, classe), exist_ok=True)

contador = {classe: 0 for classe in CLASSES}

# Varre a pasta de origem (Alunos)
for aluno in os.listdir(ORIGEM):
    caminho_aluno = os.path.join(ORIGEM, aluno)

    if not os.path.isdir(caminho_aluno):
        continue

    # Varre as classes dentro do aluno
    for classe in CLASSES:
        caminho_classe = os.path.join(caminho_aluno, classe)

        if not os.path.exists(caminho_classe):
            continue

        for arquivo in os.listdir(caminho_classe):
            if not arquivo.lower().endswith(EXTENSOES):
                continue

            caminho_arquivo = os.path.join(caminho_classe, arquivo)
            temp_file = None

            try:
                # 1. Conversão se não for WAV
                if not arquivo.lower().endswith(".wav"):
                    temp_file = converter_audio(caminho_arquivo)
                    if temp_file is None:
                        continue
                    caminho_arquivo = temp_file

                # 2. Leitura do arquivo
                audio, sr = sf.read(caminho_arquivo)

                # 3. Transforma em Mono se for Estéreo
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)

                # 4. Garante a taxa de amostragem correta (16kHz) antes do preprocessamento
                audio = reamostrar(audio, sr, SAMPLE_RATE)

                # 5. Todo o processamento novo centralizado aqui!
                # Mude training=True se quiser ativar a simulação de microfone no dataset
                audio = preprocess_audio(audio, target_length=TARGET_LENGTH, training=False)

                # 6. Salvar arquivo final
                nome = f"{classe}_{contador[classe]}.wav"
                destino_final = os.path.join(DESTINO, classe, nome)
                
                sf.write(destino_final, audio, SAMPLE_RATE)
                print(f"✅ Processado com sucesso: {nome}")
                
                contador[classe] += 1

            except Exception as e:
                print(f"❌ Erro em {caminho_arquivo}: {e}")

            finally:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)

print("\n🎉 dataset_final criado e padronizado com sucesso!")