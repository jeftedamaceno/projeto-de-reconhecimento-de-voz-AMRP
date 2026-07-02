import os
import numpy as np
import soundfile as sf

PASTA_ENTRADA = "ruido_raw"
DESTINO = "dataset_final/ruido"

SAMPLE_RATE = 16000
DURACAO = 1.5
PASSO = 0.6

EXTENSOES = (".wav", ".ogg", ".flac")

os.makedirs(DESTINO, exist_ok=True)

def energia(signal):
    return np.convolve(signal**2, np.ones(400)/400, mode='same')

def ajustar(audio, sr):
    tamanho = int(DURACAO * sr)
    if len(audio) > tamanho:
        return audio[:tamanho]
    return np.pad(audio, (0, tamanho - len(audio)))

def reamostrar(audio, sr_origem, sr_destino):
    if sr_origem == sr_destino:
        return audio
    tempo = np.linspace(0, len(audio)/sr_origem, int(len(audio)*sr_destino/sr_origem))
    return np.interp(tempo, np.linspace(0, len(audio)/sr_origem, len(audio)), audio)

contador = 0

for arquivo in os.listdir(PASTA_ENTRADA):

    if not arquivo.lower().endswith(EXTENSOES):
        continue

    caminho = os.path.join(PASTA_ENTRADA, arquivo)

    try:
        audio, sr = sf.read(caminho)

        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        audio = reamostrar(audio, sr, SAMPLE_RATE)

        duracao = len(audio) / SAMPLE_RATE

        janela = int(DURACAO * SAMPLE_RATE)
        passo = int(PASSO * SAMPLE_RATE)

        if duracao > 3.0:
            for i in range(0, len(audio) - janela, passo):
                trecho = audio[i:i+janela]

                env = energia(trecho)
                if np.max(env) < 0.001:
                    continue

                trecho = trecho / (np.max(np.abs(trecho)) + 1e-6)
                trecho = ajustar(trecho, SAMPLE_RATE)

                nome = f"ruido_{contador}.wav"
                sf.write(os.path.join(DESTINO, nome), trecho, SAMPLE_RATE)

                contador += 1

        else:
            env = energia(audio)
            if np.max(env) < 0.001:
                continue

            audio = audio / (np.max(np.abs(audio)) + 1e-6)
            audio = ajustar(audio, SAMPLE_RATE)

            nome = f"ruido_{contador}.wav"
            sf.write(os.path.join(DESTINO, nome), audio, SAMPLE_RATE)

            contador += 1

    except Exception as e:
        print(f"Erro em {caminho}: {e}")

print(f"{contador} arquivos de ruido criados")