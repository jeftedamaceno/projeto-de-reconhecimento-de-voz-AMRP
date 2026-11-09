import os
import subprocess
from collections import Counter
import numpy as np
import soundfile as sf

# Tenta importar a função de pré-processamento do seu módulo local
try:
    from utils import preprocess_audio
except ImportError:

    def preprocess_audio(audio, target_length, training=False):
        if len(audio) > target_length:
            return audio[:target_length]
        return np.pad(audio, (0, target_length - len(audio)))


# --- CONFIGURAÇÕES DE CAMINHO E PARÂMETROS ---
BASE_DIR = r"C:\Users\jefte\projetos em python\ufc 2025 a 2026\aprendizado de maquina\projeto de reconhecimento de voz AMRP"

ORIGEM_VOZ = os.path.join(BASE_DIR, "dataset_340")
ORIGEM_RUIDO = os.path.join(BASE_DIR, "ruido_raw")
DESTINO = os.path.join(BASE_DIR, "dataset_processado_340")

CLASSES = ["direita", "esquerda", "siga", "pare", "voltar"]

# --- LISTA AMPLIADA COM TODAS AS EXTENSÕES ENCONTRADAS ---
EXTENSOES_PERMITIDAS = (
    ".wav",
    ".flac",
    ".ogg",
    ".m4a",
    ".mp3",
    ".mp4",
    ".oga",
    ".ogx",
    ".opus",
    ".webm",
)

SAMPLE_RATE = 16000
DURACAO = 1.5
TARGET_LENGTH = int(DURACAO * SAMPLE_RATE)
PASSO_RUIDO = 0.6

FFMPEG_PATH = r"ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"

# Métricas globais de relatório
extensoes_encontradas = Counter()
arquivos_nao_convertidos = []


# --- FUNÇÕES AUXILIARES ---
def energia(signal):
    return np.convolve(signal**2, np.ones(400) / 400, mode="same")


def converter_audio(entrada):
    """Converte qualquer formato suportado pelo FFmpeg (incluindo vídeo/containers como .mp4/.webm)

    para um arquivo .wav temporário em mono e 16kHz.
    """
    saida = entrada + "_temp.wav"
    comando = [
        FFMPEG_PATH,
        "-y",
        "-i",
        entrada,
        "-vn",  # Ignora qualquer fluxo de vídeo (útil para mp4 e webm)
        "-ac",
        "1",  # Mono
        "-ar",
        str(SAMPLE_RATE),  # 16000 Hz
        "-loglevel",
        "error",
        saida,
    ]
    result = subprocess.run(comando, capture_output=True, text=True)
    if result.returncode != 0:
        return None, result.stderr
    return saida, None


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


# --- ETAPA 1: PROCESSAR COMANDOS DE VOZ (dataset_340) ---
print("=== Processando comandos de voz ===")
os.makedirs(DESTINO, exist_ok=True)
for classe in CLASSES:
    os.makedirs(os.path.join(DESTINO, classe), exist_ok=True)

contador_voz = {classe: 0 for classe in CLASSES}

for classe in CLASSES:
    caminho_classe = os.path.join(ORIGEM_VOZ, classe)

    if not os.path.exists(caminho_classe):
        print(
            f"⚠️ Pasta da classe '{classe}' não encontrada em: {caminho_classe}"
        )
        continue

    for arquivo in os.listdir(caminho_classe):
        ext = os.path.splitext(arquivo)[1].lower()
        extensoes_encontradas[ext if ext else "sem_extensao"] += 1

        caminho_arquivo = os.path.join(caminho_classe, arquivo)

        if not arquivo.lower().endswith(EXTENSOES_PERMITIDAS):
            arquivos_nao_convertidos.append(
                (caminho_arquivo, "Extensão não listada nas permitidas")
            )
            continue

        temp_file = None
        try:
            # Força a conversão via FFmpeg para todos os formatos exceto .wav
            if not arquivo.lower().endswith(".wav"):
                temp_file, err = converter_audio(caminho_arquivo)
                if temp_file is None:
                    arquivos_nao_convertidos.append(
                        (caminho_arquivo, f"Erro no FFmpeg: {err.strip()}")
                    )
                    continue
                caminho_leitura = temp_file
            else:
                caminho_leitura = caminho_arquivo

            audio, sr = sf.read(caminho_leitura)

            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            audio = reamostrar(audio, sr, SAMPLE_RATE)
            audio = preprocess_audio(
                audio, target_length=TARGET_LENGTH, training=False
            )

            nome_saida = f"{classe}_{contador_voz[classe]}.wav"
            destino_final = os.path.join(DESTINO, classe, nome_saida)

            sf.write(destino_final, audio, SAMPLE_RATE)
            contador_voz[classe] += 1

        except Exception as e:
            arquivos_nao_convertidos.append((caminho_arquivo, str(e)))
        finally:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)

# --- ETAPA 2: PROCESSAR RUÍDOS (ruido_raw) ---
print("\n=== Processando áudios de ruído ===")
destino_ruido = os.path.join(DESTINO, "ruido")
os.makedirs(destino_ruido, exist_ok=True)
contador_ruido = 0

if os.path.exists(ORIGEM_RUIDO):
    for arquivo in os.listdir(ORIGEM_RUIDO):
        ext = os.path.splitext(arquivo)[1].lower()
        extensoes_encontradas[ext if ext else "sem_extensao"] += 1

        caminho_arquivo = os.path.join(ORIGEM_RUIDO, arquivo)

        if not arquivo.lower().endswith(EXTENSOES_PERMITIDAS):
            arquivos_nao_convertidos.append(
                (caminho_arquivo, "Extensão não listada nas permitidas")
            )
            continue

        temp_file = None
        try:
            if not arquivo.lower().endswith(".wav"):
                temp_file, err = converter_audio(caminho_arquivo)
                if temp_file is None:
                    arquivos_nao_convertidos.append(
                        (caminho_arquivo, f"Erro no FFmpeg: {err.strip()}")
                    )
                    continue
                caminho_leitura = temp_file
            else:
                caminho_leitura = caminho_arquivo

            audio, sr = sf.read(caminho_leitura)

            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            audio = reamostrar(audio, sr, SAMPLE_RATE)
            duracao_audio = len(audio) / SAMPLE_RATE

            janela = TARGET_LENGTH
            passo = int(PASSO_RUIDO * SAMPLE_RATE)

            if duracao_audio > 3.0:
                for i in range(0, len(audio) - janela, passo):
                    trecho = audio[i : i + janela]
                    if np.max(energia(trecho)) < 0.001:
                        continue

                    trecho = trecho / (np.max(np.abs(trecho)) + 1e-6)
                    trecho = preprocess_audio(
                        trecho, target_length=TARGET_LENGTH, training=False
                    )

                    nome_saida = f"ruido_{contador_ruido}.wav"
                    sf.write(
                        os.path.join(destino_ruido, nome_saida),
                        trecho,
                        SAMPLE_RATE,
                    )
                    contador_ruido += 1
            else:
                if np.max(energia(audio)) >= 0.001:
                    audio = audio / (np.max(np.abs(audio)) + 1e-6)
                    audio = preprocess_audio(
                        audio, target_length=TARGET_LENGTH, training=False
                    )

                    nome_saida = f"ruido_{contador_ruido}.wav"
                    sf.write(
                        os.path.join(destino_ruido, nome_saida),
                        audio,
                        SAMPLE_RATE,
                    )
                    contador_ruido += 1

        except Exception as e:
            arquivos_nao_convertidos.append((caminho_arquivo, str(e)))
        finally:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
else:
    print(
        f"⚠️ Diretório de ruídos não encontrado em: {ORIGEM_RUIDO}. Etapa ignorada."
    )


# --- EXIBIÇÃO DOS RESULTADOS E RELATÓRIO FINAL ---
print("\n" + "=" * 50)
print("             RELATÓRIO DE PROCESSAMENTO            ")
print("=" * 50)

print("\n📊 Resumo de Arquivos Criados por Classe:")
for classe, qtd in contador_voz.items():
    print(f"  - {classe}: {qtd} arquivos")
print(f"  - ruido: {contador_ruido} arquivos")

print("\n📂 Extensões Encontradas na Leitura:")
for ext, qtd in extensoes_encontradas.items():
    print(f"  - {ext}: {qtd} arquivo(s)")

print(
    f"\n❌ Arquivos que NÃO PUDERAM ser convertidos ({len(arquivos_nao_convertidos)}):"
)
if arquivos_nao_convertidos:
    for caminho, motivo in arquivos_nao_convertidos:
        print(f"  - [{motivo}] -> {caminho}")
else:
    print("  Nenhum arquivo apresentou erro.")