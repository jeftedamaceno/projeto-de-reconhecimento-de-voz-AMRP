import os
import numpy as np
import soundfile as sf


from utils import add_noise, random_shift, random_gain, standardize_audio

INPUT_DIR = "dataset_final"
OUTPUT_DIR = "dataset_ruido"
SAMPLE_RATE = 16000

os.makedirs(OUTPUT_DIR, exist_ok=True)

def augment_audio_manual(audio):
    audios = []

    audios.append(audio)


    audios.append(add_noise(audio, noise_level=0.003))

    audios.append(random_shift(audio, shift_max=0.1))

    audios.append(random_gain(audio))

    return audios


for label in os.listdir(INPUT_DIR):
    input_label_path = os.path.join(INPUT_DIR, label)

    if not os.path.isdir(input_label_path):
        continue

    output_label_path = os.path.join(OUTPUT_DIR, label)
    os.makedirs(output_label_path, exist_ok=True)

    contador = 0

    for file in os.listdir(input_label_path):
        if not file.lower().endswith(('.wav', '.flac')):
            continue
            
        file_path = os.path.join(input_label_path, file)

        try:
            audio, sr = sf.read(file_path)

            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            audio = standardize_audio(audio)

            versoes = augment_audio_manual(audio)

            for i, aug_audio in enumerate(versoes):
                aug_audio = np.clip(aug_audio, -1.0, 1.0)
                
                nome = f"{os.path.splitext(file)[0]}_aug_{i}.wav"
                caminho_saida = os.path.join(output_label_path, nome)

          
                sf.write(caminho_saida, aug_audio, SAMPLE_RATE)
                contador += 1

        except Exception as e:
            print(f"Erro em {file_path}: {e}")

    print(f"Classe [{label}]: {contador} arquivos gerados ao todo.")

print("\n🎉 Novo dataset aumentado criado com sucesso")