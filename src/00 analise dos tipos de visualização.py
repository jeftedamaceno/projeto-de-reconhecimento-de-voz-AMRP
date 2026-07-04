import os
import numpy as np
from numpy.lib.stride_tricks import as_strided
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.fftpack import dct
from scipy.signal import resample_poly
from math import gcd


def stft(
    y,
    n_fft=2048,
    hop_length=None,
    win_length=None,
    center=True,
    pad_mode="constant"
):

    y = np.asarray(y)

    if hop_length is None:
        hop_length = n_fft // 4

    if win_length is None:
        win_length = n_fft

    # Janela Hann periódica (igual ao scipy.signal.get_window('hann', ..., fftbins=True))
    window = np.hanning(win_length + 1)[:-1]

    # Centraliza a janela dentro do FFT
    if win_length < n_fft:
        left = (n_fft - win_length) // 2
        right = n_fft - win_length - left
        window = np.pad(window, (left, right))

    if center:
        pad = n_fft // 2
        y = np.pad(y, (pad, pad), mode=pad_mode)

    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))

    n_frames = 1 + (len(y) - n_fft) // hop_length
    shape = (n_fft, n_frames)
    strides = (y.strides[0], hop_length * y.strides[0])
    frames = as_strided(y, shape=shape, strides=strides)
    frames = frames * window[:, None]
    D = np.fft.rfft(frames, axis=0)
    return D

def amplitude_to_db(
    S,
    ref=1.0,
    amin=1e-5,
    top_db=80.0
):

    S = np.asarray(S)

    if np.iscomplexobj(S):
        magnitude = np.abs(S)
    else:
        magnitude = S

    if np.any(magnitude < 0):
        raise ValueError("amplitude_to_db() only accepts non-negative amplitudes.")

    if amin <= 0:
        raise ValueError("amin must be strictly positive.")

    # Valor de referência
    if callable(ref):
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    magnitude = np.maximum(amin, magnitude)
    ref_value = np.maximum(amin, ref_value)

    log_spec = 20.0 * np.log10(magnitude)
    log_spec -= 20.0 * np.log10(ref_value)

    if top_db is not None:
        if top_db < 0:
            raise ValueError("top_db must be non-negative.")
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec

def specshow(
    data,
    sr=22050,
    hop_length=512,
    n_fft=2048,
    x_axis="time",
    y_axis="hz",
    cmap=None,
    ax=None,
    shading="auto",
    **kwargs
):

    if ax is None:
        ax = plt.gca()

    n_freqs, n_frames = data.shape

    # eixo x
    if x_axis == "time":
        x = np.arange(n_frames + 1) * hop_length / sr
        ax.set_xlabel("Time (s)")
    else:
        x = np.arange(n_frames + 1)

    # eixo y
    if y_axis == "hz":
        y = np.linspace(0, sr / 2, n_freqs + 1)
        ax.set_ylabel("Hz")
    else:
        y = np.arange(n_freqs + 1)

    img = ax.pcolormesh(
        x,
        y,
        data,
        shading=shading,
        cmap=cmap,
        **kwargs
    )

    return img


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filter(
    sr=22050,
    n_fft=2048,
    n_mels=128,
    fmin=0.0,
    fmax=None,
    htk=False,
    norm="slaney",
):
    if fmax is None:
        fmax = sr / 2

    if htk:
        min_mel = hz_to_mel(fmin)
        max_mel = hz_to_mel(fmax)
    else:
        min_mel = hz_to_mel(fmin)
        max_mel = hz_to_mel(fmax)

    mel_points = np.linspace(min_mel, max_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)

    weights = np.zeros((n_mels, len(fft_freqs)), dtype=np.float32)

    for i in range(n_mels):
        lower = hz_points[i]
        center = hz_points[i + 1]
        upper = hz_points[i + 2]

        left = (fft_freqs - lower) / (center - lower)
        right = (upper - fft_freqs) / (upper - center)

        weights[i] = np.maximum(0, np.minimum(left, right))

    if norm == "slaney":
        enorm = 2.0 / (hz_points[2:n_mels + 2] - hz_points[:n_mels])
        weights *= enorm[:, np.newaxis]

    return weights


def melspectrogram(
    y=None,
    S=None,
    sr=22050,
    n_fft=2048,
    hop_length=512,
    win_length=None,
    window="hann",
    center=True,
    pad_mode="constant",
    power=2.0,
    n_mels=128,
    fmin=0.0,
    fmax=None,
    htk=False,
    norm="slaney",
):
    if S is None:
        D = stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            center=center,
            pad_mode=pad_mode,
        )
        S = np.abs(D) ** power

    mel_basis = mel_filter(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        htk=htk,
        norm=norm,
    )

    return np.dot(mel_basis, S)


def power_to_db(
    S,
    ref=1.0,
    amin=1e-10,
    top_db=80.0,
):
    S = np.asarray(S)

    if np.iscomplexobj(S):
        raise ValueError("power_to_db was called on complex input.")

    if np.any(S < 0):
        raise ValueError("power_to_db only accepts non-negative power values.")

    if amin <= 0:
        raise ValueError("amin must be strictly positive.")

    if callable(ref):
        ref_value = ref(S)
    else:
        ref_value = np.abs(ref)

    log_spec = 10.0 * np.log10(np.maximum(amin, S))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    if top_db is not None:
        if top_db < 0:
            raise ValueError("top_db must be non-negative.")
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


def mfcc(
    y=None,
    sr=22050,
    S=None,
    n_mfcc=20,
    dct_type=2,
    norm="ortho",
    lifter=0,
    mel_norm="slaney",
    **kwargs,
):
    if S is None:
        S = power_to_db(
            melspectrogram(
                y=y,
                sr=sr,
                norm=mel_norm,
                **kwargs,
            )
        )

    M = dct(S, axis=-2, type=dct_type, norm=norm)[..., :n_mfcc, :]

    if lifter > 0:
        LI = np.sin(np.pi * np.arange(1, 1 + n_mfcc) / lifter)
        LI = 1 + (lifter / 2.0) * LI
        M *= LI[:, np.newaxis]
    elif lifter == 0:
        pass
    else:
        raise ValueError("lifter must be non-negative")

    return M


def rms(
    y=None,
    S=None,
    frame_length=2048,
    hop_length=512,
    center=True,
    pad_mode="constant",
    dtype=np.float32,
):
    if y is not None:
        y = np.asarray(y)

        if center:
            pad = frame_length // 2
            y = np.pad(y, (pad, pad), mode=pad_mode)

        if len(y) < frame_length:
            y = np.pad(y, (0, frame_length - len(y)))

        n_frames = 1 + (len(y) - frame_length) // hop_length

        shape = (frame_length, n_frames)
        strides = (y.strides[0], hop_length * y.strides[0])

        frames = as_strided(y, shape=shape, strides=strides)

        power = np.mean(np.abs(frames) ** 2, axis=0)

        return np.sqrt(power, dtype=dtype)[np.newaxis, :]

    if S is not None:
        S = np.asarray(S)

        if S.shape[0] < 2:
            raise ValueError("Spectrogram must have at least two frequency bins.")

        x = np.abs(S) ** 2
        power = 2.0 * np.sum(x[1:-1], axis=0)

        power += x[0]
        if frame_length % 2 == 0:
            power += x[-1]

        power /= frame_length ** 2

        return np.sqrt(power, dtype=dtype)[np.newaxis, :]

    raise ValueError("Either 'y' or 'S' must be provided.")


def frames_to_time(
    frames,
    sr=22050,
    hop_length=512,
    n_fft=None,
):
    frames = np.asanyarray(frames)

    offset = 0
    if n_fft is not None:
        offset = n_fft // 2

    samples = frames * hop_length + offset

    return samples / float(sr)


def _parabolic_interpolation(S):
    S = np.asarray(S)

    shift = np.zeros_like(S, dtype=np.float32)

    avg = 0.5 * (S[2:] - S[:-2])
    den = S[2:] - 2.0 * S[1:-1] + S[:-2]

    valid = np.abs(den) > np.finfo(S.dtype).eps
    shift[1:-1][valid] = -avg[valid] / den[valid]

    np.clip(shift, -1.0, 1.0, out=shift)

    return shift


def piptrack(
    y=None,
    sr=22050,
    S=None,
    n_fft=2048,
    hop_length=None,
    fmin=150.0,
    fmax=4000.0,
    threshold=0.1,
    win_length=None,
    window="hann",
    center=True,
    pad_mode="constant",
    ref=None,
):
    if S is None:
        if y is None:
            raise ValueError("Either 'y' or 'S' must be provided.")

        D = stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            center=center,
            pad_mode=pad_mode,
        )

        S = np.abs(D)
    else:
        S = np.asarray(S)

    if S.ndim != 2:
        raise ValueError("Spectrogram must be two-dimensional.")

    if ref is None:
        ref_value = np.max
    else:
        ref_value = ref

    if callable(ref_value):
        ref_mag = ref_value(S, axis=0)
    else:
        ref_mag = np.asarray(ref_value)

    fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    freq_mask = np.logical_and(
        fft_freqs >= fmin,
        fft_freqs <= fmax
    )

    shifts = _parabolic_interpolation(S)

    pitches = np.zeros_like(S, dtype=np.float32)
    mags = np.zeros_like(S, dtype=S.dtype)

    idx = np.flatnonzero(freq_mask)

    if idx.size == 0:
        return pitches, mags

    start = max(idx[0], 1)
    stop = min(idx[-1] + 1, S.shape[0] - 1)

    for t in range(S.shape[1]):
        column = S[:, t]

        if np.ndim(ref_mag) == 0:
            threshold_value = threshold * ref_mag
        else:
            threshold_value = threshold * ref_mag[t]

        for i in range(start, stop):
            if (
                column[i] > threshold_value
                and column[i] > column[i - 1]
                and column[i] >= column[i + 1]
            ):
                mags[i, t] = column[i]
                pitches[i, t] = (
                    fft_freqs[i]
                    + shifts[i, t] * sr / n_fft
                )

    return pitches, mags


def resample(
    y,
    orig_sr,
    target_sr,
    res_type="soxr_hq",
    fix=True,
    scale=False,
    axis=-1,
    **kwargs,
):
    y = np.asarray(y)

    if orig_sr <= 0:
        raise ValueError("orig_sr must be a positive integer")

    if target_sr <= 0:
        raise ValueError("target_sr must be a positive integer")

    if orig_sr == target_sr:
        return y.copy()

    supported = {
        "soxr_vhq",
        "soxr_hq",
        "soxr_mq",
        "soxr_lq",
        "soxr_qq",
        "kaiser_best",
        "kaiser_fast",
        "polyphase",
    }

    if res_type not in supported:
        raise ValueError(f"Unsupported res_type: {res_type}")

    g = gcd(int(orig_sr), int(target_sr))
    up = target_sr // g
    down = orig_sr // g

    if res_type == "polyphase":
        y_hat = resample_poly(
            y,
            up,
            down,
            axis=axis,
            padtype="constant",
        )
    else:
        if res_type in (
            "soxr_vhq",
            "soxr_hq",
            "soxr_mq",
            "soxr_lq",
            "soxr_qq",
        ):
            window = ("kaiser", 5.0)
        elif res_type == "kaiser_best":
            window = ("kaiser", 14.769656459379492)
        else:
            window = ("kaiser", 8.555504641634386)

        y_hat = resample_poly(
            y,
            up,
            down,
            axis=axis,
            window=window,
            padtype="constant",
        )

    if fix:
        n_samples = int(np.ceil(y.shape[axis] * target_sr / orig_sr))

        current = y_hat.shape[axis]

        if current < n_samples:
            pad = [(0, 0)] * y_hat.ndim
            pad[axis] = (0, n_samples - current)
            y_hat = np.pad(y_hat, pad)

        elif current > n_samples:
            index = [slice(None)] * y_hat.ndim
            index[axis] = slice(0, n_samples)
            y_hat = y_hat[tuple(index)]

    if scale:
        y_hat /= np.sqrt(target_sr / orig_sr)

    return np.asarray(y_hat, dtype=y.dtype)


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
stft_linear = np.abs(stft(audio_ex))
stft_db = amplitude_to_db(stft_linear, ref=np.max)
plt.figure(figsize=(10, 4))
specshow(stft_db, sr=sr_ex, x_axis='time', y_axis='linear', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title("2. Domínio da Frequência (Espectrograma Linear Exato)", fontsize=12, fontweight='bold')
plt.xlabel("Tempo (segundos)")
plt.ylabel("Frequência (Hz)")
plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, "2_espectrograma_linear.png"), dpi=200)
plt.close()

# Visualização 3: Espectrograma Mel
mel_spec_ex = melspectrogram(y=audio_ex, sr=sr_ex, n_mels=128)
mel_db_ex = power_to_db(mel_spec_ex, ref=np.max)
plt.figure(figsize=(10, 4))
specshow(mel_db_ex, sr=sr_ex, x_axis='time', y_axis='mel', cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title("3. Espectrograma Mel (Frequências Mapeadas ao Ouvido Humano)", fontsize=12, fontweight='bold')
plt.xlabel("Tempo (segundos)")
plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, "3_espectrograma_mel.png"), dpi=200)
plt.close()

# Visualização 4: MFCCs
mfccs_ex = mfcc(y=audio_ex, sr=sr_ex, n_mfcc=13)
plt.figure(figsize=(10, 4))
specshow(mfccs_ex, sr=sr_ex, x_axis='time', cmap='coolwarm')
plt.colorbar()
plt.title("4. MFCCs (Filtro do Trato Vocal)", fontsize=12, fontweight='bold')
plt.xlabel("Tempo (segundos)")
plt.ylabel("Coeficientes MFCC")
plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, "4_mfccs.png"), dpi=200)
plt.close()

# Visualização 5: Cadência e Prosódia
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
rms_cadencia = rms(y=audio_ex)[0]
frames_tempo = frames_to_time(range(len(rms_cadencia)), sr=sr_ex)
ax1.plot(frames_tempo, rms_cadencia, color='#d62728', linewidth=2.5)
ax1.fill_between(frames_tempo, rms_cadencia, alpha=0.2, color='#d62728')
ax1.set_title("5. Análise de Cadência: Envelope de Volume (Acentuação e Ritmo)", fontsize=11, fontweight='bold')
ax1.set_ylabel("Energia RMS")
ax1.grid(True, alpha=0.3)

pitches, magnitudes = piptrack(y=audio_ex, sr=sr_ex)
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
            wave_feat = resample(audio, orig_sr=sr, target_sr=2000)[:3000]
            
            mel_spec = melspectrogram(y=audio, sr=sr, n_mels=128)
            mel_feat = np.mean(power_to_db(mel_spec, ref=np.max), axis=1)
            
            mfcc_spec = mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc_feat = np.mean(mfcc_spec, axis=1)
            
            rms_feat = rms(y=audio)[0]
            
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
