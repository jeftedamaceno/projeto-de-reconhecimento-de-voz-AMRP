import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt

# ============================================================
# 1. CONFIGURAÇÕES E PARÂMETROS
# ============================================================
VIDEO_ENTRADA = "pontilhado2.mp4"
VIDEO_SAIDA = "resultado_3_metodos.mp4"
LARGURA_PROC, ALTURA_PROC = 128, 128

# Configuração de velocidade: 1 para processar TODOS os frames, 2 para pular de 2 em 2
PULAR_FRAMES = 1

LIMIAR_ESCURIDAO = 75
AREA_MINIMA = 25
LARGURA_MINIMA = 4
ALTURA_MINIMA = 6

CLASSES_CNN = [
    "Reta", "Curva_Aberta_Esq", "Curva_Aberta_Dir",
    "Curva_Fechada_Esq", "Curva_Fechada_Dir",
    "Bifurcacao_Esq_Dir", "Cruzamento_3Pistas",
    "Cruzamento_4Pistas", "Sem_Caminho"
]

# ============================================================
# 2. VERIFICAÇÃO E CARREGAMENTO DOS MODELOS
# ============================================================
model_cnn = None
model_hibrido = None

try:
    import tensorflow as tf
    
    # Modelo 1: Classificador CNN
    if os.path.exists("modelo_sintetico_comparacao.h5"):
        model_cnn = tf.keras.models.load_model("modelo_sintetico_comparacao.h5")
        print("[OK] Modelo M1 (CNN Classificador) encontrado e carregado.")
    else:
        print("[ALERTA] 'modelo_sintetico_comparacao.h5' não encontrado! M1 funcionará como N/A.")

    # Modelo 3: Segmentador Híbrido
    if os.path.exists("modelo_hibrido_segmentador.h5"):
        model_hibrido = tf.keras.models.load_model("modelo_hibrido_segmentador.h5")
        print("[OK] Modelo M3 (Segmentador Híbrido) encontrado e carregado.")
    else:
        print("[ALERTA] 'modelo_hibrido_segmentador.h5' não encontrado! Execute o treino do Híbrido primeiro.")

except Exception as e:
    print(f"[ERRO] Falha ao inicializar o TensorFlow: {e}")

# ============================================================
# 3. GEOMETRIA E SEGMENTAÇÃO BASE DA PISTA
# ============================================================
def calcular_geometria_roi(binaria_img):
    mascara_roi = np.zeros_like(binaria_img)
    pts_roi = np.array([
        [int(LARGURA_PROC * 0.20), int(ALTURA_PROC * 0.40)],
        [int(LARGURA_PROC * 0.80), int(ALTURA_PROC * 0.40)],
        [LARGURA_PROC, ALTURA_PROC],
        [0, ALTURA_PROC]
    ], dtype=np.int32)
    cv2.fillPoly(mascara_roi, [pts_roi], 255)

    bin_roi = cv2.bitwise_and(binaria_img, binaria_img, mask=mascara_roi)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_roi, connectivity=8)
    bin_filtrada = np.zeros_like(binaria_img)
    for i in range(1, num_labels):
        w, h, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if area >= AREA_MINIMA and w >= LARGURA_MINIMA and h >= ALTURA_MINIMA:
            bin_filtrada[labels == i] = 255

    fechada = cv2.morphologyEx(bin_filtrada, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # Região superior (fundo)
    sub_fundo = fechada[int(ALTURA_PROC * 0.40):int(ALTURA_PROC * 0.60), :]
    num_sub, _, stats_sub, centroides_sub = cv2.connectedComponentsWithStats(sub_fundo, connectivity=8)

    ramos_topo = []
    for i in range(1, num_sub):
        if stats_sub[i, cv2.CC_STAT_AREA] >= AREA_MINIMA:
            cx = int(centroides_sub[i][0])
            cy = int(centroides_sub[i][1]) + int(ALTURA_PROC * 0.40)
            ramos_topo.append((cx, cy))

    # Região inferior (próximo ao robô)
    sub_base = fechada[int(ALTURA_PROC * 0.75):ALTURA_PROC, :]
    M_base = cv2.moments(sub_base)

    if M_base["m00"] <= AREA_MINIMA or len(ramos_topo) == 0:
        return "Sem_Caminho", 0.0, fechada, None, []

    cx_base = int(M_base["m10"] / M_base["m00"])
    cy_base = int(M_base["m01"] / M_base["m00"]) + int(ALTURA_PROC * 0.75)
    pt_base = (cx_base, cy_base)

    angulos = []
    for pt in ramos_topo:
        dx = pt[0] - pt_base[0]
        dy = pt_base[1] - pt[1]
        ang = np.degrees(np.arctan2(dx, dy))
        angulos.append(ang)

    if len(ramos_topo) >= 3:
        classe = "Cruzamento_4Pistas"
    elif len(ramos_topo) == 2:
        if any(abs(a) <= 12 for a in angulos):
            classe = "Cruzamento_3Pistas"
        else:
            classe = "Bifurcacao_Esq_Dir"
    elif len(ramos_topo) == 1:
        ang = angulos[0]
        if -12 <= ang <= 12: classe = "Reta"
        elif -40 <= ang < -12: classe = "Curva_Aberta_Esq"
        elif 12 < ang <= 40: classe = "Curva_Aberta_Dir"
        elif ang < -40: classe = "Curva_Fechada_Esq"
        else: classe = "Curva_Fechada_Dir"
    else:
        classe = "Sem_Caminho"

    angulo_principal = angulos[0] if len(angulos) > 0 else 0.0
    return classe, angulo_principal, fechada, pt_base, ramos_topo

# ============================================================
# 4. EXECUÇÃO DOS MÉTODOS
# ============================================================
def executar_metodo_1(binaria_img):
    if model_cnn is None:
        return "N/A"
    
    if len(binaria_img.shape) == 2 or binaria_img.shape[2] == 1:
        binaria_3canais = cv2.cvtColor(binaria_img, cv2.COLOR_GRAY2BGR)
        binaria_rgb = cv2.cvtColor(binaria_3canais, cv2.COLOR_BGR2RGB)
    else:
        binaria_rgb = cv2.cvtColor(binaria_img, cv2.COLOR_BGR2RGB)
    
    inp = np.expand_dims(binaria_rgb.astype(np.float32), axis=0)
    pred = model_cnn.predict(inp, verbose=0)
    return CLASSES_CNN[np.argmax(pred[0])]

def executar_metodo_2(frame_bgr):
    cinza = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(cinza, LIMIAR_ESCURIDAO, 255, cv2.THRESH_BINARY_INV)
    return calcular_geometria_roi(binaria)

def executar_metodo_3(frame_rgb):
    if model_hibrido is None:
        return "N/A", 0.0, np.zeros((128, 128), dtype=np.uint8), None, []

    inp = np.expand_dims(frame_rgb.astype(np.float32) / 255.0, axis=0)
    pred_mask = model_hibrido.predict(inp, verbose=0)[0, :, :, 0]
    
    binaria_ia = ((pred_mask > 0.3).astype(np.uint8)) * 255
    return calcular_geometria_roi(binaria_ia)

# ============================================================
# 5. ABERTURA E GRAVAÇÃO DO VÍDEO (COMPATÍVEL COM WINDOWS)
# ============================================================
cap = cv2.VideoCapture(VIDEO_ENTRADA)
if not cap.isOpened():
    raise FileNotFoundError(f"Vídeo de entrada não encontrado: {VIDEO_ENTRADA}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 0:
    fps = 30.0

# Gravador usando Codec XVID totalmente compatível com Windows
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(VIDEO_SAIDA, fourcc, fps, (768, 256), True)

if not out.isOpened():
    print("[AVISO] Codec XVID indisponível. Alternando para MJPG...")
    VIDEO_SAIDA = "resultado_3_metodos.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(VIDEO_SAIDA, fourcc, fps, (768, 256), True)

tempos_m1, tempos_m2, tempos_m3 = [], [], []
angulos_m2, angulos_m3 = [], []

print(f"[INFO] Processando vídeo a {fps:.2f} FPS...")

contador_frame = 0

while True:
    ret, frame = cap.read()
    if not ret: 
        break

    contador_frame += 1
    if contador_frame % PULAR_FRAMES != 0:
        continue

    frame_red = cv2.resize(frame, (LARGURA_PROC, ALTURA_PROC))
    frame_rgb = cv2.cvtColor(frame_red, cv2.COLOR_BGR2RGB)

    # 1. Executa M2 (Lógica Geométrica)
    t0 = time.time()
    c_m2, a_m2, bin_m2, pt_base_m2, ramos_m2 = executar_metodo_2(frame_red)
    t_m2 = (time.time() - t0) * 1000
    tempos_m2.append(t_m2)
    angulos_m2.append(a_m2)

    # 2. Executa M1 (Passando a imagem binarizada do M2)
    t0 = time.time()
    c_m1 = executar_metodo_1(bin_m2)
    t_m1 = (time.time() - t0) * 1000
    tempos_m1.append(t_m1)

    # 3. Executa M3 (Híbrido)
    t0 = time.time()
    c_m3, a_m3, bin_m3, pt_base_m3, ramos_m3 = executar_metodo_3(frame_rgb)
    t_m3 = (time.time() - t0) * 1000
    tempos_m3.append(t_m3)
    angulos_m3.append(a_m3)

    # ---------------- MONTAGEM DOS PAINÉIS DE VÍDEO ----------------
    p1 = frame_red.copy()
    p2 = cv2.cvtColor(bin_m2, cv2.COLOR_GRAY2BGR)
    p3 = cv2.cvtColor(bin_m3, cv2.COLOR_GRAY2BGR)

    # Desenho no Painel 2 (Lógica)
    if pt_base_m2 is not None:
        cv2.circle(p2, pt_base_m2, 3, (0, 0, 255), -1)
        for pt in ramos_m2:
            cv2.circle(p2, pt, 3, (0, 255, 0), -1)
            cv2.line(p2, pt_base_m2, pt, (0, 255, 255), 2)

    # Desenho no Painel 3 (Híbrido)
    if pt_base_m3 is not None:
        cv2.circle(p3, pt_base_m3, 3, (0, 0, 255), -1)
        for pt in ramos_m3:
            cv2.circle(p3, pt, 3, (0, 255, 0), -1)
            cv2.line(p3, pt_base_m3, pt, (0, 255, 255), 2)

    # Legendas sobrepostas
    cv2.putText(p1, f"M1: {c_m1}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.putText(p2, f"M2: {c_m2} ({a_m2:.1f}d)", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    cv2.putText(p3, f"M3: {c_m3} ({a_m3:.1f}d)", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 100, 255), 1)

    # Concatenação e gravação
    painel_ampliado = cv2.resize(np.hstack((p1, p2, p3)), (768, 256), interpolation=cv2.INTER_NEAREST)
    out.write(painel_ampliado)

cap.release()
out.release()

# ============================================================
# 6. GRÁFICOS COMPARATIVOS
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(tempos_m1, label=f"M1: CNN Classificacao ({np.mean(tempos_m1):.1f}ms)", color="green", alpha=0.7)
axes[0].plot(tempos_m2, label=f"M2: Logica Geometrica ({np.mean(tempos_m2):.1f}ms)", color="orange", alpha=0.8)
axes[0].plot(tempos_m3, label=f"M3: Hibrido ({np.mean(tempos_m3):.1f}ms)", color="purple", alpha=0.8)
axes[0].set_title("Desempenho - Tempo de Resposta por Frame (ms)")
axes[0].set_xlabel("Frame")
axes[0].set_ylabel("Tempo (ms)")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].plot(angulos_m2, label="M2: Ângulo Geométrico", color="orange", linestyle="--")
axes[1].plot(angulos_m3, label="M3: Ângulo Híbrido", color="purple")
axes[1].set_title("Estabilidade da Curvatura (Graus)")
axes[1].set_xlabel("Frame")
axes[1].set_ylabel("Ângulo (º)")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("grafico_comparativo_desempenho.png", dpi=300)
plt.show()

print("\n[PROCESSO FINALIZADO]")
print(f" - Vídeo gerado com sucesso: '{VIDEO_SAIDA}'")
print(" - Gráfico gerado: 'grafico_comparativo_desempenho.png'")