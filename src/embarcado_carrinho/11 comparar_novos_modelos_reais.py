import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from collections import deque

# ============================================================
# 1. CONFIGURAÇÕES E PARÂMETROS
# ============================================================
VIDEO_ENTRADA = "pontilhado2.mp4"
VIDEO_SAIDA = "resultado_comparacao_modelos_reais.mp4"
LARGURA_PROC, ALTURA_PROC = 128, 128

PULAR_FRAMES = 1 # 1 para processar todos os quadros

# Filtros antiruído
AREA_MINIMA = 15
LARGURA_MINIMA = 3
ALTURA_MINIMA = 4

# Histórico para média móvel de 5 frames (suavização angular)
historico_m2 = deque(maxlen=5)
historico_m3 = deque(maxlen=5)

CLASSES_CNN = [
    "Reta", "Curva_Aberta_Esq", "Curva_Aberta_Dir",
    "Curva_Fechada_Esq", "Curva_Fechada_Dir",
    "Bifurcacao_Esq_Dir", "Cruzamento_3Pistas",
    "Cruzamento_4Pistas", "Sem_Caminho"
]

# ============================================================
# 2. CARREGAMENTO DOS MODELOS ATUALIZADOS
# ============================================================
model_cnn = None
model_hibrido = None

try:
    import tensorflow as tf
    
    if os.path.exists("modelo_sintetico_comparacao.h5"):
        model_cnn = tf.keras.models.load_model("modelo_sintetico_comparacao.h5")
        print("[OK] Modelo M1 (CNN Classificador) carregado.")
    else:
        print("[ALERTA] 'modelo_sintetico_comparacao.h5' não encontrado.")

    if os.path.exists("modelo_hibrido_segmentador.h5"):
        model_hibrido = tf.keras.models.load_model("modelo_hibrido_segmentador.h5", compile=False)
        print("[OK] Modelo M3 (Segmentador Híbrido Reais/U-Net) carregado.")
    else:
        print("[ALERTA] 'modelo_hibrido_segmentador.h5' não encontrado.")

except Exception as e:
    print(f"[ERRO] Falha no TensorFlow: {e}")

# ============================================================
# 3. GEOMETRIA E VARREDURA AMPLIADA (0.30 - 0.75)
# ============================================================
def calcular_geometria_roi_avancada(binaria_img):
    mascara_roi = np.zeros_like(binaria_img)
    pts_roi = np.array([
        [int(LARGURA_PROC * 0.15), int(ALTURA_PROC * 0.30)],
        [int(LARGURA_PROC * 0.85), int(ALTURA_PROC * 0.30)],
        [LARGURA_PROC, ALTURA_PROC],
        [0, ALTURA_PROC]
    ], dtype=np.int32)
    cv2.fillPoly(mascara_roi, [pts_roi], 255)

    bin_roi = cv2.bitwise_and(binaria_img, binaria_img, mask=mascara_roi)

    # Filtragem Conectada
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_roi, connectivity=8)
    bin_filtrada = np.zeros_like(binaria_img)
    for i in range(1, num_labels):
        w, h, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if area >= AREA_MINIMA and w >= LARGURA_MINIMA and h >= ALTURA_MINIMA:
            bin_filtrada[labels == i] = 255

    fechada = cv2.morphologyEx(bin_filtrada, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # Varredura Ampliada: do topo até o meio (0.30 a 0.75)
    sub_destino = fechada[int(ALTURA_PROC * 0.30):int(ALTURA_PROC * 0.75), :]
    num_sub, _, stats_sub, centroides_sub = cv2.connectedComponentsWithStats(sub_destino, connectivity=8)

    ramos_destino = []
    for i in range(1, num_sub):
        if stats_sub[i, cv2.CC_STAT_AREA] >= AREA_MINIMA:
            cx = int(centroides_sub[i][0])
            cy = int(centroides_sub[i][1]) + int(ALTURA_PROC * 0.30)
            ramos_destino.append((cx, cy))

    # Região Base (Robô)
    sub_base = fechada[int(ALTURA_PROC * 0.75):ALTURA_PROC, :]
    M_base = cv2.moments(sub_base)

    if M_base["m00"] <= AREA_MINIMA or len(ramos_destino) == 0:
        return "Sem_Caminho", 0.0, fechada, None, []

    cx_base = int(M_base["m10"] / M_base["m00"])
    cy_base = int(M_base["m01"] / M_base["m00"]) + int(ALTURA_PROC * 0.75)
    pt_base = (cx_base, cy_base)

    angulos = []
    for pt in ramos_destino:
        dx = pt[0] - pt_base[0]
        dy = pt_base[1] - pt[1]
        ang = np.degrees(np.arctan2(dx, dy))
        angulos.append(ang)

    # Classificação
    if len(ramos_destino) >= 3:
        classe = "Cruzamento_4Pistas"
    elif len(ramos_destino) == 2:
        classe = "Cruzamento_3Pistas" if any(abs(a) <= 15 for a in angulos) else "Bifurcacao_Esq_Dir"
    elif len(ramos_destino) == 1:
        ang = angulos[0]
        if -12 <= ang <= 12: classe = "Reta"
        elif -40 <= ang < -12: classe = "Curva_Aberta_Esq"
        elif 12 < ang <= 40: classe = "Curva_Aberta_Dir"
        elif ang < -40: classe = "Curva_Fechada_Esq"
        else: classe = "Curva_Fechada_Dir"
    else:
        classe = "Sem_Caminho"

    return classe, angulos[0] if angulos else 0.0, fechada, pt_base, ramos_destino

# ============================================================
# 4. BINARIZAÇÃO E INFERÊNCIA DOS MÉTODOS
# ============================================================
def binarizar_blackhat(frame_bgr):
    cinza = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    desfocado = cv2.GaussianBlur(cinza, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    blackhat = cv2.morphologyEx(desfocado, cv2.MORPH_BLACKHAT, kernel)
    _, binaria = cv2.threshold(blackhat, 25, 255, cv2.THRESH_BINARY)
    return binaria

def executar_metodo_1(binaria_img):
    if model_cnn is None: return "N/A"
    binaria_3c = cv2.cvtColor(binaria_img, cv2.COLOR_GRAY2BGR)
    inp = np.expand_dims(cv2.cvtColor(binaria_3c, cv2.COLOR_BGR2RGB).astype(np.float32), axis=0)
    pred = model_cnn.predict(inp, verbose=0)
    return CLASSES_CNN[np.argmax(pred[0])]

def executar_metodo_2(frame_bgr):
    binaria_fita = binarizar_blackhat(frame_bgr)
    c, a, mask, p, r = calcular_geometria_roi_avancada(binaria_fita)
    historico_m2.append(a)
    return c, float(np.mean(historico_m2)) if len(historico_m2) > 0 else 0.0, mask, p, r

def executar_metodo_3(frame_rgb):
    if model_hibrido is None:
        return "N/A", 0.0, np.zeros((128, 128), dtype=np.uint8), None, []

    inp = np.expand_dims(frame_rgb.astype(np.float32) / 255.0, axis=0)
    pred_mask = model_hibrido.predict(inp, verbose=0)[0, :, :, 0]
    binaria_ia = ((pred_mask > 0.35).astype(np.uint8)) * 255
    
    c, a, mask, p, r = calcular_geometria_roi_avancada(binaria_ia)
    historico_m3.append(a)
    return c, float(np.mean(historico_m3)) if len(historico_m3) > 0 else 0.0, mask, p, r

# ============================================================
# 5. EXECUÇÃO DO PROCESSAMENTO DO VÍDEO
# ============================================================
cap = cv2.VideoCapture(VIDEO_ENTRADA)
if not cap.isOpened():
    raise FileNotFoundError(f"Vídeo {VIDEO_ENTRADA} não encontrado.")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(VIDEO_SAIDA, fourcc, fps, (768, 256), True)

if not out.isOpened():
    VIDEO_SAIDA = "resultado_comparacao_modelos_reais.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(VIDEO_SAIDA, fourcc, fps, (768, 256), True)

tempos_m1, tempos_m2, tempos_m3 = [], [], []
angulos_m2, angulos_m3 = [], []

# Contadores para taxa de sucesso (Confiança na detecção)
validos_m1, validos_m2, validos_m3 = 0, 0, 0
total_frames = 0

print("[INFO] Iniciando comparação analítica dos novos modelos...")

while True:
    ret, frame = cap.read()
    if not ret: break

    total_frames += 1
    if total_frames % PULAR_FRAMES != 0: continue

    frame_red = cv2.resize(frame, (LARGURA_PROC, ALTURA_PROC))
    frame_rgb = cv2.cvtColor(frame_red, cv2.COLOR_BGR2RGB)

    # Executa M2 (Lógica Black Hat)
    t0 = time.time()
    c_m2, a_m2, bin_m2, pt_base_m2, ramos_m2 = executar_metodo_2(frame_red)
    t_m2 = (time.time() - t0) * 1000
    tempos_m2.append(t_m2)
    angulos_m2.append(a_m2)
    if c_m2 != "Sem_Caminho": validos_m2 += 1

    # Executa M1 (CNN Classificadora sobre a fita binarizada)
    t0 = time.time()
    c_m1 = executar_metodo_1(bin_m2)
    t_m1 = (time.time() - t0) * 1000
    tempos_m1.append(t_m1)
    if c_m1 != "Sem_Caminho": validos_m1 += 1

    # Executa M3 (Híbrido U-Net Reais + Geometria)
    t0 = time.time()
    c_m3, a_m3, bin_m3, pt_base_m3, ramos_m3 = executar_metodo_3(frame_rgb)
    t_m3 = (time.time() - t0) * 1000
    tempos_m3.append(t_m3)
    angulos_m3.append(a_m3)
    if c_m3 != "Sem_Caminho": validos_m3 += 1

    # Montagem dos Painéis
    p1 = frame_red.copy()
    p2 = cv2.cvtColor(bin_m2, cv2.COLOR_GRAY2BGR)
    p3 = cv2.cvtColor(bin_m3, cv2.COLOR_GRAY2BGR)

    if pt_base_m2 is not None:
        cv2.circle(p2, pt_base_m2, 3, (0, 0, 255), -1)
        for pt in ramos_m2:
            cv2.circle(p2, pt, 3, (0, 255, 0), -1)
            cv2.line(p2, pt_base_m2, pt, (0, 255, 255), 2)

    if pt_base_m3 is not None:
        cv2.circle(p3, pt_base_m3, 3, (0, 0, 255), -1)
        for pt in ramos_m3:
            cv2.circle(p3, pt, 3, (0, 255, 0), -1)
            cv2.line(p3, pt_base_m3, pt, (0, 255, 255), 2)

    cv2.putText(p1, f"M1: {c_m1}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.putText(p2, f"M2: {c_m2} ({a_m2:.1f}d)", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    cv2.putText(p3, f"M3: {c_m3} ({a_m3:.1f}d)", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 100, 255), 1)

    out.write(cv2.resize(np.hstack((p1, p2, p3)), (768, 256), interpolation=cv2.INTER_NEAREST))

cap.release()
out.release()

# ============================================================
# 6. GERAÇÃO DO DASHBOARD DE 4 GRÁFICOS ANALÍTICOS
# ============================================================
print("[INFO] Gerando Dashboard de Comparação Avançada...")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# 1. Latência por Frame (ms)
axes[0, 0].plot(tempos_m1, label=f"M1: CNN ({np.mean(tempos_m1):.1f}ms)", color="green", alpha=0.7)
axes[0, 0].plot(tempos_m2, label=f"M2: Lógica BlackHat ({np.mean(tempos_m2):.1f}ms)", color="orange", alpha=0.8)
axes[0, 0].plot(tempos_m3, label=f"M3: Híbrido Reais ({np.mean(tempos_m3):.1f}ms)", color="purple", alpha=0.8)
axes[0, 0].set_title("1. Latência por Frame (ms)")
axes[0, 0].set_ylabel("Tempo (ms)")
axes[0, 0].legend()
axes[0, 0].grid(True, linestyle="--", alpha=0.5)

# 2. Estabilidade Angular (Graus)
axes[0, 1].plot(angulos_m2, label="M2: Lógica Suavizada", color="orange", linestyle="--")
axes[0, 1].plot(angulos_m3, label="M3: Híbrido U-Net", color="purple")
axes[0, 1].set_title("2. Estabilidade da Trajetória (Ângulo em Graus)")
axes[0, 1].set_ylabel("Ângulo (º)")
axes[0, 1].legend()
axes[0, 1].grid(True, linestyle="--", alpha=0.5)

# 3. Boxplot: Distribuição de Tempo e Estabilidade de CPU
axes[1, 0].boxplot([tempos_m1, tempos_m2, tempos_m3], labels=["M1 (CNN)", "M2 (Lógica)", "M3 (Híbrido)"])
axes[1, 0].set_title("3. Dispersão de Tempo de Processamento (Boxplot)")
axes[1, 0].set_ylabel("Tempo (ms)")
axes[1, 0].grid(True, linestyle="--", alpha=0.5)

# 4. Taxa de Sucesso / Confiança na Detecção (% de Pista Válida)
taxas = [
    (validos_m1 / max(total_frames, 1)) * 100,
    (validos_m2 / max(total_frames, 1)) * 100,
    (validos_m3 / max(total_frames, 1)) * 100
]
barras = axes[1, 1].bar(["M1 (CNN)", "M2 (Lógica)", "M3 (Híbrido)"], taxas, color=["green", "orange", "purple"], alpha=0.8)
axes[1, 1].set_title("4. Taxa de Retenção de Pista (% de Frames Válidos)")
axes[1, 1].set_ylabel("Taxa de Sucesso (%)")
axes[1, 1].set_ylim(0, 110)
axes[1, 1].grid(True, linestyle="--", alpha=0.5)

# Adiciona os rótulos de porcentagem em cima das barras
for bar in barras:
    yval = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("grafico_comparativo_modelos_reais.png", dpi=300)
plt.show()

print("\n[PROCESSO FINALIZADO]")
print(f" - Vídeo comparativo gerado: '{VIDEO_SAIDA}'")
print(" - Dashboard com 4 gráficos gerado: 'grafico_comparativo_modelos_reais.png'")