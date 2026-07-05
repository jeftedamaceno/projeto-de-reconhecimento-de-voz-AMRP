import os
import sys
import json
import math
import time
import queue
import threading
import numpy as np
import tensorflow as tf
import sounddevice as sd
import pygame

@tf.keras.utils.register_keras_serializable(package="Custom")
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = inputs * a
        return tf.keras.backend.sum(output, axis=1)

    def get_config(self):
        return super(AttentionLayer, self).get_config()

# Configurações de áudio fixas do seu modelo
SAMPLE_RATE = 16000
DURATION = 1.5  
TOTAL_SAMPLES = int(SAMPLE_RATE * DURATION) # 24000 amostras

# Parâmetros para o modo contínuo (Sliding Window)
STEP_DURATION = 0.20  # A cada 200ms o modelo faz uma nova varredura
STEP_SAMPLES = int(SAMPLE_RATE * STEP_DURATION)
SILENCE_THRESHOLD = 0.005  # Ignora se o som for muito baixo (ajuste se necessário)

MODEL_PATH = "modelo_hibrido_1s5_atencao.h5" 
if not os.path.exists(MODEL_PATH):
    print(f"ERRO: O arquivo '{MODEL_PATH}' nao foi encontrado no diretorio.")
    sys.exit()

model_audio = tf.keras.models.load_model(
    MODEL_PATH, 
    custom_objects={"AttentionLayer": AttentionLayer}
)

with open("labels_1s5_atencao.json", "r") as f:
    label_map = json.load(f)
inv_label_map = {v: k for k, v in label_map.items()}

voice_command_queue = queue.Queue()
hud_update_queue = queue.Queue()

# Buffer que conterá exatamente os últimos 1,5 segundos de áudio o tempo todo
audio_buffer = np.zeros(TOTAL_SAMPLES, dtype=np.float32)

def audio_callback(indata, frames, time_info, status):
    """Callback invocado pelo sounddevice para encher o buffer continuamente."""
    global audio_buffer
    # Desloca o buffer antigo para a esquerda e insere os novos dados na direita
    audio_buffer = np.roll(audio_buffer, -frames)
    audio_buffer[-frames:] = indata[:, 0]

def audio_continuous_inference_thread():
    """Thread que analisa a janela deslizante de 1,5 segundos periodicamente."""
    global audio_buffer  # <-- COLOQUE AQUI, NA PRIMEIRA LINHA DA FUNÇÃO!
    
    print("\n[SISTEMA] Escuta contínua ativada. Pode falar os comandos a qualquer momento...")
    
    # Inicia o fluxo de entrada contínuo
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=STEP_SAMPLES)
    with stream:
        while True:
            # Captura uma cópia instantânea e segura do buffer de 1.5 segundos
            audio_raw = audio_buffer.copy()
            
            # 1. Filtro de Silêncio (Evita sobrecarregar o modelo com ruído de fundo vazio)
            rms = np.sqrt(np.mean(audio_raw**2))
            if rms < SILENCE_THRESHOLD:
                time.sleep(STEP_DURATION)
                continue
            
            # 2. Mantém o pré-processamento idêntico ao seu pipeline original
            audio_int16 = (audio_raw * 32767.0).astype(np.float32)
            
            if np.std(audio_int16) > 0:
                audio_preprocessed = (audio_int16 - np.mean(audio_int16)) / np.std(audio_int16)
            else:
                audio_preprocessed = audio_int16
                
            input_neural = np.expand_dims(audio_preprocessed, axis=(0, -1))
            
            # 3. Predição em tempo real
            preds = model_audio.predict(input_neural, verbose=0)[0]
            hud_update_queue.put(preds) 
            
            idx_classe = np.argmax(preds)
            confianca = preds[idx_classe]
            
            # Exige uma confiança sólida (ex: 75%) para evitar ativações fantasmas por ruído

            if confianca > 0.75:
                palavra_detectada = inv_label_map[idx_classe].lower()
                voice_command_queue.put(palavra_detectada)
                
                # --- APENAS LIMPA O BUFFER (a declaração global já está no topo da função) ---
                audio_buffer = np.zeros(TOTAL_SAMPLES, dtype=np.float32)
                
                # Faz a thread dormir por 1.0 segundo inteiro para dar tempo de terminar de falar
                time.sleep(1.0)
                continue
                
            time.sleep(STEP_DURATION)

# Inicia a captura e inferência de forma totalmente automatizada
threading.Thread(target=audio_continuous_inference_thread, daemon=True).start()
WORLD_X_MIN, WORLD_X_MAX = -1.0, 5.0
WORLD_Y_MIN, WORLD_Y_MAX = -2.0, 3.0
FINISH_X = 4.9
FINISH_Y_START = -2.0
FINISH_Y_END = -1.0
ROBOT_RADIUS = 0.06
WALL_HEIGHT = 1.6

WALLS = [
    (WORLD_X_MIN, WORLD_Y_MIN, WORLD_X_MIN, WORLD_Y_MAX),
    (WORLD_X_MIN, WORLD_Y_MAX, WORLD_X_MAX, WORLD_Y_MAX),
    (WORLD_X_MAX, WORLD_Y_MAX, WORLD_X_MAX, WORLD_Y_MIN),
    (WORLD_X_MIN, WORLD_Y_MIN, WORLD_X_MAX, WORLD_Y_MIN),
    (-1, 1, 1, 1), (2, 3, 2, 2), (2, 1, 2, -1), (0, -1, 3, -1),
    (3, 1, 5, 1), (4, -1, 5, -1)
]

def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

class Camera3D:
    def __init__(self, width, height, focal_length=250, cam_height=0.4, horizon_y=None):
        self.width = width
        self.height = height
        self.focal_length = focal_length
        self.cam_height = cam_height
        self.horizon_y = horizon_y if horizon_y is not None else int(height * 0.4)
        self.center_x = width / 2
        self.cam_x, self.cam_y = 0.0, 0.0

    def set_position(self, cam_x, cam_y):
        self.cam_x, self.cam_y = cam_x, cam_y

    def project(self, wx, wy, car_x, car_y, car_theta, obj_height=0.0):
        dx, dy, dz = wx - self.cam_x, wy - self.cam_y, obj_height - self.cam_height
        cos_t, sin_t = math.cos(car_theta), math.sin(car_theta)
        frente = dx * cos_t + dy * sin_t
        lateral = -dx * sin_t + dy * cos_t
        if frente <= 0.01: return None, None, None
        inv_depth = 1.0 / frente
        sx = self.center_x - lateral * inv_depth * self.focal_length
        sy = self.horizon_y - dz * inv_depth * self.focal_length
        return sx, sy, frente

def project_cuboid(cam, car_x, car_y, car_theta, pos_x, pos_y, angle, width, length, height):
    hw, hl = width / 2.0, length / 2.0
    local_verts = [(-hl, -hw, 0), (hl, -hw, 0), (hl, hw, 0), (-hl, hw, 0),
                   (-hl, -hw, height), (hl, -hw, height), (hl, hw, height), (-hl, hw, height)]
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    world_verts = [(pos_x + lx*cos_a - ly*sin_a, pos_y + lx*sin_a + ly*cos_a, lz) for lx, ly, lz in local_verts]
    proj, depths = [], []
    for (wx, wy, wz) in world_verts:
        sx, sy, d = cam.project(wx, wy, car_x, car_y, car_theta, obj_height=wz)
        proj.append((sx, sy))
        depths.append(d if d is not None else 1e9)
    faces_def = [(0,1,2,3, (180,0,0)), (4,5,6,7, (220,50,50)), (0,3,7,4, (160,0,0)),
                 (1,2,6,5, (160,0,0)), (0,1,5,4, (200,20,20)), (2,3,7,6, (200,20,20))]
    face_list = []
    for i1, i2, i3, i4, cor in faces_def:
        pts = [proj[i] for i in (i1, i2, i3, i4)]
        if any(p[0] is None for p in pts): continue
        face_list.append((pts, cor, (depths[i1] + depths[i2] + depths[i3] + depths[i4]) / 4.0))
    face_list.sort(key=lambda f: f[2], reverse=True)
    return face_list

def segment_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if abs(denom) < 1e-12: return None
    t = ((x1 - x3)*(y3 - y4) - (y1 - y3)*(x3 - x4)) / denom
    u = -((x1 - x2)*(y1 - y3) - (y1 - y2)*(x1 - x3)) / denom
    if 0 <= t <= 1 and 0 <= u <= 1: return t
    return None

def point_segment_distance(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    fx, fy = px - x1, py - y1
    t = (fx*dx + fy*dy) / (dx*dx + dy*dy + 1e-12)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (x1 + t*dx), py - (y1 + t*dy))

def adjust_camera(car_x, car_y, car_theta, desired_dist, walls, min_dist=0.3, wall_margin=0.15):
    cam_x = car_x - desired_dist * math.cos(car_theta)
    cam_y = car_y - desired_dist * math.sin(car_theta)
    best_t = 1.0
    for (wx1, wy1, wx2, wy2) in WALLS:
        t = segment_intersection(car_x, car_y, cam_x, cam_y, wx1, wy1, wx2, wy2)
        if t is not None and t < best_t: best_t = t
    dist = max(min_dist, best_t * desired_dist)
    cam_x = car_x - dist * math.cos(car_theta)
    cam_y = car_y - dist * math.sin(car_theta)
    return cam_x, cam_y

def circle_segment_collision(cx, cy, radius, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    fx, fy = cx - x1, cy - y1
    t = (fx*dx + fy*dy) / (dx*dx + dy*dy + 1e-12)
    t = max(0.0, min(1.0, t))
    return math.hypot(cx - (x1 + t*dx), cy - (y1 + t*dy)) <= radius

def main():
    pygame.init()
    WIDTH, HEIGHT = 900, 680
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Robo - Escuta Contínua Ativa (Sem Botões)")
    clock = pygame.time.Clock()
    
    font_bold = pygame.font.SysFont("Arial", 18, bold=True)
    font_small = pygame.font.SysFont("Arial", 13)

    dt = 0.02
    b = 0.094 / 2.0
    CAR_WIDTH, CAR_LENGTH, CAR_HEIGHT = 2.0 * b, 2.5 * b, 0.1
    x, y, theta = -0.5, 2.5, 0.0
    trail = [(x, y)]
    prev_x = x

    base_speed = 0.26
    omega_rotate = 1.5

    command = 'straight' 
    historico_palavras_ditas = []
    probabilidades_classes = np.zeros(len(label_map))

    cam3d = Camera3D(WIDTH, HEIGHT - 140, focal_length=250, cam_height=0.4, horizon_y=int(HEIGHT*0.32))

    MAP_WIDTH, MAP_HEIGHT = 180, 130
    MAP_X, MAP_Y = WIDTH - MAP_WIDTH - 15, 15

    running = True
    finish_crossed = False
    collision = False

    while running:
        clock.tick(50)

        while not hud_update_queue.empty():
            probabilidades_classes = hud_update_queue.get()

# --- FILTRO ANTIDUPLICAÇÃO DE COMANDOS DE VOZ ---
        if not voice_command_queue.empty():
            palavra = voice_command_queue.get()
            
            # Só aceitamos um comando de rotação se o carrinho NÃO estiver girando no momento
            pode_girar = command not in ('rot_left', 'rot_right', 'rot_180')

            comando_valido = True
            if palavra in ['siga', 'frente', 'ir']: 
                proximo_comando = 'straight'
            elif palavra in ['esquerda', 'esq'] and pode_girar: 
                proximo_comando = 'rot_left'
            elif palavra in ['direita', 'dir'] and pode_girar: 
                proximo_comando = 'rot_right'
            elif palavra in ['pare', 'parar', 'stop']: 
                proximo_comando = 'stop'
            elif palavra in ['voltar', 're', '180'] and pode_girar: 
                proximo_comando = 'rot_180'
            else:
                comando_valido = False # Ignora se tentou girar enquanto já estava girando

            if comando_valido:
                command = proximo_comando
                historico_palavras_ditas.append(palavra.upper())
                if len(historico_palavras_ditas) > 4: 
                    historico_palavras_ditas.pop(0)

                # Inicializa as variáveis de rotação apenas se o novo comando exigir isso
                if command in ('rot_left', 'rot_right', 'rot_180'):
                    rot_start_theta = theta
                    rot_accumulated = 0.0
                    if command == 'rot_left': 
                        rot_target_delta = math.pi / 2.0
                        rot_omega_sign = 1.0
                    elif command == 'rot_right': 
                        rot_target_delta = -math.pi / 2.0
                        rot_omega_sign = -1.0
                    elif command == 'rot_180': 
                        rot_target_delta = math.pi
                        rot_omega_sign = 1.0
                        
        if command == 'stop': 
            v = 0.0
            delta_theta = 0.0
            estado_atual_nome = "PARADO"
        elif command == 'straight': 
            v = base_speed
            delta_theta = 0.0
            estado_atual_nome = "SIGA (EM FRENTE)"
        elif command in ('rot_left', 'rot_right', 'rot_180'):
            v = 0.0
            # Calcula o quanto viraria neste frame
            proximo_passo = rot_omega_sign * omega_rotate * dt
            
            # Verifica se este frame vai estourar ou atingir o limite estrito do ângulo
            if abs(rot_accumulated + proximo_passo) >= abs(rot_target_delta):
                # Força o ângulo final exato para evitar "oversteer" (virar demais)
                theta = normalize_angle(rot_start_theta + rot_target_delta)
                command = 'straight'  # Retorna a ir em frente imediatamente
                delta_theta = 0.0
                estado_atual_nome = "SIGA (EM FRENTE)"
            else:
                delta_theta = proximo_passo
                rot_accumulated += delta_theta
                theta += delta_theta
                estado_atual_nome = f"GIRANDO ({command.upper()})"

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        # Removida a dependência do botão 'G' para a física do carrinho
        if command == 'stop': v = 0.0; delta_theta = 0.0
        elif command == 'straight': v = base_speed; delta_theta = 0.0
        elif command in ('rot_left', 'rot_right', 'rot_180'):
            v = 0.0
            delta_theta = rot_omega_sign * omega_rotate * dt
            rot_accumulated += delta_theta
            if abs(rot_accumulated) >= abs(rot_target_delta):
                theta = normalize_angle(rot_start_theta + rot_target_delta)
                command = 'straight' 
                delta_theta = 0.0
            else: theta += delta_theta
        estado_atual_nome = command.upper()

        delta_x = dt * v * math.cos(theta) if command == 'straight' else 0.0
        delta_y = dt * v * math.sin(theta) if command == 'straight' else 0.0
        new_x, new_y = x + delta_x, y + delta_y

        hit_wall = False
        for (wx1, wy1, wx2, wy2) in WALLS:
            if circle_segment_collision(new_x, new_y, ROBOT_RADIUS, wx1, wy1, wx2, wy2):
                hit_wall = True; break
        if hit_wall: collision = True; running = False
        else: x, y = new_x, new_y

        if not finish_crossed and prev_x < FINISH_X and x >= FINISH_X:
            if FINISH_Y_START <= y <= FINISH_Y_END: finish_crossed = True; running = False
        prev_x = x
        trail.append((x, y))

        cam_x, cam_y = adjust_camera(x, y, theta, 0.7, WALLS)
        cam3d.set_position(cam_x, cam_y)

        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, (100, 149, 237), (0, 0, WIDTH, cam3d.horizon_y)) 
        pygame.draw.rect(screen, (50, 50, 50), (0, cam3d.horizon_y, WIDTH, (HEIGHT - 140) - cam3d.horizon_y)) 

        objects = []
        for (wx1, wy1, wx2, wy2) in WALLS:
            length = math.hypot(wx2 - wx1, wy2 - wy1)
            if length < 0.01: continue
            n_samples = max(2, int(length * 10))
            prev_sx_base = prev_sy_base = prev_sx_top = prev_sy_top = None
            prev_d_base = prev_d_top = 0.0
            for i in range(n_samples + 1):
                t_w = i / n_samples
                px, py = wx1 + t_w * (wx2 - wx1), wy1 + t_w * (wy2 - wy1)
                sx_base, sy_base, d_base = cam3d.project(px, py, x, y, theta, obj_height=0.0)
                sx_top, sy_top, d_top = cam3d.project(px, py, x, y, theta, obj_height=WALL_HEIGHT)
                if sx_base is not None and sx_top is not None:
                    if prev_sx_base is not None:
                        pts = [(prev_sx_base, prev_sy_base), (sx_base, sy_base), (sx_top, sy_top), (prev_sx_top, prev_sy_top)]
                        objects.append(((prev_d_base + d_base + d_top + prev_d_top) / 4.0, 'wall', pts))
                    prev_sx_base, prev_sy_base = sx_base, sy_base
                    prev_sx_top, prev_sy_top = sx_top, sy_top
                    prev_d_base, prev_d_top = d_base, d_top
                else: prev_sx_base = None

        objects.sort(key=lambda obj: obj[0], reverse=True)
        for depth, typ, data in objects:
            if typ == 'wall':
                pygame.draw.polygon(screen, (160, 160, 160), data)
                pygame.draw.polygon(screen, (30, 30, 30), data, 1)

        car_faces = project_cuboid(cam3d, x, y, theta, x, y, theta, CAR_WIDTH, CAR_LENGTH, CAR_HEIGHT)
        for pts, cor, _ in car_faces:
            if len(pts) >= 3:
                pygame.draw.polygon(screen, cor, pts)
                pygame.draw.polygon(screen, (0, 0, 0), pts, 1)

        minimap_surf = pygame.Surface((MAP_WIDTH, MAP_HEIGHT))
        minimap_surf.fill((15, 15, 15))
        scale_x = MAP_WIDTH / 6.0
        scale_y = MAP_HEIGHT / 5.0
        def world_to_map(wx, wy):
            return int((wx - (-1.0)) * scale_x), int((3.0 - wy) * scale_y)

        for (wx1, wy1, wx2, wy2) in WALLS:
            pygame.draw.line(minimap_surf, (100, 100, 100), world_to_map(wx1, wy1), world_to_map(wx2, wy2), 1)
        if len(trail) > 1:
            pygame.draw.lines(minimap_surf, (0, 255, 0), False, [world_to_map(px, py) for px, py in trail], 1)
        pygame.draw.circle(minimap_surf, (255, 0, 0), world_to_map(x, y), 4)
        screen.blit(minimap_surf, (MAP_X, MAP_Y))

        hud_top_y = HEIGHT - 140
        pygame.draw.rect(screen, (20, 20, 30), (0, hud_top_y, WIDTH, 140))
        
        # O HUD agora reflete a escuta contínua ativa
        pygame.draw.line(screen, (0, 255, 150), (0, hud_top_y), (WIDTH, hud_top_y), 2)
        lbl_grav = font_bold.render("ESCUTA CONTÍNUA ATIVA (FALE SEUS COMANDOS)...", True, (0, 255, 150))
        screen.blit(lbl_grav, (20, hud_top_y + 70))

        lbl_historico = font_bold.render("HISTORICO DO LEITOR DE VOZ (1D MODEL - 1.5S):", True, (240, 240, 240))
        screen.blit(lbl_historico, (20, hud_top_y + 15))
        
        texto_linha_historico = " -> ".join(historico_palavras_ditas) if historico_palavras_ditas else "Aguardando comandos de voz..."
        lbl_linha = font_small.render(texto_linha_historico, True, (0, 255, 150))
        screen.blit(lbl_linha, (20, hud_top_y + 42))

        lbl_estado = font_small.render(f"Fisica do Carrinho: {estado_atual_nome}", True, (180, 180, 180))
        screen.blit(lbl_estado, (20, hud_top_y + 95))

        bar_start_x = 480
        bar_start_y = hud_top_y + 12
        for c_name, c_idx in sorted(label_map.items(), key=lambda item: item[1]):
            prob = probabilidades_classes[c_idx] if c_idx < len(probabilidades_classes) else 0.0
            text_c = font_small.render(f"{c_name.upper():<9}", True, (230, 230, 230))
            screen.blit(text_c, (bar_start_x, bar_start_y))
            pygame.draw.rect(screen, (50, 50, 60), (bar_start_x + 80, bar_start_y + 2, 200, 12))
            bar_color = (0, 255, 150) if prob > 0.75 else (255, 165, 0) if prob > 0.30 else (100, 100, 110)
            pygame.draw.rect(screen, bar_color, (bar_start_x + 80, bar_start_y + 2, int(prob * 200), 12))
            text_p = font_small.render(f"{prob * 100:.1f}%", True, (200, 200, 200))
            screen.blit(text_p, (bar_start_x + 290, bar_start_y))
            bar_start_y += 22

        pygame.display.flip()

    pygame.quit()
    if finish_crossed: print("\nVitoria! O carrinho alcancou o final do percurso.")
    if collision: print("\nBatida violenta na parede detectada!")

if __name__ == "__main__":
    main()


# import os
# import sys
# import json
# import math
# import time
# import queue
# import threading
# import numpy as np
# import tensorflow as tf
# import sounddevice as sd
# import pygame
# import scipy.io.wavfile as wav
# import cv2

# @tf.keras.utils.register_keras_serializable(package="Custom")
# class AttentionLayer(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(AttentionLayer, self).__init__(**kwargs)

#     def build(self, input_shape):
#         self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
#         self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
#         super(AttentionLayer, self).build(input_shape)

#     def call(self, inputs):
#         e = tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W) + self.b)
#         a = tf.keras.backend.softmax(e, axis=1)
#         output = inputs * a
#         return tf.keras.backend.sum(output, axis=1)

#     def get_config(self):
#         return super(AttentionLayer, self).get_config()

# SAMPLE_RATE = 16000
# DURATION = 1.5  
# TOTAL_SAMPLES = int(SAMPLE_RATE * DURATION)
# STEP_DURATION = 0.20  
# STEP_SAMPLES = int(SAMPLE_RATE * STEP_DURATION)
# SILENCE_THRESHOLD = 0.005  

# MODEL_PATH = "modelo_hibrido_1s5_atencao.h5" 
# if not os.path.exists(MODEL_PATH):
#     sys.exit()

# model_audio = tf.keras.models.load_model(
#     MODEL_PATH, 
#     custom_objects={"AttentionLayer": AttentionLayer}
# )

# with open("labels_1s5_atencao.json", "r") as f:
#     label_map = json.load(f)
# inv_label_map = {v: k for k, v in label_map.items()}

# voice_command_queue = queue.Queue()
# hud_update_queue = queue.Queue()
# audio_buffer = np.zeros(TOTAL_SAMPLES, dtype=np.float32)

# audio_sessao_completa = []

# def audio_callback(indata, frames, time_info, status):
#     global audio_buffer, audio_sessao_completa
#     audio_buffer = np.roll(audio_buffer, -frames)
#     audio_buffer[-frames:] = indata[:, 0]
#     audio_sessao_completa.extend(indata[:, 0].tolist())

# def audio_continuous_inference_thread():
#     global audio_buffer  
#     stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=STEP_SAMPLES)
#     with stream:
#         while True:
#             audio_raw = audio_buffer.copy()
#             rms = np.sqrt(np.mean(audio_raw**2))
#             if rms < SILENCE_THRESHOLD:
#                 time.sleep(STEP_DURATION)
#                 continue
            
#             audio_int16 = (audio_raw * 32767.0).astype(np.float32)
#             if np.std(audio_int16) > 0:
#                 audio_preprocessed = (audio_int16 - np.mean(audio_int16)) / np.std(audio_int16)
#             else:
#                 audio_preprocessed = audio_int16
                
#             input_neural = np.expand_dims(audio_preprocessed, axis=(0, -1))
#             preds = model_audio.predict(input_neural, verbose=0)[0]
#             hud_update_queue.put(preds) 
            
#             idx_classe = np.argmax(preds)
#             confianca = preds[idx_classe]

#             if confianca > 0.75:
#                 palavra_detectada = inv_label_map[idx_classe].lower()
#                 voice_command_queue.put(palavra_detectada)
#                 audio_buffer = np.zeros(TOTAL_SAMPLES, dtype=np.float32)
#                 time.sleep(1.0)
#                 continue
                
#             time.sleep(STEP_DURATION)

# threading.Thread(target=audio_continuous_inference_thread, daemon=True).start()

# WORLD_X_MIN, WORLD_X_MAX = -1.0, 5.0
# WORLD_Y_MIN, WORLD_Y_MAX = -2.0, 3.0
# FINISH_X = 4.9
# FINISH_Y_START = -2.0
# FINISH_Y_END = -1.0
# ROBOT_RADIUS = 0.06
# WALL_HEIGHT = 1.6

# WALLS = [
#     (WORLD_X_MIN, WORLD_Y_MIN, WORLD_X_MIN, WORLD_Y_MAX),
#     (WORLD_X_MIN, WORLD_Y_MAX, WORLD_X_MAX, WORLD_Y_MAX),
#     (WORLD_X_MAX, WORLD_Y_MAX, WORLD_X_MAX, WORLD_Y_MIN),
#     (WORLD_X_MIN, WORLD_Y_MIN, WORLD_X_MAX, WORLD_Y_MIN),
#     (-1, 1, 1, 1), (2, 3, 2, 2), (2, 1, 2, -1), (0, -1, 3, -1),
#     (3, 1, 5, 1), (4, -1, 5, -1)
# ]

# def normalize_angle(angle):
#     return (angle + math.pi) % (2 * math.pi) - math.pi

# class Camera3D:
#     def __init__(self, width, height, focal_length=250, cam_height=0.4, horizon_y=None):
#         self.width = width
#         self.height = height
#         self.focal_length = focal_length
#         self.cam_height = cam_height
#         self.horizon_y = horizon_y if horizon_y is not None else int(height * 0.4)
#         self.center_x = width / 2
#         self.cam_x, self.cam_y = 0.0, 0.0

#     def set_position(self, cam_x, cam_y):
#         self.cam_x, self.cam_y = cam_x, cam_y

#     def project(self, wx, wy, car_x, car_y, car_theta, obj_height=0.0):
#         dx, dy, dz = wx - self.cam_x, wy - self.cam_y, obj_height - self.cam_height
#         cos_t, sin_t = math.cos(car_theta), math.sin(car_theta)
#         frente = dx * cos_t + dy * sin_t
#         lateral = -dx * sin_t + dy * cos_t
#         if frente <= 0.01: return None, None, None
#         inv_depth = 1.0 / frente
#         sx = self.center_x - lateral * inv_depth * self.focal_length
#         sy = self.horizon_y - dz * inv_depth * self.focal_length
#         return sx, sy, frente

# def project_cuboid(cam, car_x, car_y, car_theta, pos_x, pos_y, angle, width, length, height):
#     hw, hl = width / 2.0, length / 2.0
#     local_verts = [(-hl, -hw, 0), (hl, -hw, 0), (hl, hw, 0), (-hl, hw, 0),
#                    (-hl, -hw, height), (hl, -hw, height), (hl, hw, height), (-hl, hw, height)]
#     cos_a, sin_a = math.cos(angle), math.sin(angle)
#     world_verts = [(pos_x + lx*cos_a - ly*sin_a, pos_y + lx*sin_a + ly*cos_a, lz) for lx, ly, lz in local_verts]
#     proj, depths = [], []
#     for (wx, wy, wz) in world_verts:
#         sx, sy, d = cam.project(wx, wy, car_x, car_y, car_theta, obj_height=wz)
#         proj.append((sx, sy))
#         depths.append(d if d is not None else 1e9)
#     faces_def = [(0,1,2,3, (180,0,0)), (4,5,6,7, (220,50,50)), (0,3,7,4, (160,0,0)),
#                  (1,2,6,5, (160,0,0)), (0,1,5,4, (200,20,20)), (2,3,7,6, (200,20,20))]
#     face_list = []
#     for i1, i2, i3, i4, cor in faces_def:
#         pts = [proj[i] for i in (i1, i2, i3, i4)]
#         if any(p[0] is None for p in pts): continue
#         face_list.append((pts, cor, (depths[i1] + depths[i2] + depths[i3] + depths[i4]) / 4.0))
#     face_list.sort(key=lambda f: f[2], reverse=True)
#     return face_list

# def segment_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
#     denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
#     if abs(denom) < 1e-12: return None
#     t = ((x1 - x3)*(y3 - y4) - (y1 - y3)*(x3 - x4)) / denom
#     u = -((x1 - x2)*(y1 - y3) - (y1 - y2)*(x1 - x3)) / denom
#     if 0 <= t <= 1 and 0 <= u <= 1: return t
#     return None

# def adjust_camera(car_x, car_y, car_theta, desired_dist, walls, min_dist=0.3, wall_margin=0.15):
#     cam_x = car_x - desired_dist * math.cos(car_theta)
#     cam_y = car_y - desired_dist * math.sin(car_theta)
#     best_t = 1.0
#     for (wx1, wy1, wx2, wy2) in WALLS:
#         t = segment_intersection(car_x, car_y, cam_x, cam_y, wx1, wy1, wx2, wy2)
#         if t is not None and t < best_t: best_t = t
#     dist = max(min_dist, best_t * desired_dist)
#     cam_x = car_x - dist * math.cos(car_theta)
#     cam_y = car_y - dist * math.sin(car_theta)
#     return cam_x, cam_y

# def circle_segment_collision(cx, cy, radius, x1, y1, x2, y2):
#     dx, dy = x2 - x1, y2 - y1
#     fx, fy = cx - x1, cy - y1
#     t = (fx*dx + fy*dy) / (dx*dx + dy*dy + 1e-12)
#     t = max(0.0, min(1.0, t))
#     return math.hypot(cx - (x1 + t*dx), cy - (y1 + t*dy)) <= radius

# def main():
#     pygame.init()
#     WIDTH, HEIGHT = 900, 680
#     screen = pygame.display.set_mode((WIDTH, HEIGHT))
#     pygame.display.set_caption("Robo - Escuta Contínua Ativa (Sem Botões)")
#     clock = pygame.time.Clock()
    
#     font_bold = pygame.font.SysFont("Arial", 18, bold=True)
#     font_small = pygame.font.SysFont("Arial", 13)

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     video_writer = cv2.VideoWriter('gravacao_apresentacao.mp4', fourcc, 50, (WIDTH, HEIGHT))

#     dt = 0.02
#     b = 0.094 / 2.0
#     CAR_WIDTH, CAR_LENGTH, CAR_HEIGHT = 2.0 * b, 2.5 * b, 0.1
#     x, y, theta = -0.5, 2.5, 0.0
#     trail = [(x, y)]
#     prev_x = x

#     base_speed = 0.26
#     omega_rotate = 1.5

#     command = 'straight' 
#     historico_palavras_ditas = []
#     probabilidades_classes = np.zeros(len(label_map))

#     cam3d = Camera3D(WIDTH, HEIGHT - 140, focal_length=250, cam_height=0.4, horizon_y=int(HEIGHT*0.32))

#     MAP_WIDTH, MAP_HEIGHT = 180, 130
#     MAP_X, MAP_Y = WIDTH - MAP_WIDTH - 15, 15

#     running = True
#     finish_crossed = False
#     collision = False

#     while running:
#         clock.tick(50)

#         while not hud_update_queue.empty():
#             probabilidades_classes = hud_update_queue.get()

#         if not voice_command_queue.empty():
#             palavra = voice_command_queue.get()
#             pode_girar = command not in ('rot_left', 'rot_right', 'rot_180')
#             comando_valido = True
#             if palavra in ['siga', 'frente', 'ir']: 
#                 proximo_comando = 'straight'
#             elif palavra in ['esquerda', 'esq'] and pode_girar: 
#                 proximo_comando = 'rot_left'
#             elif palavra in ['direita', 'dir'] and pode_girar: 
#                 proximo_comando = 'rot_right'
#             elif palavra in ['pare', 'parar', 'stop']: 
#                 proximo_comando = 'stop'
#             elif palavra in ['voltar', 're', '180'] and pode_girar: 
#                 proximo_comando = 'rot_180'
#             else:
#                 comando_valido = False

#             if comando_valido:
#                 command = proximo_comando
#                 historico_palavras_ditas.append(palavra.upper())
#                 if len(historico_palavras_ditas) > 4: 
#                     historico_palavras_ditas.pop(0)

#                 if command in ('rot_left', 'rot_right', 'rot_180'):
#                     rot_start_theta = theta
#                     rot_accumulated = 0.0
#                     if command == 'rot_left': 
#                         rot_target_delta = math.pi / 2.0
#                         rot_omega_sign = 1.0
#                     elif command == 'rot_right': 
#                         rot_target_delta = -math.pi / 2.0
#                         rot_omega_sign = -1.0
#                     elif command == 'rot_180': 
#                         rot_target_delta = math.pi
#                         rot_omega_sign = 1.0
                        
#         if command == 'stop': 
#             v = 0.0
#             delta_theta = 0.0
#             estado_atual_nome = "PARADO"
#         elif command == 'straight': 
#             v = base_speed
#             delta_theta = 0.0
#             estado_atual_nome = "SIGA (EM FRENTE)"
#         elif command in ('rot_left', 'rot_right', 'rot_180'):
#             v = 0.0
#             proximo_passo = rot_omega_sign * omega_rotate * dt
            
#             if abs(rot_accumulated + proximo_passo) >= abs(rot_target_delta):
#                 theta = normalize_angle(rot_start_theta + rot_target_delta)
#                 command = 'straight'  
#                 delta_theta = 0.0
#                 estado_atual_nome = "SIGA (EM FRENTE)"
#             else:
#                 delta_theta = proximo_passo
#                 rot_accumulated += delta_theta
#                 theta += delta_theta
#                 estado_atual_nome = f"GIRANDO ({command.upper()})"

#         for event in pygame.event.get():
#             if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
#                 running = False

#         if command == 'stop': v = 0.0; delta_theta = 0.0
#         elif command == 'straight': v = base_speed; delta_theta = 0.0
#         elif command in ('rot_left', 'rot_right', 'rot_180'):
#             v = 0.0
#             delta_theta = rot_omega_sign * omega_rotate * dt
#             rot_accumulated += delta_theta
#             if abs(rot_accumulated) >= abs(rot_target_delta):
#                 theta = normalize_angle(rot_start_theta + rot_target_delta)
#                 command = 'straight' 
#                 delta_theta = 0.0
#             else: theta += delta_theta
#         estado_atual_nome = command.upper()

#         delta_x = dt * v * math.cos(theta) if command == 'straight' else 0.0
#         delta_y = dt * v * math.sin(theta) if command == 'straight' else 0.0
#         new_x, new_y = x + delta_x, y + delta_y

#         hit_wall = False
#         for (wx1, wy1, wx2, wy2) in WALLS:
#             if circle_segment_collision(new_x, new_y, ROBOT_RADIUS, wx1, wy1, wx2, wy2):
#                 hit_wall = True; break
#         if hit_wall: collision = True; running = False
#         else: x, y = new_x, new_y

#         if not finish_crossed and prev_x < FINISH_X and x >= FINISH_X:
#             if FINISH_Y_START <= y <= FINISH_Y_END: finish_crossed = True; running = False
#         prev_x = x
#         trail.append((x, y))

#         cam_x, cam_y = adjust_camera(x, y, theta, 0.7, WALLS)
#         cam3d.set_position(cam_x, cam_y)

#         screen.fill((0, 0, 0))
#         pygame.draw.rect(screen, (100, 149, 237), (0, 0, WIDTH, cam3d.horizon_y)) 
#         pygame.draw.rect(screen, (50, 50, 50), (0, cam3d.horizon_y, WIDTH, (HEIGHT - 140) - cam3d.horizon_y)) 

#         objects = []
#         for (wx1, wy1, wx2, wy2) in WALLS:
#             length = math.hypot(wx2 - wx1, wy2 - wy1)
#             if length < 0.01: continue
#             n_samples = max(2, int(length * 10))
#             prev_sx_base = prev_sy_base = prev_sx_top = prev_sy_top = None
#             prev_d_base = prev_d_top = 0.0
#             for i in range(n_samples + 1):
#                 t_w = i / n_samples
#                 px, py = wx1 + t_w * (wx2 - wx1), wy1 + t_w * (wy2 - wy1)
#                 sx_base, sy_base, d_base = cam3d.project(px, py, x, y, theta, obj_height=0.0)
#                 sx_top, sy_top, d_top = cam3d.project(px, py, x, y, theta, obj_height=WALL_HEIGHT)
#                 if sx_base is not None and sx_top is not None:
#                     if prev_sx_base is not None:
#                         pts = [(prev_sx_base, prev_sy_base), (sx_base, sy_base), (sx_top, sy_top), (prev_sx_top, prev_sy_top)]
#                         objects.append(((prev_d_base + d_base + d_top + prev_d_top) / 4.0, 'wall', pts))
#                     prev_sx_base, prev_sy_base = sx_base, sy_base
#                     prev_sx_top, prev_sy_top = sx_top, sy_top
#                     prev_d_base, prev_d_top = d_base, d_top
#                 else: prev_sx_base = None

#         objects.sort(key=lambda obj: obj[0], reverse=True)
#         for depth, typ, data in objects:
#             if typ == 'wall':
#                 pygame.draw.polygon(screen, (160, 160, 160), data)
#                 pygame.draw.polygon(screen, (30, 30, 30), data, 1)

#         car_faces = project_cuboid(cam3d, x, y, theta, x, y, theta, CAR_WIDTH, CAR_LENGTH, CAR_HEIGHT)
#         for pts, cor, _ in car_faces:
#             if len(pts) >= 3:
#                 pygame.draw.polygon(screen, cor, pts)
#                 pygame.draw.polygon(screen, (0, 0, 0), pts, 1)

#         minimap_surf = pygame.Surface((MAP_WIDTH, MAP_HEIGHT))
#         minimap_surf.fill((15, 15, 15))
#         scale_x = MAP_WIDTH / 6.0
#         scale_y = MAP_HEIGHT / 5.0
#         def world_to_map(wx, wy):
#             return int((wx - (-1.0)) * scale_x), int((3.0 - wy) * scale_y)

#         for (wx1, wy1, wx2, wy2) in WALLS:
#             pygame.draw.line(minimap_surf, (100, 100, 100), world_to_map(wx1, wy1), world_to_map(wx2, wy2), 1)
#         if len(trail) > 1:
#             pygame.draw.lines(minimap_surf, (0, 255, 0), False, [world_to_map(px, py) for px, py in trail], 1)
#         pygame.draw.circle(minimap_surf, (255, 0, 0), world_to_map(x, y), 4)
#         screen.blit(minimap_surf, (MAP_X, MAP_Y))

#         hud_top_y = HEIGHT - 140
#         pygame.draw.rect(screen, (20, 20, 30), (0, hud_top_y, WIDTH, 140))
        
#         pygame.draw.line(screen, (0, 255, 150), (0, hud_top_y), (WIDTH, hud_top_y), 2)
#         lbl_grav = font_bold.render("ESCUTA CONTÍNUA ATIVA (FALE SEUS COMANDOS)...", True, (0, 255, 150))
#         screen.blit(lbl_grav, (20, hud_top_y + 70))

#         lbl_historico = font_bold.render("HISTORICO DO LEITOR DE VOZ (1D MODEL - 1.5S):", True, (240, 240, 240))
#         screen.blit(lbl_historico, (20, hud_top_y + 15))
        
#         texto_linha_historico = " -> ".join(historico_palavras_ditas) if historico_palavras_ditas else "Aguardando comandos de voz..."
#         lbl_linha = font_small.render(texto_linha_historico, True, (0, 255, 150))
#         screen.blit(lbl_linha, (20, hud_top_y + 42))

#         lbl_estado = font_small.render(f"Fisica do Carrinho: {estado_atual_nome}", True, (180, 180, 180))
#         screen.blit(lbl_estado, (20, hud_top_y + 95))

#         bar_start_x = 480
#         bar_start_y = hud_top_y + 12
#         for c_name, c_idx in sorted(label_map.items(), key=lambda item: item[1]):
#             prob = probabilidades_classes[c_idx] if c_idx < len(probabilidades_classes) else 0.0
#             text_c = font_small.render(f"{c_name.upper():<9}", True, (230, 230, 230))
#             screen.blit(text_c, (bar_start_x, bar_start_y))
#             pygame.draw.rect(screen, (50, 50, 60), (bar_start_x + 80, bar_start_y + 2, 200, 12))
#             bar_color = (0, 255, 150) if prob > 0.75 else (255, 165, 0) if prob > 0.30 else (100, 100, 110)
#             pygame.draw.rect(screen, bar_color, (bar_start_x + 80, bar_start_y + 2, int(prob * 200), 12))
#             text_p = font_small.render(f"{prob * 100:.1f}%", True, (200, 200, 200))
#             screen.blit(text_p, (bar_start_x + 290, bar_start_y))
#             bar_start_y += 22

#         pygame.display.flip()

#         view = pygame.surfarray.array3d(screen)
#         view = view.transpose([1, 0, 2])
#         view = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
#         video_writer.write(view)

#     pygame.quit()
#     video_writer.release()

#     if audio_sessao_completa:
#         audio_array = np.array(audio_sessao_completa, dtype=np.float32)
#         audio_int16 = (audio_array * 32767.0).astype(np.int16)
#         wav.write("historico_completo.wav", SAMPLE_RATE, audio_int16)

#     if finish_crossed: print("\nVitoria! O carrinho alcancou o final do percurso.")
#     if collision: print("\nBatida violenta na parede detectada!")

# if __name__ == "__main__":
#     main()