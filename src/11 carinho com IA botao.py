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
# from scipy.io import wavfile
# import pygame
# import matplotlib.pyplot as plt

# # ======================================================================
# # 1. CONFIGURAÇÕES DO MODELO E ÁUDIO CONTÍNUO
# # ======================================================================
# SAMPLE_RATE = 16000
# DURATION = 1  
# TARGET_SIZE = 64 
# STEP_SIZE = 3200        # Bloco de análise a cada 200ms
# VOLUME_THRESHOLD = 0.02 # Limiar para ignorar silêncio/ruídos muito baixos

# MODEL_PATH = "modelo_cadencia_lstm_federado_manual_2.h5" 
# if not os.path.exists(MODEL_PATH):
#     print(f"ERRO: O arquivo '{MODEL_PATH}' não foi encontrado.")
#     sys.exit()

# model_audio = tf.keras.models.load_model(MODEL_PATH)

# with open("labels_cadencia_lstm.json", "r") as f:
#     label_map = json.load(f)
# inv_label_map = {v: k for k, v in label_map.items()}

# # Filas de comunicação entre as threads
# voice_command_queue = queue.Queue()
# hud_update_queue = queue.Queue()

# # Listas globais para salvar todo o áudio da sessão para análise posterior
# audio_session_records = []
# record_lock = threading.Lock()

# # ======================================================================
# # 2. FUNÇÕES DE PRÉ-PROCESSAMENTO MATEMÁTICO
# # ======================================================================
# def resize_matriz_manual(matriz, target_size):
#     orig_h, orig_w = matriz.shape
#     row_indices = (np.arange(target_size) * (orig_h / target_size)).astype(np.int32)
#     col_indices = (np.arange(target_size) * (orig_w / target_size)).astype(np.int32)
#     return matriz[row_indices[:, None], col_indices]

# def calcular_cadencia_manual(audio, threshold_percentage=0.2):
#     audio_2k = audio[::8]
#     matriz_distancias = np.abs(audio_2k[:, None] - audio_2k[None, :])
#     limiar = np.percentile(matriz_distancias, threshold_percentage * 100)
#     matriz_recorrencia = (matriz_distancias <= limiar).astype(np.float32)
#     matriz_redimensionada = resize_matriz_manual(matriz_recorrencia, TARGET_SIZE)
#     return np.expand_dims(matriz_redimensionada, axis=-1)

# # ======================================================================
# # 3. THREAD DE CAPTURA CONTÍNUA E GRAVAÇÃO COMPLETA
# # ======================================================================
# def audio_stream_thread():
#     buffer_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    
#     def audio_callback(indata, frames, time_info, status):
#         nonlocal buffer_audio
#         data_chunk = indata[:, 0].copy()
        
#         # Guarda o trecho de áudio bruto para salvar no arquivo final ao colidir/vencer
#         with record_lock:
#             audio_session_records.append(data_chunk)
            
#         # Alimenta o buffer deslizante de 1 segundo para a IA
#         buffer_audio = np.roll(buffer_audio, -len(data_chunk))
#         buffer_audio[-len(data_chunk):] = data_chunk

#     with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=STEP_SIZE):
#         last_triggered_time = 0
#         while True:
#             time.sleep(0.05)
            
#             # Debounce de 1.2s para evitar comandos repetidos atropelando a física
#             if time.time() - last_triggered_time < 1.2:
#                 continue
                
#             audio_window = buffer_audio.copy()
            
#             if np.max(np.abs(audio_window)) < VOLUME_THRESHOLD:
#                 continue
                
#             # Conversão float32 -> int16 (Garante o comportamento idêntico ao celular)
#             audio_int16 = (audio_window * 32767.0).astype(np.float32)
            
#             if np.std(audio_int16) > 0:
#                 audio_norm = (audio_int16 - np.mean(audio_int16)) / np.std(audio_int16)
#             else:
#                 audio_norm = audio_int16
                
#             matriz_cadencia = calcular_cadencia_manual(audio_norm)
#             input_neural = np.expand_dims(matriz_cadencia, axis=0)
            
#             preds = model_audio.predict(input_neural, verbose=0)[0]
#             idx_classe = np.argmax(preds)
#             confianca = preds[idx_classe]
            
#             # Atualiza o HUD continuamente com as probabilidades atuais de cada classe
#             hud_update_queue.put(preds)
            
#             # Filtro de confiança estrito para o modo contínuo (75%)
#             if confianca > 0.75:
#                 palavra_detectada = inv_label_map[idx_classe].lower()
#                 voice_command_queue.put(palavra_detectada)
#                 last_triggered_time = time.time()

# threading.Thread(target=audio_stream_thread, daemon=True).start()

# # ======================================================================
# # 4. PARÂMETROS DO LABIRINTO (DO SEU MODELO ORIGINAL)
# # ======================================================================
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
#     def __init__(self, width, height, focal_length=250, cam_height=0.8, horizon_y=None):
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

# def segment_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
#     denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
#     if abs(denom) < 1e-12: return None
#     t = ((x1 - x3)*(y3 - y4) - (y1 - y3)*(x3 - x4)) / denom
#     u = -((x1 - x2)*(y1 - y3) - (y1 - y2)*(x1 - x3)) / denom
#     if 0 <= t <= 1 and 0 <= u <= 1: return t
#     return None

# def point_segment_distance(px, py, x1, y1, x2, y2):
#     dx, dy = x2 - x1, y2 - y1
#     fx, fy = px - x1, py - y1
#     t = (fx*dx + fy*dy) / (dx*dx + dy*dy + 1e-12)
#     t = max(0.0, min(1.0, t))
#     return math.hypot(px - (x1 + t*dx), py - (y1 + t*dy))

# def adjust_camera(car_x, car_y, car_theta, desired_dist, walls, min_dist=0.3, wall_margin=0.15):
#     cam_x = car_x - desired_dist * math.cos(car_theta)
#     cam_y = car_y - desired_dist * math.sin(car_theta)
#     best_t = 1.0
#     for (wx1, wy1, wx2, wy2) in walls:
#         t = segment_intersection(car_x, car_y, cam_x, cam_y, wx1, wy1, wx2, wy2)
#         if t is not None and t < best_t: best_t = t
#     dist = max(min_dist, best_t * desired_dist)
#     cam_x = car_x - dist * math.cos(car_theta)
#     cam_y = car_y - dist * math.sin(car_theta)
#     for _ in range(5):
#         for (wx1, wy1, wx2, wy2) in walls:
#             d = point_segment_distance(cam_x, cam_y, wx1, wy1, wx2, wy2)
#             if d < wall_margin:
#                 dx, dy = wx2 - wx1, wy2 - wy1
#                 length = math.hypot(dx, dy)
#                 if length < 1e-6: continue
#                 nx, ny = -dy / length, dx / length
#                 push = wall_margin - d
#                 if (cam_x - wx1) * nx + (cam_y - wy1) * ny < 0:
#                     nx, ny = -nx, -ny
#                 cam_x += nx * push; cam_y += ny * push
#     return cam_x, cam_y

# def circle_segment_collision(cx, cy, radius, x1, y1, x2, y2):
#     dx, dy = x2 - x1, y2 - y1
#     fx, fy = cx - x1, cy - y1
#     t = (fx*dx + fy*dy) / (dx*dx + dy*dy + 1e-12)
#     t = max(0.0, min(1.0, t))
#     return math.hypot(cx - (x1 + t*dx), cy - (y1 + t*dy)) <= radius

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

# # ======================================================================
# # 5. LOOP DE EXECUÇÃO DO SIMULADOR (INTERATIVO)
# # ======================================================================
# def main():
#     pygame.init()
#     WIDTH, HEIGHT = 900, 680 # Tela ligeiramente maior para caber as barras de confiança
#     screen = pygame.display.set_mode((WIDTH, HEIGHT))
#     pygame.display.set_caption("Robô - Diagnóstico de Voz Completo")
#     clock = pygame.time.Clock()
    
#     font_bold = pygame.font.SysFont("Arial", 18, bold=True)
#     font_small = pygame.font.SysFont("Arial", 13)

#     dt = 0.02
#     b = 0.094 / 2.0
#     CAR_WIDTH, CAR_LENGTH, CAR_HEIGHT = 2.0 * b, 2.5 * b, 0.1
#     x, y, theta = -0.5, 2.5, 0.0
#     trail = [(x, y)]
#     prev_x = x

#     # MELHORIA: Velocidade base reduzida em um terço para melhor controle (0.4 / 1.5 ~= 0.26)
#     base_speed = 0.26
#     omega_rotate = 1.5

#     command = 'stop'
#     historico_palavras_ditas = [] # Leitor de voz inferior
#     probabilidades_classes = np.zeros(len(label_map)) # Barras de confiança

#     desired_cam_dist = 0.8
#     cam3d = Camera3D(WIDTH, HEIGHT - 140, focal_length=250, cam_height=0.8, horizon_y=int(HEIGHT*0.35))

#     MAP_WIDTH, MAP_HEIGHT = 180, 130
#     MAP_X, MAP_Y = WIDTH - MAP_WIDTH - 15, 15

#     running = True
#     finish_crossed = False
#     collision = False

#     while running:
#         clock.tick(50)

#         # Atualiza as barras de confiança dinâmicas vindas da IA
#         while not hud_update_queue.empty():
#             probabilidades_classes = hud_update_queue.get()

#         # Verifica se um comando de voz válido foi aceito
#         if not voice_command_queue.empty():
#             palavra = voice_command_queue.get()
            
#             # Adiciona ao leitor de voz do que já foi falado (mantém as últimas 4)
#             historico_palavras_ditas.append(palavra.upper())
#             if len(historico_palavras_ditas) > 4:
#                 historico_palavras_ditas.pop(0)
            
#             if palavra in ['siga', 'frente', 'ir']:
#                 command = 'straight'
#             elif palavra in ['esquerda', 'esq']:
#                 command = 'rot_left'
#             elif palavra in ['direita', 'dir']:
#                 command = 'rot_right'
#             elif palavra in ['pare', 'parar', 'stop']:
#                 command = 'stop'
#             elif palavra in ['voltar', 're', '180']:
#                 command = 'rot_180'

#             if command in ('rot_left', 'rot_right', 'rot_180'):
#                 rot_start_theta = theta
#                 rot_accumulated = 0.0
#                 if command == 'rot_left':
#                     rot_target_delta = math.pi / 2.0
#                     rot_omega_sign = 1.0
#                 elif command == 'rot_right':
#                     rot_target_delta = -math.pi / 2.0
#                     rot_omega_sign = -1.0
#                 elif command == 'rot_180':
#                     rot_target_delta = math.pi
#                     rot_omega_sign = 1.0

#         for event in pygame.event.get():
#             if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
#                 running = False

#         # Física do Movimento
#         if command == 'stop':
#             v = 0.0; delta_theta = 0.0
#         elif command == 'straight':
#             v = base_speed; delta_theta = 0.0
#             theta = normalize_angle(theta)
#         elif command in ('rot_left', 'rot_right', 'rot_180'):
#             v = 0.0
#             delta_theta = rot_omega_sign * omega_rotate * dt
#             rot_accumulated += delta_theta
#             if abs(rot_accumulated) >= abs(rot_target_delta):
#                 theta = normalize_angle(rot_start_theta + rot_target_delta)
#                 command = 'straight'
#                 delta_theta = 0.0
#             else:
#                 theta += delta_theta

#         delta_x = dt * v * math.cos(theta) if command == 'straight' else 0.0
#         delta_y = dt * v * math.sin(theta) if command == 'straight' else 0.0
#         new_x, new_y = x + delta_x, y + delta_y

#         hit_wall = False
#         for (wx1, wy1, wx2, wy2) in WALLS:
#             if circle_segment_collision(new_x, new_y, ROBOT_RADIUS, wx1, wy1, wx2, wy2):
#                 hit_wall = True; break
#         if hit_wall:
#             collision = True; running = False
#         else:
#             x, y = new_x, new_y

#         if not finish_crossed and prev_x < FINISH_X and x >= FINISH_X:
#             if FINISH_Y_START <= y <= FINISH_Y_END:
#                 finish_crossed = True; running = False
#         prev_x = x
#         trail.append((x, y))

#         cam_x, cam_y = adjust_camera(x, y, theta, desired_cam_dist, WALLS)
#         cam3d.set_position(cam_x, cam_y)

#         # Renderização da Janela 3D
#         screen.fill((0, 0, 0))
#         pygame.draw.rect(screen, (135, 206, 235), (0, 0, WIDTH, cam3d.horizon_y))
#         pygame.draw.rect(screen, (34, 139, 34), (0, cam3d.horizon_y, WIDTH, (HEIGHT - 140) - cam3d.horizon_y))

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
#             if typ == 'wall' and all(0 <= p[0] <= WIDTH for p in data):
#                 pygame.draw.polygon(screen, (180, 180, 180), data)
#                 pygame.draw.polygon(screen, (0,0,0), data, 1)

#         car_faces = project_cuboid(cam3d, x, y, theta, x, y, theta, CAR_WIDTH, CAR_LENGTH, CAR_HEIGHT)
#         for pts, cor, _ in car_faces:
#             if len(pts) >= 3 and all(0 <= p[0] <= WIDTH for p in pts):
#                 pygame.draw.polygon(screen, cor, pts)

#         # Renderização do Minimapa 2D
#         minimap_surf = pygame.Surface((MAP_WIDTH, MAP_HEIGHT))
#         minimap_surf.fill((10, 10, 10))
#         scale_x, scale_y = MAP_WIDTH / 6.0, MAP_HEIGHT / 5.0
#         def world_to_map(wx, wy):
#             return int((wx - (-1.0)) * scale_x), int((3.0 - wy) * scale_y)

#         for (wx1, wy1, wx2, wy2) in WALLS:
#             pygame.draw.line(minimap_surf, (150, 150, 150), world_to_map(wx1, wy1), world_to_map(wx2, wy2), 1)
#         if len(trail) > 1:
#             pygame.draw.lines(minimap_surf, (0, 255, 0), False, [world_to_map(px, py) for px, py in trail], 1)
#         pygame.draw.circle(minimap_surf, (255, 0, 0), world_to_map(x, y), 4)
#         screen.blit(minimap_surf, (MAP_X, MAP_Y))

#         # ======================================================================
#         # 6. REGIAO INFERIOR: HUD DE CONFIANÇA E LEITOR DE VOZ DETALHADO
#         # ======================================================================
#         hud_top_y = HEIGHT - 140
#         pygame.draw.rect(screen, (20, 20, 30), (0, hud_top_y, WIDTH, 140))
#         pygame.draw.line(screen, (0, 255, 150), (0, hud_top_y), (WIDTH, hud_top_y), 2)

#         # Lado Esquerdo: Leitor de Voz (Histórico de comandos já captados)
#         lbl_historico = font_bold.render("LEITOR DE VOZ (HISTÓRICO CONTÍNUO):", True, (240, 240, 240))
#         screen.blit(lbl_historico, (20, hud_top_y + 15))
        
#         texto_linha_historico = " -> ".join(historico_palavras_ditas) if historico_palavras_ditas else "Nenhum som captado com precisão..."
#         lbl_linha = font_small.render(texto_linha_historico, True, (0, 255, 150) if historico_palavras_ditas else (130, 130, 140))
#         screen.blit(lbl_linha, (20, hud_top_y + 45))

#         lbl_estado = font_small.render(f"Estado Físico: {command.upper()} | Velocidade: {base_speed} m/s", True, (180, 180, 180))
#         screen.blit(lbl_estado, (20, hud_top_y + 95))

#         # Lado Direito: Barras Dinâmicas de Confiança por Classe
#         bar_start_x = 480
#         bar_start_y = hud_top_y + 15
        
#         for c_name, c_idx in sorted(label_map.items(), key=lambda item: item[1]):
#             prob = probabilidades_classes[c_idx] if c_idx < len(probabilidades_classes) else 0.0
            
#             # Texto da Classe
#             text_c = font_small.render(f"{c_name.upper():<9}", True, (230, 230, 230))
#             screen.blit(text_c, (bar_start_x, bar_start_y))
            
#             # Fundo da Barra
#             pygame.draw.rect(screen, (50, 50, 60), (bar_start_x + 80, bar_start_y + 2, 200, 12))
#             # Preenchimento Proporcional à Probabilidade da Rede
#             bar_color = (0, 255, 150) if prob > 0.75 else (255, 165, 0) if prob > 0.3 else (100, 100, 110)
#             pygame.draw.rect(screen, bar_color, (bar_start_x + 80, bar_start_y + 2, int(prob * 200), 12))
            
#             # Porcentagem numérica
#             text_p = font_small.render(f"{prob * 100:.1f}%", True, (200, 200, 200))
#             screen.blit(text_p, (bar_start_x + 290, bar_start_y))
            
#             bar_start_y += 22

#         pygame.display.flip()

#     pygame.quit()

#     # ======================================================================
#     # 7. ENCERRAMENTO: GRAVAÇÃO DO ÁUDIO COMPLETO DA SESSÃO EM ARQUIVO
#     # ======================================================================
#     print("\n[SALVAMENTO AUTOMÁTICO] Compilando áudio da sessão...")
#     with record_lock:
#         if audio_session_records:
#             full_audio_session = np.concatenate(audio_session_records)
#             # Converte de volta para int16 para que o arquivo .wav final fique audível e padronizado
#             audio_to_save = (full_audio_session * 32767.0).astype(np.int16)
            
#             filename = "historico_audio_simulacao.wav"
#             wavfile.write(filename, SAMPLE_RATE, audio_to_save)
#             print(f"-> Sucesso! Todo o áudio capturado foi gravado em '{filename}'.")
#             print("Abra esse arquivo para verificar se o microfone está pegando muito ruído de fundo ou eco!")
#         else:
#             print("Nenhum sinal de áudio foi registrado.")

#     if finish_crossed: print("\n🏁 Vitória! O carro cruzou a linha de chegada.")
#     if collision: print("\n💥 Batida detectada na parede!")

# if __name__ == "__main__":
#     main()

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
import keyboard

# ======================================================================
# 1. DECLARAÇÃO DA CAMADA CUSTOMIZADA DE ATENÇÃO (OBRIGATÓRIO)
# ======================================================================
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

# ======================================================================
# 2. CONFIGURAÇÕES DO MODELO 1D E PARÂMETROS DE ÁUDIO
# ======================================================================
SAMPLE_RATE = 16000
DURATION = 1.0  

MODEL_PATH = "modelo_hibrido_com_atencao.h5" 
if not os.path.exists(MODEL_PATH):
    print(f"ERRO: O arquivo '{MODEL_PATH}' não foi encontrado no diretório.")
    sys.exit()

# Carrega o modelo injetando a camada customizada para evitar erros de compilação
model_audio = tf.keras.models.load_model(
    MODEL_PATH, 
    custom_objects={"AttentionLayer": AttentionLayer}
)

# Mapeamento de rótulos (Certifique-se de que o arquivo .json existe com as classes corretas)
with open("label_map_hibrido.json", "r") as f:
    label_map = json.load(f)
inv_label_map = {v: k for k, v in label_map.items()}

voice_command_queue = queue.Queue()
hud_update_queue = queue.Queue()

is_recording = False

# ======================================================================
# 3. THREAD DE CAPTURA DE ÁUDIO (IDÊNTICA AO SEU VALIDADOR 00 TEST...)
# ======================================================================
def audio_capture_thread():
    global is_recording
    while True:
        # Quando a tecla 'g' for pressionada
        if keyboard.is_pressed('g') and not is_recording:
            is_recording = True
            
            # Aguarda o usuário soltar a tecla para não captar o estalo mecânico do teclado
            while keyboard.is_pressed('g'):
                time.sleep(0.01)
                
            print("\n[GRAVANDO] Fale o comando agora...")
            
            # Coleta idêntica e bloqueante do validador funcional
            audio_float32 = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            
            is_recording = False
            print("[GRAVANDO] Processando inferência...")
            
            audio_raw = audio_float32.flatten()
            
            # Conversão matemática Float32 -> Int16 antes do Z-score
            audio_int16 = (audio_raw * 32767.0).astype(np.float32)
            
            # Aplicação exata do Z-score do pipeline de validação
            if np.std(audio_int16) > 0:
                audio_preprocessed = (audio_int16 - np.mean(audio_int16)) / np.std(audio_int16)
            else:
                audio_preprocessed = audio_int16
                
            # Formatação para a entrada do modelo 1D híbrido: (1, 16000, 1)
            input_neural = np.expand_dims(audio_preprocessed, axis=(0, -1))
            
            # Predição da rede neural
            preds = model_audio.predict(input_neural, verbose=0)[0]
            hud_update_queue.put(preds) # Atualiza o HUD gráfico
            
            idx_classe = np.argmax(preds)
            confianca = preds[idx_classe]
            
            # Filtro de confiança adaptado para o botão
            if confianca > 0.55:
                palavra_detectada = inv_label_map[idx_classe].lower()
                voice_command_queue.put(palavra_detectada)
                
        time.sleep(0.05)

threading.Thread(target=audio_capture_thread, daemon=True).start()

# ======================================================================
# 4. CONFIGURAÇÃO DO LABIRINTO (PROJEÇÃO DO SEU MODELO ORIGINAL)
# ======================================================================
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

# ======================================================================
# 5. LOOP DE EXECUÇÃO PRINCIPAL (INTERATIVO PYGAME)
# ======================================================================
def main():
    global is_recording
    pygame.init()
    WIDTH, HEIGHT = 900, 680
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Robô - Modelo de Atenção 1D Híbrido por Botão 'G'")
    clock = pygame.time.Clock()
    
    font_bold = pygame.font.SysFont("Arial", 18, bold=True)
    font_small = pygame.font.SysFont("Arial", 13)

    dt = 0.02
    b = 0.094 / 2.0
    CAR_WIDTH, CAR_LENGTH, CAR_HEIGHT = 2.0 * b, 2.5 * b, 0.1
    x, y, theta = -0.5, 2.5, 0.0
    trail = [(x, y)]
    prev_x = x

    # Velocidade reduzida em um terço para perfeita cadência de comando físico
    base_speed = 0.26
    omega_rotate = 1.5

    command = 'straight' # Inicia se movendo
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

        # Trata os scores dinâmicos de confiança vindos do modelo 1D
        while not hud_update_queue.empty():
            probabilidades_classes = hud_update_queue.get()

        # Verifica novas instruções na fila de voz pós-processamento do botão
        if not voice_command_queue.empty():
            palavra = voice_command_queue.get()
            historico_palavras_ditas.append(palavra.upper())
            if len(historico_palavras_ditas) > 4: historico_palavras_ditas.pop(0)
            
            if palavra in ['siga', 'frente', 'ir']: command = 'straight'
            elif palavra in ['esquerda', 'esq']: command = 'rot_left'
            elif palavra in ['direita', 'dir']: command = 'rot_right'
            elif palavra in ['pare', 'parar', 'stop']: command = 'stop'
            elif palavra in ['voltar', 're', '180']: command = 'rot_180'

            if command in ('rot_left', 'rot_right', 'rot_180'):
                rot_start_theta = theta
                rot_accumulated = 0.0
                if command == 'rot_left': rot_target_delta = math.pi / 2.0; rot_omega_sign = 1.0
                elif command == 'rot_right': rot_target_delta = -math.pi / 2.0; rot_omega_sign = -1.0
                elif command == 'rot_180': rot_target_delta = math.pi; rot_omega_sign = 1.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        # Lógica de Controle da Física: Trava mecânica imediata se estiver gravando
        if is_recording or keyboard.is_pressed('g'):
            v = 0.0; delta_theta = 0.0
            estado_atual_nome = "GRAVANDO ÁUDIO (CARRINHO PARADO)..."
        else:
            if command == 'stop': v = 0.0; delta_theta = 0.0
            elif command == 'straight': v = base_speed; delta_theta = 0.0
            elif command in ('rot_left', 'rot_right', 'rot_180'):
                v = 0.0
                delta_theta = rot_omega_sign * omega_rotate * dt
                rot_accumulated += delta_theta
                if abs(rot_accumulated) >= abs(rot_target_delta):
                    theta = normalize_angle(rot_start_theta + rot_target_delta)
                    command = 'straight' # Retoma marcha automática reta após curva
                    delta_theta = 0.0
                else: theta += delta_theta
            estado_atual_nome = command.upper()

        delta_x = dt * v * math.cos(theta) if (command == 'straight' and not is_recording) else 0.0
        delta_y = dt * v * math.sin(theta) if (command == 'straight' and not is_recording) else 0.0
        new_x, new_y = x + delta_x, y + delta_y

        # Tratamento de Colisão com as Paredes
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

        # Atualiza Posicionamento da Câmera 3D
        cam_x, cam_y = adjust_camera(x, y, theta, 0.7, WALLS)
        cam3d.set_position(cam_x, cam_y)

        # RENDERIZAÇÃO AMBIENTE GRÁFICO 3D
        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, (100, 149, 237), (0, 0, WIDTH, cam3d.horizon_y)) # Horizonte / Céu
        pygame.draw.rect(screen, (50, 50, 50), (0, cam3d.horizon_y, WIDTH, (HEIGHT - 140) - cam3d.horizon_y)) # Chão

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

        # Projeta o carrinho Vermelho em 3D
        car_faces = project_cuboid(cam3d, x, y, theta, x, y, theta, CAR_WIDTH, CAR_LENGTH, CAR_HEIGHT)
        for pts, cor, _ in car_faces:
            if len(pts) >= 3:
                pygame.draw.polygon(screen, cor, pts)
                pygame.draw.polygon(screen, (0, 0, 0), pts, 1)

        # MINIMAPA 2D SUPERIOR DIREITO
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

        # ======================================================================
        # INTERFACE HUD SUPERIOR DO DIAGNÓSTICO (PAINEL INFERIOR)
        # ======================================================================
        hud_top_y = HEIGHT - 140
        pygame.draw.rect(screen, (20, 20, 30), (0, hud_top_y, WIDTH, 140))
        
        # Alerta visual se estiver gravando ou com o botão engajado
        if is_recording or keyboard.is_pressed('g'):
            pygame.draw.line(screen, (255, 50, 50), (0, hud_top_y), (WIDTH, hud_top_y), 3)
            lbl_grav = font_bold.render("🎤 SECOLE/SOLTE 'G' E FALE IMEDIATAMENTE (1 SEGUNDO DE GRAVAÇÃO)...", True, (255, 50, 50))
            screen.blit(lbl_grav, (20, hud_top_y + 70))
        else:
            pygame.draw.line(screen, (0, 255, 150), (0, hud_top_y), (WIDTH, hud_top_y), 2)

        lbl_historico = font_bold.render("HISTÓRICO DO LEITOR DE VOZ (1D MODEL):", True, (240, 240, 240))
        screen.blit(lbl_historico, (20, hud_top_y + 15))
        
        texto_linha_historico = " -> ".join(historico_palavras_ditas) if historico_palavras_ditas else "Mantenha 'G' pressionado, solte e fale..."
        lbl_linha = font_small.render(texto_linha_historico, True, (0, 255, 150))
        screen.blit(lbl_linha, (20, hud_top_y + 42))

        lbl_estado = font_small.render(f"Física do Carrinho: {estado_atual_nome}", True, (180, 180, 180))
        screen.blit(lbl_estado, (20, hud_top_y + 95))

        # Barras de Confiança Gráficas alinhadas à direita
        bar_start_x = 480
        bar_start_y = hud_top_y + 12
        for c_name, c_idx in sorted(label_map.items(), key=lambda item: item[1]):
            prob = probabilidades_classes[c_idx] if c_idx < len(probabilidades_classes) else 0.0
            text_c = font_small.render(f"{c_name.upper():<9}", True, (230, 230, 230))
            screen.blit(text_c, (bar_start_x, bar_start_y))
            pygame.draw.rect(screen, (50, 50, 60), (bar_start_x + 80, bar_start_y + 2, 200, 12))
            bar_color = (0, 255, 150) if prob > 0.60 else (255, 165, 0) if prob > 0.25 else (100, 100, 110)
            pygame.draw.rect(screen, bar_color, (bar_start_x + 80, bar_start_y + 2, int(prob * 200), 12))
            text_p = font_small.render(f"{prob * 100:.1f}%", True, (200, 200, 200))
            screen.blit(text_p, (bar_start_x + 290, bar_start_y))
            bar_start_y += 22

        pygame.display.flip()

    pygame.quit()
    if finish_crossed: print("\n🏁 Vitória! O carrinho alcançou o final do percurso.")
    if collision: print("\n💥 Batida violenta na parede detectada!")

if __name__ == "__main__":
    main()