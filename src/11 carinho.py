import math
import pygame
import matplotlib.pyplot as plt


def sinc_safe(z: float) -> float:
    if abs(z) < 1e-9:
        return 1.0 - z * z / 6.0
    return math.sin(z) / z

def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


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
    (-1, 1, 1, 1),
    (2, 3, 2, 2),
    (2, 1, 2, -1),
    (0, -1, 3, -1),
    (3, 1, 5, 1),
    (4, -1, 5, -1)
]


class Camera3D:
    def __init__(self, width, height, focal_length=250, cam_height=0.8, horizon_y=None):
        self.width = width
        self.height = height
        self.focal_length = focal_length
        self.cam_height = cam_height
        self.horizon_y = horizon_y if horizon_y is not None else int(height * 0.4)
        self.center_x = width / 2
        self.cam_x = 0.0
        self.cam_y = 0.0

    def set_position(self, cam_x, cam_y):
        self.cam_x = cam_x
        self.cam_y = cam_y

    def project(self, wx, wy, car_x, car_y, car_theta, obj_height=0.0):
        # Usa a posição da câmera armazenada
        cam_x, cam_y = self.cam_x, self.cam_y
        cam_z = self.cam_height

        dx = wx - cam_x
        dy = wy - cam_y
        dz = obj_height - cam_z

        cos_t = math.cos(car_theta)
        sin_t = math.sin(car_theta)
        frente = dx * cos_t + dy * sin_t
        lateral = -dx * sin_t + dy * cos_t
        vertical = dz

        if frente <= 0.01:
            return None, None, None

        inv_depth = 1.0 / frente
        sx = self.center_x - lateral * inv_depth * self.focal_length
        sy = self.horizon_y - vertical * inv_depth * self.focal_length
        return sx, sy, frente


def segment_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    """Retorna t no segmento (x1,y1)->(x2,y2) da interseção, ou None."""
    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if abs(denom) < 1e-12:
        return None
    t = ((x1 - x3)*(y3 - y4) - (y1 - y3)*(x3 - x4)) / denom
    u = -((x1 - x2)*(y1 - y3) - (y1 - y2)*(x1 - x3)) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        return t
    return None

def point_segment_distance(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    fx = px - x1
    fy = py - y1
    t = (fx*dx + fy*dy) / (dx*dx + dy*dy + 1e-12)
    t = max(0.0, min(1.0, t))
    closest_x = x1 + t*dx
    closest_y = y1 + t*dy
    return math.hypot(px - closest_x, py - closest_y)

def adjust_camera(car_x, car_y, car_theta, desired_dist, walls, min_dist=0.3, wall_margin=0.15):
    """Retorna a posição final da câmera, ajustada para não entrar nas paredes."""
    # Posição desejada
    cam_x = car_x - desired_dist * math.cos(car_theta)
    cam_y = car_y - desired_dist * math.sin(car_theta)


    best_t = 1.0
    for (wx1, wy1, wx2, wy2) in walls:
        t = segment_intersection(car_x, car_y, cam_x, cam_y, wx1, wy1, wx2, wy2)
        if t is not None and t < best_t:
            best_t = t
    # Aplica distância reduzida
    dist = max(min_dist, best_t * desired_dist)
    cam_x = car_x - dist * math.cos(car_theta)
    cam_y = car_y - dist * math.sin(car_theta)


    for _ in range(5):  # iterações para resolver múltiplas paredes
        for (wx1, wy1, wx2, wy2) in walls:
            d = point_segment_distance(cam_x, cam_y, wx1, wy1, wx2, wy2)
            if d < wall_margin:
           
                dx = wx2 - wx1
                dy = wy2 - wy1
                length = math.hypot(dx, dy)
                if length < 1e-6:
                    continue
                nx = -dy / length
                ny = dx / length
                # Empurra na direção da normal (para longe da parede)
                push = wall_margin - d
                # Verifica de que lado a câmera está: sinal da projeção
                vx = cam_x - wx1
                vy = cam_y - wy1
                side = vx * nx + vy * ny
                if side < 0:
                    nx = -nx
                    ny = -ny
                cam_x += nx * push
                cam_y += ny * push
      
        d_car = math.hypot(cam_x - car_x, cam_y - car_y)
        if d_car < min_dist:
            cam_x = car_x + (cam_x - car_x) * min_dist / d_car
            cam_y = car_y + (cam_y - car_y) * min_dist / d_car

    return cam_x, cam_y


def circle_segment_collision(cx, cy, radius, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    fx = cx - x1
    fy = cy - y1
    t = (fx*dx + fy*dy) / (dx*dx + dy*dy + 1e-12)
    t = max(0.0, min(1.0, t))
    closest_x = x1 + t*dx
    closest_y = y1 + t*dy
    dist_sq = (cx - closest_x)**2 + (cy - closest_y)**2
    return dist_sq <= radius**2


def project_cuboid(cam, car_x, car_y, car_theta, pos_x, pos_y, angle, width, length, height):
    hw = width / 2.0
    hl = length / 2.0
    local_verts = [
        (-hl, -hw, 0), ( hl, -hw, 0), ( hl,  hw, 0), (-hl,  hw, 0),
        (-hl, -hw, height), ( hl, -hw, height), ( hl,  hw, height), (-hl,  hw, height)
    ]
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    world_verts = []
    for (lx, ly, lz) in local_verts:
        wx = pos_x + lx * cos_a - ly * sin_a
        wy = pos_y + lx * sin_a + ly * cos_a
        wz = lz
        world_verts.append((wx, wy, wz))

    proj = []
    depths = []
    for (wx, wy, wz) in world_verts:
        sx, sy, d = cam.project(wx, wy, car_x, car_y, car_theta, obj_height=wz)
        proj.append((sx, sy))
        depths.append(d if d is not None else 1e9)

    faces_def = [
        (0,1,2,3, (180,0,0)),
        (4,5,6,7, (220,50,50)),
        (0,3,7,4, (160,0,0)),
        (1,2,6,5, (160,0,0)),
        (0,1,5,4, (200,20,20)),
        (2,3,7,6, (200,20,20))
    ]
    face_list = []
    for i1, i2, i3, i4, cor in faces_def:
        pts = [proj[i] for i in (i1, i2, i3, i4)]
        if any(p[0] is None for p in pts):
            continue
        d_avg = (depths[i1] + depths[i2] + depths[i3] + depths[i4]) / 4.0
        face_list.append((pts, cor, d_avg))
    face_list.sort(key=lambda f: f[2], reverse=True)
    return face_list

def interactive_simulation(r, b, dt, x0, y0, theta0):
    pygame.init()
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Robô – Labirinto 3D com minimapa")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)
    small_font = pygame.font.SysFont("Arial", 12)

    CAR_WIDTH = 2.0 * b
    CAR_LENGTH = 2.5 * b
    CAR_HEIGHT = 0.1

    x, y, theta = float(x0), float(y0), float(theta0)
    trail = [(x, y)]
    prev_x = x0

    base_speed = 0.4
    omega_rotate = 1.5

    command = 'stop'
    rot_start_theta = 0.0
    rot_target_delta = 0.0
    rot_accumulated = 0.0
    rot_omega_sign = 0

    desired_cam_dist = 0.8
    cam3d = Camera3D(WIDTH, HEIGHT, focal_length=250, cam_height=0.8, horizon_y=int(HEIGHT*0.45))

    SKY_TOP = (100, 150, 255)
    SKY_BOTTOM = (180, 210, 255)
    GROUND_COLOR = (34, 130, 34)
    WALL_COLOR = (180, 180, 180)
    FINISH_COLOR = (255, 255, 0)

    # Minimapa
    MAP_WIDTH, MAP_HEIGHT = 200, 150
    MAP_X, MAP_Y = WIDTH - MAP_WIDTH - 10, 10

    running = True
    finish_crossed = False
    collision = False

    while running:
        clock.tick(int(1 / dt))
        dt_actual = dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    command = 'stop'
                elif event.key == pygame.K_UP:
                    command = 'straight'
                elif event.key == pygame.K_LEFT:
                    command = 'rot_left'
                elif event.key == pygame.K_RIGHT:
                    command = 'rot_right'
                elif event.key == pygame.K_DOWN:
                    command = 'rot_180'

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

        # Movimento
        if command == 'stop':
            v = 0.0; omega = 0.0; delta_theta = 0.0
        elif command == 'straight':
            v = base_speed; omega = 0.0; delta_theta = 0.0
            theta = normalize_angle(theta)
        elif command in ('rot_left', 'rot_right', 'rot_180'):
            v = 0.0
            delta_theta = rot_omega_sign * omega_rotate * dt_actual
            rot_accumulated += delta_theta
            if abs(rot_accumulated) >= abs(rot_target_delta):
                target_theta = normalize_angle(rot_start_theta + rot_target_delta)
                delta_theta = target_theta - theta
                theta = target_theta
                command = 'straight'
                delta_theta = 0.0
                rot_accumulated = 0.0
            else:
                theta += delta_theta
            omega = delta_theta / dt_actual
        else:
            v = 0.0; omega = 0.0; delta_theta = 0.0

        # Odometria linear
        if command in ('rot_left', 'rot_right', 'rot_180'):
            delta_x = 0.0; delta_y = 0.0
        else:
            if v == 0.0:
                delta_x = 0.0; delta_y = 0.0
            else:
                delta_x = dt_actual * v * math.cos(theta)
                delta_y = dt_actual * v * math.sin(theta)

        new_x = x + delta_x
        new_y = y + delta_y

        # Colisão com paredes
        hit_wall = False
        for (wx1, wy1, wx2, wy2) in WALLS:
            if circle_segment_collision(new_x, new_y, ROBOT_RADIUS, wx1, wy1, wx2, wy2):
                hit_wall = True
                break
        if hit_wall:
            collision = True
            running = False
        else:
            x, y = new_x, new_y

        if command not in ('rot_left', 'rot_right', 'rot_180'):
            theta = normalize_angle(theta)

        # Linha de chegada
        if not finish_crossed and prev_x < FINISH_X and x >= FINISH_X:
            if FINISH_Y_START <= y <= FINISH_Y_END:
                finish_crossed = True
                running = False
        prev_x = x

        trail.append((x, y))

        # Ajuste de câmera
        cam_x, cam_y = adjust_camera(x, y, theta, desired_cam_dist, WALLS)
        cam3d.set_position(cam_x, cam_y)

        
        screen.fill((0,0,0))
        for i in range(cam3d.horizon_y):
            t = i / cam3d.horizon_y
            r = int(SKY_TOP[0] + (SKY_BOTTOM[0] - SKY_TOP[0]) * t)
            g = int(SKY_TOP[1] + (SKY_BOTTOM[1] - SKY_TOP[1]) * t)
            b = int(SKY_TOP[2] + (SKY_BOTTOM[2] - SKY_TOP[2]) * t)
            pygame.draw.line(screen, (r, g, b), (0, i), (WIDTH, i))
        pygame.draw.rect(screen, GROUND_COLOR, (0, cam3d.horizon_y, WIDTH, HEIGHT - cam3d.horizon_y))

        objects = []

        # Paredes
        for (wx1, wy1, wx2, wy2) in WALLS:
            length = math.hypot(wx2 - wx1, wy2 - wy1)
            if length < 0.01:
                continue
            n_samples = max(2, int(length * 10))
            prev_sx_base = prev_sy_base = prev_sx_top = prev_sy_top = None
            prev_d_base = prev_d_top = 0.0
            for i in range(n_samples + 1):
                t = i / n_samples
                px = wx1 + t * (wx2 - wx1)
                py = wy1 + t * (wy2 - wy1)
                sx_base, sy_base, d_base = cam3d.project(px, py, x, y, theta, obj_height=0.0)
                sx_top, sy_top, d_top = cam3d.project(px, py, x, y, theta, obj_height=WALL_HEIGHT)
                if sx_base is not None and sx_top is not None and d_base and d_top:
                    if prev_sx_base is not None:
                        pts = [(prev_sx_base, prev_sy_base), (sx_base, sy_base),
                               (sx_top, sy_top), (prev_sx_top, prev_sy_top)]
                        avg_depth = (prev_d_base + d_base + d_top + prev_d_top) / 4.0
                        objects.append((avg_depth, 'wall', pts))
                    prev_sx_base, prev_sy_base = sx_base, sy_base
                    prev_sx_top, prev_sy_top = sx_top, sy_top
                    prev_d_base, prev_d_top = d_base, d_top
                else:
                    prev_sx_base = prev_sy_base = prev_sx_top = prev_sy_top = None

        # Linha de chegada
        step = 0.1
        wy = FINISH_Y_START
        prev_sx = prev_sy = None
        prev_depth = 0.0
        while wy <= FINISH_Y_END:
            sx, sy, depth = cam3d.project(FINISH_X, wy, x, y, theta, 0.0)
            if sx is not None and sy is not None and depth:
                if prev_sx is not None:
                    avg_depth = (prev_depth + depth) / 2.0
                    objects.append((avg_depth, 'finish', [(prev_sx, prev_sy), (sx, sy)]))
                prev_sx, prev_sy = sx, sy
                prev_depth = depth
            else:
                prev_sx = prev_sy = None
            wy += step

        objects.sort(key=lambda obj: obj[0], reverse=True)

        for depth, typ, data in objects:
            if typ == 'wall':
                if all(0 <= p[0] <= WIDTH for p in data):
                    pygame.draw.polygon(screen, WALL_COLOR, data)
                    pygame.draw.polygon(screen, (0,0,0), data, 1)
            elif typ == 'finish':
                p1, p2 = data
                if 0 <= p1[0] <= WIDTH and 0 <= p2[0] <= WIDTH:
                    pygame.draw.line(screen, FINISH_COLOR, p1, p2, 5)

        # Carro
        car_faces = project_cuboid(cam3d, x, y, theta, x, y, theta, CAR_WIDTH, CAR_LENGTH, CAR_HEIGHT)
        for pts, cor, _ in car_faces:
            if len(pts) >= 3 and all(0 <= p[0] <= WIDTH for p in pts):
                pygame.draw.polygon(screen, cor, pts)
                pygame.draw.polygon(screen, (0,0,0), pts, 1)

      
        minimap_surf = pygame.Surface((MAP_WIDTH, MAP_HEIGHT))
        minimap_surf.fill((0, 0, 0))  # fundo preto opaco
        scale_x = MAP_WIDTH / (WORLD_X_MAX - WORLD_X_MIN)
        scale_y = MAP_HEIGHT / (WORLD_Y_MAX - WORLD_Y_MIN)
        def world_to_map(wx, wy):
            mx = int((wx - WORLD_X_MIN) * scale_x)
            my = int((WORLD_Y_MAX - wy) * scale_y)
            return mx, my

        # Paredes (brancas, grossas)
        for (wx1, wy1, wx2, wy2) in WALLS:
            p1 = world_to_map(wx1, wy1)
            p2 = world_to_map(wx2, wy2)
            pygame.draw.line(minimap_surf, (200, 200, 200), p1, p2, 2)

      
        p_start = world_to_map(FINISH_X, FINISH_Y_START)
        p_end = world_to_map(FINISH_X, FINISH_Y_END)
        pygame.draw.line(minimap_surf, (255, 255, 0), p_start, p_end, 2)

      
        if len(trail) > 1:
            map_trail = [world_to_map(px, py) for px, py in trail]
            pygame.draw.lines(minimap_surf, (0, 200, 0), False, map_trail, 1)

       
        cx, cy = world_to_map(x, y)
        angle = -theta
        size = 5
        p_nose = (cx + size * math.cos(angle), cy + size * math.sin(angle))
        p_left = (cx - size * 0.5 * math.cos(angle + 2.5), cy - size * 0.5 * math.sin(angle + 2.5))
        p_right = (cx - size * 0.5 * math.cos(angle - 2.5), cy - size * 0.5 * math.sin(angle - 2.5))
        pygame.draw.polygon(minimap_surf, (255, 0, 0), [p_nose, p_left, p_right])
        pygame.draw.polygon(minimap_surf, (0, 0, 0), [p_nose, p_left, p_right], 1)

        screen.blit(minimap_surf, (MAP_X, MAP_Y))

        
        if command == 'stop':
            status = "Parado"
        elif command == 'straight':
            status = "Seguindo"
        elif command == 'rot_left':
            status = f"Virando esq. {math.degrees(abs(rot_accumulated)):.0f}°"
        elif command == 'rot_right':
            status = f"Virando dir. {math.degrees(abs(rot_accumulated)):.0f}°"
        elif command == 'rot_180':
            status = f"Giro 180° {math.degrees(abs(rot_accumulated)):.0f}°"
        else:
            status = "---"
        info = [
            f"Vel: {v*3.6:.0f} km/h",
            f"θ: {math.degrees(theta):.0f}°",
            status,
            "ESC: sair"
        ]
        for i, line in enumerate(info):
            text = small_font.render(line, True, (255,255,255))
            screen.blit(text, (10, 10 + i*18))

        if finish_crossed:
            msg = font.render("VITORIA!", True, (255,255,0))
            screen.blit(msg, (WIDTH//2-60, HEIGHT//2-20))
        if collision:
            msg = font.render("COLISÃO!", True, (255,0,0))
            screen.blit(msg, (WIDTH//2-60, HEIGHT//2-20))

        pygame.display.flip()

        if finish_crossed or collision:
            pygame.time.wait(2000)
            running = False

    pygame.quit()
    return trail, finish_crossed, collision


def main():
    print("Simulador interativo – Labirinto 3D com minimapa")
    print("Controles: setas, espaço, ESC")
    r      = float(input("Raio da roda r (m) [0.034]: ") or 0.034)
    two_b  = float(input("Distância entre rodas 2b (m) [0.094]: ") or 0.094)
    b = two_b / 2.0
    T_ms   = float(input("Período de integração T (ms) [20]: ") or 20)
    dt = T_ms / 1000.0
    x0     = float(input("x inicial (m) [-0.5]: ") or -0.5)
    y0     = float(input("y inicial (m) [2.5]: ") or 2.5)
    theta0 = float(input("θ inicial (rad) [0.0]: ") or 0.0)

    print(f"\nLabirinto com paredes. Chegada em x = {FINISH_X} m, y de {FINISH_Y_START} a {FINISH_Y_END}.")
    print("Não encoste nas paredes!\n")
    input("Pressione Enter para iniciar...")

    trail, finished, crashed = interactive_simulation(r, b, dt, x0, y0, theta0)

    if finished:
        print("\n🏁 Parabéns! Cruzou a linha de chegada sem bater!")
    elif crashed:
        print("\n💥 Você colidiu com uma parede!")
    else:
        print("\nSimulação interrompida.")

    n = len(trail)
    step = max(1, n // 20)
    print("\nCaminho percorrido (amostras):")
    for i in range(0, n, step):
        px, py = trail[i]
        print(f"  ponto {i}: ({px:.4f}, {py:.4f})")
    if n > 0:
        print(f"  ponto {n-1}: ({trail[-1][0]:.4f}, {trail[-1][1]:.4f})")
    print(f"Total de pontos: {n}")

    if trail:
        traj_x, traj_y = zip(*trail)
        plt.figure(figsize=(8,6))
        plt.plot(traj_x, traj_y, '.-', markersize=2, linewidth=1, label='Trajetória')
        for (wx1, wy1, wx2, wy2) in WALLS:
            plt.plot([wx1, wx2], [wy1, wy2], 'k-', linewidth=2)
        plt.plot([FINISH_X, FINISH_X], [FINISH_Y_START, FINISH_Y_END], 'y-', linewidth=3, label='Chegada')
        plt.plot(traj_x[0], traj_y[0], 'go', label='Início')
        plt.plot(traj_x[-1], traj_y[-1], 'ro', label='Fim')
        plt.axis('equal')
        plt.xlim(WORLD_X_MIN, WORLD_X_MAX)
        plt.ylim(WORLD_Y_MIN, WORLD_Y_MAX)
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title("Trajetória no labirinto")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()