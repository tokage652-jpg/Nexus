import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge, Circle
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# =========================================================
# 공통: 교실 8개의 학생수 (세 도형 모두 동일하게 유지)
# =========================================================
STUDENT_COUNTS = [22, 24, 19, 24, 25, 26, 27, 26]
RES = 220


# =========================================================
# 공용 함수
# =========================================================
def in_rect(px, py, r):
    return (px >= r['xmin']) & (px <= r['xmax']) & (py >= r['ymin']) & (py <= r['ymax'])

def points_in_polygon(px, py, poly):
    n = len(poly)
    inside = np.zeros(len(px), dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > py) != (yj > py)) & \
               (px < (xj - xi) * (py - yi) / (yj - yi + 1e-9) + xi)
        inside ^= cond
        j = i
    return inside

def build_and_solve(xmin, xmax, ymin, ymax, building_fn, blocked_fn,
                     toilet_male, toilet_female, door_positions, counts):
    xs = np.linspace(xmin, xmax, RES)
    ys = np.linspace(ymin, ymax, RES)
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    gx, gy = np.meshgrid(xs, ys, indexing='ij')
    gx_flat, gy_flat = gx.ravel(), gy.ravel()

    building = building_fn(gx_flat, gy_flat)
    blocked = blocked_fn(gx_flat, gy_flat)
    walkable = building & ~blocked

    idx_grid = -np.ones(RES * RES, dtype=int)
    walk_idx = np.where(walkable)[0]
    idx_grid[walk_idx] = np.arange(len(walk_idx))
    N = len(walk_idx)

    ii, jj = np.meshgrid(np.arange(RES), np.arange(RES), indexing='ij')
    ii, jj = ii.ravel(), jj.ravel()
    flat_idx = ii * RES + jj
    diag = (dx ** 2 + dy ** 2) ** 0.5
    neighbors = [(1, 0, dx), (-1, 0, dx), (0, 1, dy), (0, -1, dy),
                 (1, 1, diag), (1, -1, diag), (-1, 1, diag), (-1, -1, diag)]
    graph = lil_matrix((N, N))
    for di, dj, w in neighbors:
        ni, nj = ii + di, jj + dj
        valid = (ni >= 0) & (ni < RES) & (nj >= 0) & (nj < RES)
        a = flat_idx[valid]
        b = ni[valid] * RES + nj[valid]
        both = walkable[a] & walkable[b]
        graph[idx_grid[a[both]], idx_grid[b[both]]] = w
    graph = graph.tocsr()

    def nearest_idx(pt):
        d = (gx_flat - pt[0]) ** 2 + (gy_flat - pt[1]) ** 2
        d = np.where(walkable, d, np.inf)
        return idx_grid[np.argmin(d)]

    src_m, src_f = nearest_idx(toilet_male), nearest_idx(toilet_female)
    dist_m = dijkstra(graph, directed=False, indices=[src_m])[0]
    dist_f = dijkstra(graph, directed=False, indices=[src_f])[0]

    dm = np.array([dist_m[nearest_idx(d)] for d in door_positions])
    df = np.array([dist_f[nearest_idx(d)] for d in door_positions])
    counts = np.array(counts)

    D_avg_m, D_max_m = np.sum(counts * dm) / counts.sum(), dm.max()
    D_avg_f, D_max_f = np.sum(counts * df) / counts.sum(), df.max()
    E_m = 1 / ((D_avg_m*0.0001 + D_max_m*0.0001) * (D_max_m / D_avg_m))
    E_f = 1 / (((D_avg_f)*0.0001 + D_max_f*0.0001) * (D_max_f / D_avg_f))

    return dict(D_avg_m=D_avg_m, D_max_m=D_max_m, E_m=E_m,
                D_avg_f=D_avg_f, D_max_f=D_max_f, E_f=E_f,
                walkable=walkable, gx_flat=gx_flat, gy_flat=gy_flat, n_walk=walkable.sum())


# =========================================================
# 구조 1: 막대형(I자형) - 기존에 쓰던 학교 평면도
# =========================================================
def solve_bar_shape():
    boundary = np.array([
        [210, 1010], [210, 70], [430, 70], [430, 390], [690, 390], [690, 540],
        [1260, 540], [1260, 70], [1480, 70], [1480, 660], [690, 660], [690, 780],
        [430, 780], [430, 1010],
    ])
    rooms = [
        {'xmin': 210, 'xmax': 300, 'ymin': 940, 'ymax': 1010},
        {'xmin': 210, 'xmax': 300, 'ymin': 860, 'ymax': 940},
        {'xmin': 210, 'xmax': 300, 'ymin': 780, 'ymax': 860},
        {'xmin': 210, 'xmax': 300, 'ymin': 700, 'ymax': 780},
        {'xmin': 210, 'xmax': 300, 'ymin': 310, 'ymax': 390},
        {'xmin': 210, 'xmax': 300, 'ymin': 230, 'ymax': 310},
        {'xmin': 210, 'xmax': 300, 'ymin': 150, 'ymax': 230},
        {'xmin': 210, 'xmax': 300, 'ymin': 70, 'ymax': 150},
    ]
    toilet_male, toilet_female = np.array([500, 500]), np.array([620, 500])
    doors = [np.array([r['xmax'], (r['ymin'] + r['ymax']) / 2]) for r in rooms]

    def building_fn(px, py):
        return points_in_polygon(px, py, boundary)

    def blocked_fn(px, py):
        b = np.zeros(len(px), dtype=bool)
        for r in rooms:
            b |= in_rect(px, py, r)
        return b

    xmin, ymin = boundary.min(axis=0)
    xmax, ymax = boundary.max(axis=0)
    res = build_and_solve(xmin, xmax, ymin, ymax, building_fn, blocked_fn,
                           toilet_male, toilet_female, doors, STUDENT_COUNTS)
    res.update(shape="bar", boundary=boundary, rooms=rooms,
               toilet_male=toilet_male, toilet_female=toilet_female)
    return res


# =========================================================
# 구조 2: ㅁ자형(십자 복도형) - 중정을 가로지르는 십자 복도 적용
# =========================================================
def solve_ring_shape():
    OUTER = {'xmin': 0, 'xmax': 1000, 'ymin': 0, 'ymax': 1000}
    HOLE = {'xmin': 300, 'xmax': 700, 'ymin': 300, 'ymax': 700}
    BRIDGE_HALF_WIDTH = 20  # 복도 반쪽 너비 (총 너비 40)
    
    rooms = [
        {'xmin': 200, 'xmax': 400, 'ymin': 880, 'ymax': 1000},
        {'xmin': 600, 'xmax': 800, 'ymin': 880, 'ymax': 1000},
        {'xmin': 200, 'xmax': 400, 'ymin': 0, 'ymax': 120},
        {'xmin': 600, 'xmax': 800, 'ymin': 0, 'ymax': 120},
        {'xmin': 0, 'xmax': 120, 'ymin': 600, 'ymax': 800},
        {'xmin': 0, 'xmax': 120, 'ymin': 200, 'ymax': 400},
        {'xmin': 880, 'xmax': 1000, 'ymin': 600, 'ymax': 800},
        {'xmin': 880, 'xmax': 1000, 'ymin': 200, 'ymax': 400},
    ]
    toilet_male, toilet_female = np.array([480, 500]), np.array([520, 500])

    def door_of(c):
        cx, cy = (c['xmin'] + c['xmax']) / 2, (c['ymin'] + c['ymax']) / 2
        if c['ymax'] >= 990: return np.array([cx, c['ymin']])
        if c['ymin'] <= 10: return np.array([cx, c['ymax']])
        if c['xmin'] <= 10: return np.array([c['xmax'], cy])
        return np.array([c['xmin'], cy])

    doors = [door_of(r) for r in rooms]
    outer_poly = np.array([[OUTER['xmin'], OUTER['ymin']], [OUTER['xmax'], OUTER['ymin']],
                            [OUTER['xmax'], OUTER['ymax']], [OUTER['xmin'], OUTER['ymax']]])

    def building_fn(px, py):
        # 1. 기본 ㅁ자 외곽에서 중정 파내기
        base_ring = points_in_polygon(px, py, outer_poly) & ~in_rect(px, py, HOLE)
        
        # 2. 십자(✚) 모양 통로 추가 (가로 + 세로)
        in_horizontal = (np.abs(py - 500) <= BRIDGE_HALF_WIDTH) & (px >= OUTER['xmin']) & (px <= OUTER['xmax'])
        in_vertical = (np.abs(px - 500) <= BRIDGE_HALF_WIDTH) & (py >= OUTER['ymin']) & (py <= OUTER['ymax'])
        in_bridge = in_horizontal | in_vertical
        
        return base_ring | in_bridge

    def blocked_fn(px, py):
        b = np.zeros(len(px), dtype=bool)
        for r in rooms:
            b |= in_rect(px, py, r)
        return b

    res = build_and_solve(OUTER['xmin'], OUTER['xmax'], OUTER['ymin'], OUTER['ymax'],
                           building_fn, blocked_fn, toilet_male, toilet_female, doors, STUDENT_COUNTS)
    res.update(shape="ring", outer=OUTER, hole=HOLE, rooms=rooms,
               toilet_male=toilet_male, toilet_female=toilet_female)
    return res


# =========================================================
# 구조 3: ○자형(원형) - 십자 복도가 포함된 원형
# =========================================================
def solve_circle_shape():
    R_OUTER, R_HOLE = 550, 250
    R_CLASS_IN, R_CLASS_OUT = 430, 550
    BRIDGE_HALF_WIDTH = 40
    N_ROOMS = 8
    SECTOR = 360 / N_ROOMS
    sectors = [(k * SECTOR, (k + 1) * SECTOR) for k in range(N_ROOMS)]
    toilet_male, toilet_female = np.array([-20, 0]), np.array([20, 0])

    def polar(px, py):
        r = np.sqrt(px ** 2 + py ** 2)
        theta = (np.degrees(np.arctan2(py, px)) + 360) % 360
        return r, theta

    def door_of(sec):
        mid = np.radians((sec[0] + sec[1]) / 2)
        return np.array([R_CLASS_IN * np.cos(mid), R_CLASS_IN * np.sin(mid)])

    doors = [door_of(s) for s in sectors]

    def building_fn(px, py):
        r, theta = polar(px, py)
        in_horizontal = (np.abs(py) <= BRIDGE_HALF_WIDTH) & (np.abs(px) <= R_CLASS_IN)
        in_vertical = (np.abs(px) <= BRIDGE_HALF_WIDTH) & (np.abs(py) <= R_CLASS_IN)
        in_bridge = in_horizontal | in_vertical
        return ((r <= R_OUTER) & ~(r <= R_HOLE)) | in_bridge

    def blocked_fn(px, py):
        r, theta = polar(px, py)
        b = np.zeros(len(px), dtype=bool)
        for tmin, tmax in sectors:
            b |= (r >= R_CLASS_IN) & (r <= R_CLASS_OUT) & (theta >= tmin) & (theta < tmax)
        return b

    res = build_and_solve(-R_OUTER, R_OUTER, -R_OUTER, R_OUTER, building_fn, blocked_fn,
                           toilet_male, toilet_female, doors, STUDENT_COUNTS)
    res.update(shape="circle", R_OUTER=R_OUTER, R_HOLE=R_HOLE,
               R_CLASS_IN=R_CLASS_IN, R_CLASS_OUT=R_CLASS_OUT, sectors=sectors,
               toilet_male=toilet_male, toilet_female=toilet_female)
    return res


def solve_real_redesign():
    boundary = np.array([
        [210, 1010], [210, 70], [430, 70], [430, 390], [690, 390], [690, 540],
        [1260, 540], [1260, 70], [1480, 70], [1480, 660], [690, 660], [690, 780],
        [430, 780], [430, 1010],
    ])
    rooms = [
        {'xmin': 210, 'xmax': 300, 'ymin': 930, 'ymax': 1010, 'count': 22},
        {'xmin': 210, 'xmax': 300, 'ymin': 760, 'ymax': 840, 'count': 24},
        {'xmin': 210, 'xmax': 300, 'ymin': 590, 'ymax': 670, 'count': 19},
        {'xmin': 210, 'xmax': 300, 'ymin': 420, 'ymax': 500, 'count': 24},
        {'xmin': 1390, 'xmax': 1480, 'ymin': 580, 'ymax': 660, 'count': 25},
        {'xmin': 1390, 'xmax': 1480, 'ymin': 410, 'ymax': 490, 'count': 26},
        {'xmin': 1390, 'xmax': 1480, 'ymin': 240, 'ymax': 320, 'count': 27},
        {'xmin': 1390, 'xmax': 1480, 'ymin': 70, 'ymax': 150, 'count': 26},
    ]
    doors = ([np.array([r['xmax'], (r['ymin'] + r['ymax']) / 2]) for r in rooms[:4]] +
              [np.array([r['xmin'], (r['ymin'] + r['ymax']) / 2]) for r in rooms[4:]])
    # 양쪽 교실 클러스터의 중간 지점에 화장실을 대칭으로 배치 (ㅁ/○자형과 같은 원리)
    left_mean = np.mean(doors[:4], axis=0)
    right_mean = np.mean(doors[4:], axis=0)
    center = (left_mean + right_mean) / 2
    toilet_male, toilet_female = center + np.array([-20, 0]), center + np.array([20, 0])
 
    def building_fn(px, py):
        return points_in_polygon(px, py, boundary)
 
    def blocked_fn(px, py):
        b = np.zeros(len(px), dtype=bool)
        for r in rooms:
            b |= in_rect(px, py, r)
        return b
 
    xmin, ymin = boundary.min(axis=0)
    xmax, ymax = boundary.max(axis=0)
    res = build_and_solve(xmin, xmax, ymin, ymax, building_fn, blocked_fn,
                           toilet_male, toilet_female, doors, [r['count'] for r in rooms])
    res.update(shape="bar", boundary=boundary, rooms=rooms,
               toilet_male=toilet_male, toilet_female=toilet_female)
    return res
 

# =========================================================
# 실행 및 비교
# =========================================================
results = {
    "막대형(I자)": solve_bar_shape(),
    "ㅁ자형(십자)": solve_ring_shape(),
    "○자형(원형)": solve_circle_shape(),
    "실제 구조 + 배치 변화": solve_real_redesign()
}

print("=" * 70)
print(f"{'구조':16}{'걷는칸수':>10}{'남D_avg':>10}{'남D_max':>10}{'남E':>10}{'여E':>10}")
print("-" * 70)
for label, r in results.items():
    print(f"{label:16}{r['n_walk']:>10,}{r['D_avg_m']:>10.1f}{r['D_max_m']:>10.1f}{r['E_m']:>10.5f}{r['E_f']:>10.5f}")
print("=" * 70)


# =========================================================
# 시각화
# =========================================================
fig, axes = plt.subplots(1, 4, figsize=(21, 7))

for ax, (label, r) in zip(axes, results.items()):
    ax.scatter(r['gx_flat'][r['walkable']], r['gy_flat'][r['walkable']],
               c='#f2f2f2', s=2, zorder=1)

    if r['shape'] == 'bar':
        for rm in r['rooms']:
            ax.add_patch(Rectangle((rm['xmin'], rm['ymin']), rm['xmax']-rm['xmin'], rm['ymax']-rm['ymin'],
                                   facecolor='white', edgecolor='black', zorder=2))
        bc = np.vstack([r['boundary'], r['boundary'][0]])
        ax.plot(bc[:, 0], bc[:, 1], 'k-', linewidth=1.5, zorder=3)

    elif r['shape'] == 'ring':
        for rm in r['rooms']:
            ax.add_patch(Rectangle((rm['xmin'], rm['ymin']), rm['xmax']-rm['xmin'], rm['ymax']-rm['ymin'],
                                   facecolor='white', edgecolor='black', zorder=2))
        o = r['outer']
        ax.add_patch(Rectangle((o['xmin'], o['ymin']), o['xmax']-o['xmin'], o['ymax']-o['ymin'],
                                fill=False, edgecolor='black', linewidth=1.5, zorder=3))
        h = r['hole']
        ax.add_patch(Rectangle((h['xmin'], h['ymin']), h['xmax']-h['xmin'], h['ymax']-h['ymin'],
                                fill=False, edgecolor='black', linewidth=1.2, zorder=3))

    elif r['shape'] == 'circle':
        for tmin, tmax in r['sectors']:
            ax.add_patch(Wedge((0, 0), r['R_CLASS_OUT'], tmin, tmax, width=r['R_CLASS_OUT']-r['R_CLASS_IN'],
                                   facecolor='white', edgecolor='black', zorder=2))
        ax.add_patch(Circle((0, 0), r['R_OUTER'], fill=False, edgecolor='black', linewidth=1.5, zorder=3))
        ax.add_patch(Circle((0, 0), r['R_HOLE'], fill=False, edgecolor='black', linewidth=1.2, zorder=3))

    ax.plot(*r['toilet_male'], '^', color='tab:blue', markersize=13, markeredgecolor='white', zorder=5)
    ax.plot(*r['toilet_female'], '^', color='tab:pink', markersize=13, markeredgecolor='white', zorder=5)
    ax.set_aspect('equal')
    ax.set_title(f"{label}\nE_남={r['E_m']:.5f}, E_여={r['E_f']:.5f}")

plt.tight_layout()
plt.show()