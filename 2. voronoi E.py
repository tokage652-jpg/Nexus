import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree

# =========================================================
# 1. 시설(생성점) 좌표
# =========================================================
# points = np.array([
#     [0.2, 0.3],
#     [0.8, 0.2],
#     [0.5, 0.7],
#     [0.3, 0.8],
#     [0.7, 0.2],
#     [0.4, 0.6],
#     [0.4, 0.9],
#     [0.8, 0.1],
#     [0.6, 0.3],
#     [0.9, 0.5]
# ])

# points = np.array([
#     [0.1667, 0.1667], [0.5, 0.1667], [0.8333, 0.1667],
#     [0.1667, 0.5],    [0.5, 0.5],    [0.8333, 0.5],
#     [0.1667, 0.8333], [0.5, 0.8333], [0.8333, 0.8333],
# ])

# points = np.array([[0.842731, 0.154982],
# [0.697415, 0.031864],
# [0.913527, 0.468209],
# [0.275691, 0.786134],
# [0.558047, 0.102395],
# [0.994816, 0.621583],
# [0.389470, 0.047218],
# [0.735902, 0.186754],
# [0.850163, 0.512841]])

# points = np.array([[0.174395, 0.638201],
# [0.951827, 0.082564],
# [0.497316, 0.721945],
# [0.340158, 0.269473],
# [0.886042, 0.515790],
# [0.063927, 0.973684],
# [0.782451, 0.194306],
# [0.429865, 0.857213],
# [0.557094, 0.311682]])

points = np.array([
[0.2, 0.1],
[0.1, 0.2],
[0.2, 0.3],
[0.1, 0.4],
[0.2, 0.5],
[0.1, 0.6],
[0.2, 0.7],
[0.1, 0.8],
[0.2, 0.9]])

# points = np.array([[0.421578, 0.907314],
# [0.118693, 0.564821],
# [0.752946, 0.239187],
# [0.681405, 0.993572],
# [0.046251, 0.817936],
# [0.534128, 0.370649],
# [0.289714, 0.125483],
# [0.968205, 0.456792],
# [0.603871, 0.742156]])

# points = np.array([
#     [0.5, 0.9],
#     [0.4, 0.1],
#     [0.6, 0.4],
#     [0.2, 0.5],
#     [0.8, 0.6],
#     [0.9, 0.9],
#     [0.1, 0.2],
#     [0.2, 0.4],
#     [0.4, 0.2],
#     [0.3, 0.5]
# ])

# =========================================================
# 2. 보로노이 다이어그램 생성 및 시각화
# =========================================================
vor = Voronoi(points)

fig = voronoi_plot_2d(vor, show_vertices=False)
fig.set_size_inches(5, 5)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect('equal')
plt.plot(points[:, 0], points[:, 1], 'ro')
plt.title("Voronoi Diagram")


# =========================================================
# 3. 경계 밖으로 뻗어나가는(무한) 보로노이 셀을 유한 폴리곤으로 재구성
#    (외곽 시설의 셀은 원래 경계가 없기 때문에, 분석 영역인
#     [0,1] x [0,1] 사각형 기준으로 잘라낼 수 있도록 먼저
#     멀리 있는 가상의 꼭짓점을 만들어 닫힌 폴리곤으로 만들어줌)
#    참고: scipy 공식 예제로 잘 알려진 방식 (voronoi_finite_polygons_2d)
# =========================================================
def voronoi_finite_polygons_2d(vor, radius=None):
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


# ---------------------------------------------------------
# Sutherland-Hodgman 알고리즘: 볼록 폴리곤을 [0,1] x [0,1]
# 사각형(분석 영역) 안쪽으로 잘라낸다.
# ---------------------------------------------------------
def clip_polygon_to_unit_square(poly):
    def is_inside(p, a, b):
        return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) >= 0

    def intersect(p1, p2, a, b):
        x1, y1 = p1; x2, y2 = p2; x3, y3 = a; x4, y4 = b
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-12:
            return p2
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        return (x1 + t*(x2-x1), y1 + t*(y2-y1))

    def clip_edge(poly, a, b):
        result = []
        n = len(poly)
        for i in range(n):
            cur, prev = poly[i], poly[i-1]
            cur_in, prev_in = is_inside(cur, a, b), is_inside(prev, a, b)
            if cur_in:
                if not prev_in:
                    result.append(intersect(prev, cur, a, b))
                result.append(cur)
            elif prev_in:
                result.append(intersect(prev, cur, a, b))
        return result

    square = [(0, 0), (1, 0), (1, 1), (0, 1)]  # 반시계 방향
    result = poly
    for i in range(len(square)):
        if not result:
            break
        result = clip_edge(result, square[i-1], square[i])
    return result


def polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    x = np.array([p[0] for p in poly])
    y = np.array([p[1] for p in poly])
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


regions, vertices = voronoi_finite_polygons_2d(vor)

cell_areas = []
for region in regions:
    poly = [tuple(vertices[i]) for i in region]
    clipped = clip_polygon_to_unit_square(poly)
    cell_areas.append(polygon_area(clipped))
cell_areas = np.array(cell_areas)  # points와 같은 순서 (시설별 보로노이 폼 면적)


# =========================================================
# 4. 이동거리 계산
#    - 공간을 일정 간격의 격자점으로 분할
#    - 각 격자점에서 가장 가까운 시설까지의 거리 = 이동거리
#    - cKDTree를 사용해 모든 격자점의 최근접 거리를 빠르게 계산
# =========================================================
GRID_RESOLUTION = 200  # 격자 한 변의 점 개수 (높일수록 정밀해지지만 계산이 오래 걸림)

xs = np.linspace(0, 1, GRID_RESOLUTION)
ys = np.linspace(0, 1, GRID_RESOLUTION)
gx, gy = np.meshgrid(xs, ys)
grid_points = np.column_stack([gx.ravel(), gy.ravel()])

tree = cKDTree(points)
distances, _ = tree.query(grid_points)

D_avg = np.mean(distances)   # 평균 이동거리
D_max = np.max(distances)    # 최대 이동거리


# =========================================================
# 5. 공간 효율성 지표 계산
#
#              1
#   E = ───────────────────────────────
#        (D_avg + D_max)(1+CV_A)(D_max/D_avg)
#
#   CV_A = σ_A / A_bar
#   (A_bar: 보로노이 폼 면적들의 평균, σ_A: 보로노이 폼 면적들의 표준편차)
# =========================================================
A_bar = np.mean(cell_areas)
sigma_A = np.std(cell_areas)
CV_A = sigma_A / A_bar

E = 1 / ((D_avg + D_max) * (1 + CV_A) * (D_max / D_avg))

print("=" * 45)
print(f"격자점 개수         : {len(distances):,}개")
print(f"평균 이동거리 (D_avg) : {D_avg:.5f}")
print(f"최대 이동거리 (D_max) : {D_max:.5f}")
print("-" * 45)
print(f"보로노이 폼 면적들     : {np.round(cell_areas, 5)}")
print(f"면적 합 (검산, ≈1이어야 함) : {cell_areas.sum():.5f}")
print(f"면적 평균 (A_bar)     : {A_bar:.5f}")
print(f"면적 표준편차 (σ_A)    : {sigma_A:.5f}")
print(f"변동계수 (CV_A)      : {CV_A:.5f}")
print("-" * 45)
print(f"공간 효율성 지표 (E)  : {E:.6f}")
print("=" * 45)

plt.show()