import json
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from pyproj import Transformer

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# =========================================================
# 0. 파일 경로
# =========================================================
BOUNDARY_GEOJSON_PATH = r"C:\Users\Lenovo\OneDrive - 보평고등학교\바탕 화면\유은\02. 2026\01. 학교\04. 학술제\HangJeongDong_ver20260701.geojson"
FACILITY_CSV_PATH = r"C:\Users\Lenovo\OneDrive - 보평고등학교\바탕 화면\유은\02. 2026\01. 학교\04. 학술제\소방서.csv"
POPULATION_CSV_PATH = r"C:\Users\Lenovo\Downloads\_census_reqdoc_1784693050351\2024년_인구_다사_100M.csv"   # SGIS 100m 격자 인구 자료
CITY_KEYWORD = "성남시"
NAME_KEY_CANDIDATES = ["SIG_KOR_NM", "sggnm", "ADM_NM", "adm_nm", "SGG_NM", "sido_sgg_nm"]
GRID_BOUNDARY_PATH = r"C:\Users\Lenovo\Downloads\_grid_border_grid_2025_grid_다사_grid_다사\grid_다사_100M.shp"

# =========================================================
# 1. 골든타임 기준 설정
# =========================================================
AVG_SPEED_KMH = 25.4
DETOUR_FACTOR = 1.218
GOLDEN_TIME_MIN = 5
RADIUS_KM = (AVG_SPEED_KMH * (GOLDEN_TIME_MIN / 60)) / DETOUR_FACTOR
print(f"[설정] 골든타임 {GOLDEN_TIME_MIN}분 기준 직선거리 반경: {RADIUS_KM:.3f} km")

# E 공식에서 D_max 대신 사용할 인구가중 백분위수 (0.95 = 상위 95%)
D_MAX_PERCENTILE = 0.915

# =========================================================
# 2. 최적화 설정값
# =========================================================
N_RESTARTS_E = 3        # E 최적화(로이드+SA) 재시도 횟수 - 오래 걸리므로 낮게 시작 권장
LLOYD_ITERS = 20
SA_ITERS_E = 1500
N_CANDIDATES = 2000      # 커버리지 최적화(탐욕) 후보지 개수
SA_ITERS_COV = 2000
RANDOM_SEED = 0


# =========================================================
# 3. 성남시 경계 로드
# =========================================================
with open(BOUNDARY_GEOJSON_PATH, encoding="utf-8") as f:
    geojson_obj = json.load(f)

def extract_city_polygons(geojson_obj, name_keys, keyword):
    polygons = []
    for feat in geojson_obj["features"]:
        props = feat["properties"]
        name_val = None
        for k in name_keys:
            if k in props and props[k] is not None:
                name_val = str(props[k]); break
        if name_val is None or keyword not in name_val:
            continue
        geom = feat["geometry"]
        if geom["type"] == "Polygon":
            polygons.append(np.array(geom["coordinates"][0])[:, :2])
        elif geom["type"] == "MultiPolygon":
            for part in geom["coordinates"]:
                polygons.append(np.array(part[0])[:, :2])
    return polygons

boundary_polys_lonlat = extract_city_polygons(geojson_obj, NAME_KEY_CANDIDATES, CITY_KEYWORD)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
def transform_lonlat(arr_lonlat):
    x, y = transformer.transform(arr_lonlat[:, 0], arr_lonlat[:, 1])
    return np.column_stack([x, y])
boundary_polys = [transform_lonlat(p) for p in boundary_polys_lonlat]

def point_in_single_polygon(px, py, poly):
    n = len(poly)
    inside = np.zeros(len(px), dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        cond = ((yi > py) != (yj > py)) & (px < (xj-xi)*(py-yi)/(yj-yi+1e-9)+xi)
        inside ^= cond
        j = i
    return inside

def point_in_city(px, py, polygons):
    inside = np.zeros(len(px), dtype=bool)
    for poly in polygons:
        inside |= point_in_single_polygon(px, py, poly)
    return inside

def random_points_in_boundary(n, boundary_polys, xmin, xmax, ymin, ymax, rng):
    accepted = []
    while len(accepted) < n:
        need = n - len(accepted)
        cx = rng.uniform(xmin, xmax, size=need * 4)
        cy = rng.uniform(ymin, ymax, size=need * 4)
        mask = point_in_city(cx, cy, boundary_polys)
        accepted.extend(np.column_stack([cx[mask], cy[mask]])[:need].tolist())
    return np.array(accepted[:n])

all_pts = np.vstack(boundary_polys)
xmin, ymin = all_pts.min(axis=0)
xmax, ymax = all_pts.max(axis=0)


# =========================================================
# 4. 소방서 좌표 로드 (현재 실제 배치)
#    - 성남소방서 좌표 수동 보정(하대원동 신청사) + 중복 위치 제거 포함
# =========================================================
df_fac = pd.read_csv(FACILITY_CSV_PATH, encoding="cp949")
ADDRESS_COL = "주소"
LAT_COL = "X좌표"
LON_COL = "Y좌표"
df_fac_seongnam = df_fac[df_fac[ADDRESS_COL].astype(str).str.contains(CITY_KEYWORD, na=False)]
facility_lonlat = df_fac_seongnam[[LON_COL, LAT_COL]].to_numpy(dtype=float)
points = transform_lonlat(facility_lonlat)

# [수동 보정] 성남소방서(공공데이터 주소: 수정구 제일로 111)는
# 2022년 5월 중원구 하대원동 2로 신청사 이전. 공공데이터포털 CSV가
# 구주소 기준으로 되어있어 실제 좌표로 직접 교체함.
# 출처: 성남소방서 공식 홈페이지(119.gg.go.kr) "찾아오시는 길"
seongnam_station_mask = df_fac_seongnam["소방서 및 안전센터명"] == "성남소방서"
idx_in_points = np.where(seongnam_station_mask.to_numpy())[0][0]
corrected = transform_lonlat(np.array([[127.1586809, 37.4223413]]))
points[idx_in_points] = corrected[0]

print(f"[확인] 성남시 소재 시설 개수: {len(points)}")

# 좌표가 완전히 같은(중복 위치) 시설 제거 - 먼저 나온 것만 유지
_, unique_idx = np.unique(points.round(3), axis=0, return_index=True)
unique_idx = np.sort(unique_idx)
points = points[unique_idx]
df_fac_seongnam = df_fac_seongnam.iloc[unique_idx].reset_index(drop=True)
print(f"중복 제거 후 시설 개수: {len(points)}")

current_points = points
N_FACILITIES = len(current_points)


# =========================================================
# 5. 인구 격자 로드
# =========================================================
grid_gdf = gpd.read_file(GRID_BOUNDARY_PATH)
if grid_gdf.crs is not None and grid_gdf.crs.to_epsg() != 5179:
    grid_gdf = grid_gdf.to_crs(epsg=5179)

pop_df = pd.read_csv(POPULATION_CSV_PATH, encoding="cp949", header=None,
                      names=["year", "grid_id", "stat_cd", "value"])
pop_total = pop_df[pop_df["stat_cd"] == "to_in_001"][["grid_id", "value"]].copy()
pop_total.columns = ["GRID_CD", "population"]

grid_gdf["GRID_CD"] = grid_gdf["GRID_CD"].astype(str)
pop_total["GRID_CD"] = pop_total["GRID_CD"].astype(str)
merged = grid_gdf.merge(pop_total, on="GRID_CD", how="left")
merged["population"] = merged["population"].fillna(0)

centroids = merged.geometry.centroid
pop_xy_all = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])
pop_values_all = merged["population"].to_numpy(dtype=float)

inside_mask = point_in_city(pop_xy_all[:, 0], pop_xy_all[:, 1], boundary_polys)
pop_xy = pop_xy_all[inside_mask]
pop_values = pop_values_all[inside_mask]
print(f"[확인] 인구 격자 개수: {len(pop_xy):,}개, 총 인구: {pop_values.sum():,.0f}")


# =========================================================
# 6. 두 가지 지표 함수
# =========================================================
def compute_E_weighted(points, pop_xy, pop_values, percentile=D_MAX_PERCENTILE):
    tree = cKDTree(points)
    distances, nearest_idx = tree.query(pop_xy)
    d_km = distances / 1000
    D_avg = np.sum(pop_values * d_km) / np.sum(pop_values)

    # D_max 대신 인구가중 상위 percentile 지점 거리(D_p95 등) 사용.
    # 이유: 극단적 이상치(무인지역 등) 단 하나가 D_max를 지배해서
    # 최적화가 그 지점 하나 때문에 시설을 억지로 외곽으로 보내는
    # 부작용을 줄이기 위함 (SLA에서 max 대신 p95/p99를 쓰는 것과 같은 이유)
    order = np.argsort(d_km)
    sorted_d, sorted_w = d_km[order], pop_values[order]
    cum_ratio = np.cumsum(sorted_w) / np.sum(sorted_w)
    D_pct = sorted_d[np.searchsorted(cum_ratio, percentile)]

    n = len(points)
    pop_per = np.array([pop_values[nearest_idx == i].sum() for i in range(n)])
    P_bar = pop_per.mean()
    sigma_P = pop_per.std()
    CV_P = sigma_P / P_bar if P_bar > 0 else 0.0
    return 1 / ((D_avg + D_pct) * (1 + CV_P) * (D_pct / D_avg))

def compute_coverage(points, pop_xy, pop_values, radius_km):
    tree = cKDTree(points)
    distances, _ = tree.query(pop_xy)
    covered = (distances / 1000) <= radius_km
    return pop_values[covered].sum() / pop_values.sum()


# =========================================================
# 7. [경로 A] 보로노이(E) 최적화 - 로이드 + 모의담금질
# =========================================================
def lloyd_weighted(points, pop_xy, pop_values, boundary_polys, n_iter):
    points = points.copy()
    for _ in range(n_iter):
        tree = cKDTree(points)
        _, nearest_idx = tree.query(pop_xy)
        new_points = points.copy()
        for i in range(len(points)):
            m = nearest_idx == i
            axy, aw = pop_xy[m], pop_values[m]
            if len(axy) == 0:
                continue
            centroid = np.average(axy, axis=0, weights=aw) if aw.sum() > 0 else axy.mean(axis=0)
            if point_in_city(np.array([centroid[0]]), np.array([centroid[1]]), boundary_polys)[0]:
                new_points[i] = centroid
            else:
                d = np.sum((axy - centroid) ** 2, axis=1)
                new_points[i] = axy[np.argmin(d)]
        points = new_points
    return points

def sa_E(points, pop_xy, pop_values, boundary_polys, xmin, xmax, ymin, ymax,
         n_iter, init_temp, cooling, step_frac, rng):
    best_p = points.copy()
    best_E = compute_E_weighted(points, pop_xy, pop_values)
    cur_p, cur_E = points.copy(), best_E
    T = init_temp
    step = step_frac * (xmax - xmin)
    for _ in range(n_iter):
        idx = rng.integers(len(points))
        newp = cur_p.copy()
        cand = newp[idx] + rng.normal(0, step, size=2)
        if not point_in_city(np.array([cand[0]]), np.array([cand[1]]), boundary_polys)[0]:
            continue
        newp[idx] = cand
        newE = compute_E_weighted(newp, pop_xy, pop_values)
        delta = newE - cur_E
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-9)):
            cur_p, cur_E = newp, newE
            if newE > best_E:
                best_E, best_p = newE, newp.copy()
        T *= cooling
    return best_p, best_E


# =========================================================
# 8. [경로 B] 커버리지 최적화 - 탐욕(MCLP) + 모의담금질
# =========================================================
def greedy_mclp(candidates, pop_xy, pop_values, radius_km, n_facilities):
    tree_pop = cKDTree(pop_xy)
    covered_mask = np.zeros(len(pop_xy), dtype=bool)
    selected = []
    remaining = candidates.copy()
    for _ in range(n_facilities):
        best_gain, best_i = -1, 0
        for i, cand in enumerate(remaining):
            idxs = tree_pop.query_ball_point(cand, r=radius_km * 1000)
            newly = np.zeros(len(pop_xy), dtype=bool)
            newly[idxs] = True
            newly &= ~covered_mask
            gain = pop_values[newly].sum()
            if gain > best_gain:
                best_gain, best_i = gain, i
        chosen = remaining[best_i]
        selected.append(chosen)
        idxs = tree_pop.query_ball_point(chosen, r=radius_km * 1000)
        covered_mask[idxs] = True
        remaining = np.delete(remaining, best_i, axis=0)
    return np.array(selected)

def sa_coverage(points, pop_xy, pop_values, radius_km, boundary_polys,
                 xmin, xmax, ymin, ymax, n_iter, init_temp, cooling, step_frac, rng):
    best_p = points.copy()
    best_cov = compute_coverage(points, pop_xy, pop_values, radius_km)
    cur_p, cur_cov = points.copy(), best_cov
    T = init_temp
    step = step_frac * (xmax - xmin)
    for _ in range(n_iter):
        idx = rng.integers(len(points))
        newp = cur_p.copy()
        cand = newp[idx] + rng.normal(0, step, size=2)
        if not point_in_city(np.array([cand[0]]), np.array([cand[1]]), boundary_polys)[0]:
            continue
        newp[idx] = cand
        new_cov = compute_coverage(newp, pop_xy, pop_values, radius_km)
        delta = new_cov - cur_cov
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-9)):
            cur_p, cur_cov = newp, new_cov
            if new_cov > best_cov:
                best_cov, best_p = new_cov, newp.copy()
        T *= cooling
    return best_p, best_cov


# =========================================================
# 9. 두 경로 각각 최적화 실행
# =========================================================
rng = np.random.default_rng(RANDOM_SEED)

print("\n[경로 A] 보로노이(E) 최적화 진행...")
t0 = time.time()
best_E_points, best_E_val = None, -np.inf
for r in range(N_RESTARTS_E):
    init = random_points_in_boundary(N_FACILITIES, boundary_polys, xmin, xmax, ymin, ymax, rng)
    lp = lloyd_weighted(init, pop_xy, pop_values, boundary_polys, LLOYD_ITERS)
    fp, fE = sa_E(lp, pop_xy, pop_values, boundary_polys, xmin, xmax, ymin, ymax,
                  SA_ITERS_E, 0.02, 0.998, 0.02, rng)
    print(f"  시도 {r+1}/{N_RESTARTS_E}: E = {fE:.5f}")
    if fE > best_E_val:
        best_E_val, best_E_points = fE, fp.copy()
print(f"[경로 A] 완료 [{time.time()-t0:.1f}초]")

print("\n[경로 B] 커버리지 최적화 진행...")
t0 = time.time()
candidate_idx = rng.choice(len(pop_xy), size=min(N_CANDIDATES, len(pop_xy)), replace=False)
candidates = pop_xy[candidate_idx]
greedy_points = greedy_mclp(candidates, pop_xy, pop_values, RADIUS_KM, N_FACILITIES)
best_cov_points, best_cov_val = sa_coverage(
    greedy_points, pop_xy, pop_values, RADIUS_KM, boundary_polys,
    xmin, xmax, ymin, ymax, SA_ITERS_COV, 0.02, 0.999, 0.015, rng
)
print(f"[경로 B] 완료 [{time.time()-t0:.1f}초]")


# =========================================================
# 10. 교차비교 표
#     - 현재 실제 배치 / E 최적화 배치 / 커버리지 최적화 배치
#       각각에 대해 E값과 커버리지율을 모두 계산
# =========================================================
placements = [
    ("현재 실제 배치", current_points),
    ("보로노이(E) 최적화 배치", best_E_points),
    ("커버리지 최적화 배치", best_cov_points),
]

print("\n" + "=" * 60)
print(f"{'배치':<24}{'공간효율성 E':>15}{'골든타임 커버리지':>18}")
print("-" * 60)
results = []
for name, pts in placements:
    E_val = compute_E_weighted(pts, pop_xy, pop_values)
    cov_val = compute_coverage(pts, pop_xy, pop_values, RADIUS_KM)
    results.append((name, E_val, cov_val))
    print(f"{name:<24}{E_val:>15.5f}{cov_val*100:>17.1f}%")
print("=" * 60)
print("* 대각선(자기 목적함수 기준)에서 각 배치가 가장 좋게 나오는 것이 정상입니다.")
print("  즉 'E 최적화 배치'는 E에서, '커버리지 최적화 배치'는 커버리지에서 최고여야 합니다.")


# =========================================================
# 11. 시각화: 두 최적화 배치를 각각 두 관점(보로노이 셀 / 커버리지 원)으로
# =========================================================
def plot_voronoi_cells(ax, points, pop_xy, pop_values, radius_km, boundary_polys, title, E_val, cov_val):
    n = len(points)
    tree = cKDTree(points)
    distances, nearest_idx = tree.query(pop_xy)
    covered = (distances / 1000) <= radius_km
    cmap = plt.get_cmap('tab20', n)
    sizes = np.clip(pop_values / max(pop_values.max(), 1) * 15, 1, 15)
    ax.scatter(pop_xy[covered, 0], pop_xy[covered, 1], c=nearest_idx[covered],
               cmap=cmap, vmin=0, vmax=n - 1, s=sizes[covered], marker='o', alpha=0.8)
    ax.scatter(pop_xy[~covered, 0], pop_xy[~covered, 1], c=nearest_idx[~covered],
               cmap=cmap, vmin=0, vmax=n - 1, s=sizes[~covered] + 4, marker='x', alpha=0.9)
    for poly in boundary_polys:
        pc = np.vstack([poly, poly[0]])
        ax.plot(pc[:, 0], pc[:, 1], 'k-', linewidth=0.6, alpha=0.7)
    ax.plot(points[:, 0], points[:, 1], 'k^', markersize=9)
    ax.set_aspect('equal')
    ax.set_title(f"{title}\nE={E_val:.4f}, 커버리지={cov_val*100:.1f}%")

def plot_coverage_circles(ax, points, pop_xy, pop_values, radius_km, boundary_polys, title, E_val, cov_val):
    tree = cKDTree(points)
    distances, _ = tree.query(pop_xy)
    covered = (distances / 1000) <= radius_km
    ax.scatter(pop_xy[~covered, 0], pop_xy[~covered, 1], c='lightgray',
               s=np.clip(pop_values[~covered] / max(pop_values.max(), 1) * 15, 1, 15))
    ax.scatter(pop_xy[covered, 0], pop_xy[covered, 1], c='tomato',
               s=np.clip(pop_values[covered] / max(pop_values.max(), 1) * 15, 1, 15), alpha=0.7)
    for poly in boundary_polys:
        pc = np.vstack([poly, poly[0]])
        ax.plot(pc[:, 0], pc[:, 1], 'k-', linewidth=0.5, alpha=0.6)
    for p in points:
        circle = plt.Circle(p, radius_km * 1000, color='blue', fill=False, linewidth=0.8, alpha=0.5)
        ax.add_patch(circle)
    ax.plot(points[:, 0], points[:, 1], 'b^', markersize=9)
    ax.set_aspect('equal')
    ax.set_title(f"{title}\nE={E_val:.4f}, 커버리지={cov_val*100:.1f}%")

fig, axes = plt.subplots(2, 3, figsize=(20, 13))
labels = ["현재 실제 배치", "보로노이(E) 최적화", "커버리지 최적화"]
pts_list = [current_points, best_E_points, best_cov_points]

for col, (label, pts, res) in enumerate(zip(labels, pts_list, results)):
    _, E_val, cov_val = res
    plot_voronoi_cells(axes[0, col], pts, pop_xy, pop_values, RADIUS_KM, boundary_polys, label, E_val, cov_val)
    plot_coverage_circles(axes[1, col], pts, pop_xy, pop_values, RADIUS_KM, boundary_polys, label, E_val, cov_val)

plt.tight_layout()
plt.show()