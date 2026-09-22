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
POPULATION_CSV_PATH = r"C:\Users\Lenovo\Downloads\_census_reqdoc_1784693050351\2024년_인구_다사_100M.csv"
CITY_KEYWORD = "성남시"
NAME_KEY_CANDIDATES = ["SIG_KOR_NM", "sggnm", "ADM_NM", "adm_nm", "SGG_NM", "sido_sgg_nm"]
GRID_BOUNDARY_PATH = r"C:\Users\Lenovo\Downloads\_grid_border_grid_2025_grid_다사_grid_다사\grid_다사_100M.shp"

# =========================================================
# 1. 골든타임 기준 설정
# =========================================================
AVG_SPEED_KMH = 25.4
DETOUR_FACTOR = 1.3
GOLDEN_TIME_MIN = 5
RADIUS_KM = (AVG_SPEED_KMH * (GOLDEN_TIME_MIN / 60)) / DETOUR_FACTOR
print(f"[설정] 골든타임 {GOLDEN_TIME_MIN}분 기준 직선거리 반경: {RADIUS_KM:.3f} km")

# =========================================================
# 2. 민감도 분석 설정값
# =========================================================
PERCENTILES = [1.00, 0.99, 0.97, 0.95, 0.93, 0.90, 0.87, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]
N_RESTARTS_E = 2        # percentile마다 재시도 횟수
LLOYD_ITERS = 20
SA_ITERS_E = 1500
N_CANDIDATES = 2000      # 커버리지 최적화(참고선용) 후보지 개수
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
# 6. 지표 함수 (E는 percentile을 인자로 받음)
# =========================================================
def compute_E_weighted(points, pop_xy, pop_values, percentile):
    tree = cKDTree(points)
    distances, nearest_idx = tree.query(pop_xy)
    d_km = distances / 1000
    D_avg = np.sum(pop_values * d_km) / np.sum(pop_values)

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
# 7. 로이드 알고리즘 (percentile과 무관 - 인구가중 무게중심 이동)
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


# =========================================================
# 8. 모의담금질 (주어진 percentile 기준 E를 목표로)
# =========================================================
def sa_E(points, pop_xy, pop_values, percentile, boundary_polys, xmin, xmax, ymin, ymax,
         n_iter, init_temp, cooling, step_frac, rng):
    best_p = points.copy()
    best_E = compute_E_weighted(points, pop_xy, pop_values, percentile)
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
        newE = compute_E_weighted(newp, pop_xy, pop_values, percentile)
        delta = newE - cur_E
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-9)):
            cur_p, cur_E = newp, newE
            if newE > best_E:
                best_E, best_p = newE, newp.copy()
        T *= cooling
    return best_p, best_E


# =========================================================
# 9. (참고선용) 커버리지 최적화 - percentile과 무관하게 한 번만 계산
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
# 10. 현재 배치 기준값 (참고선)
# =========================================================
coverage_current = compute_coverage(current_points, pop_xy, pop_values, RADIUS_KM)
print(f"\n[현재 실제 배치] 골든타임 커버리지: {coverage_current*100:.1f}%")

rng = np.random.default_rng(RANDOM_SEED)
print("\n[참고선] 커버리지 최적화(탐욕+SA) 진행...")
t0 = time.time()
candidate_idx = rng.choice(len(pop_xy), size=min(N_CANDIDATES, len(pop_xy)), replace=False)
candidates = pop_xy[candidate_idx]
greedy_points = greedy_mclp(candidates, pop_xy, pop_values, RADIUS_KM, N_FACILITIES)
_, coverage_ref = sa_coverage(
    greedy_points, pop_xy, pop_values, RADIUS_KM, boundary_polys,
    xmin, xmax, ymin, ymax, SA_ITERS_COV, 0.02, 0.999, 0.015, rng
)
print(f"[참고선] 완료 - 순수 커버리지 최적화 시 커버리지: {coverage_ref*100:.1f}%  [{time.time()-t0:.1f}초]")


# =========================================================
# 11. percentile별로 E 최적화를 반복하며 결과 기록
# =========================================================
results = []
grand_t0 = time.time()

for pi, pct in enumerate(PERCENTILES):
    print(f"\n[percentile {pi+1}/{len(PERCENTILES)}] p={pct:.2f} 진행...")
    t0 = time.time()
    best_E_val, best_E_points = -np.inf, None
    for r in range(N_RESTARTS_E):
        init = random_points_in_boundary(N_FACILITIES, boundary_polys, xmin, xmax, ymin, ymax, rng)
        lp = lloyd_weighted(init, pop_xy, pop_values, boundary_polys, LLOYD_ITERS)
        fp, fE = sa_E(lp, pop_xy, pop_values, pct, boundary_polys, xmin, xmax, ymin, ymax,
                      SA_ITERS_E, 0.02, 0.998, 0.02, rng)
        if fE > best_E_val:
            best_E_val, best_E_points = fE, fp.copy()

    coverage_val = compute_coverage(best_E_points, pop_xy, pop_values, RADIUS_KM)
    elapsed = time.time() - t0
    print(f"  -> E={best_E_val:.5f}, 커버리지={coverage_val*100:.1f}%  [{elapsed:.1f}초]")
    results.append({"percentile": pct, "E": best_E_val, "coverage": coverage_val, "points": best_E_points})

print(f"\n전체 소요시간: {time.time()-grand_t0:.1f}초")


# =========================================================
# 12. 결과 표 출력
# =========================================================
print("\n" + "=" * 55)
print(f"{'percentile':>12}{'E':>14}{'커버리지':>14}")
print("-" * 55)
for r in results:
    print(f"{r['percentile']:>12.2f}{r['E']:>14.5f}{r['coverage']*100:>13.1f}%")
print("-" * 55)
print(f"{'(현재 배치)':>12}{'':>14}{coverage_current*100:>13.1f}%")
print(f"{'(순수 커버리지 최적화)':>12}{'':>14}{coverage_ref*100:>13.1f}%")
print("=" * 55)


# =========================================================
# 13. 민감도 분석 그래프
# =========================================================
pcts = [r["percentile"] * 100 for r in results]
E_vals = [r["E"] for r in results]
cov_vals = [r["coverage"] * 100 for r in results]

fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = 'tab:blue'
ax1.set_xlabel("D_max 대신 사용한 인구가중 백분위수 (%)")
ax1.set_ylabel("최적화된 공간효율성 지표 E", color=color1)
ax1.plot(pcts, E_vals, 'o-', color=color1, label="E (최적화됨)")
ax1.tick_params(axis='y', labelcolor=color1)
ax1.invert_xaxis()  # 100 -> 50 순서로, 왼쪽이 100%가 되도록

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel("그 배치의 골든타임 커버리지 (%)", color=color2)
ax2.plot(pcts, cov_vals, 's-', color=color2, label="커버리지")
ax2.axhline(coverage_current * 100, color='gray', linestyle='--', linewidth=1, label="현재 배치 커버리지")
ax2.axhline(coverage_ref * 100, color='green', linestyle=':', linewidth=1, label="순수 커버리지 최적화")
ax2.tick_params(axis='y', labelcolor=color2)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')

plt.title("D_max 백분위수 기준에 따른 E와 골든타임 커버리지 민감도 분석")
plt.tight_layout()
plt.show()