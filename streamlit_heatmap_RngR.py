import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as la
import matplotlib.image as mpimg
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse

# =========================
# パス設定（要変更）
# =========================
# --- CSVファイルの相対パス ---
CSV_PATH = "hittingdata.csv"

# --- 画像の相対パス ---
IMAGE_PATH = "baseballfield.jpg"

# --- 守備位置ごとの CSV ---
ELLIPSE_CSV_PATHS = {
    "SS": "SAO-RBB(SS).csv",
    "2B": "SAO-RBB(2B).csv",
    "3B": "SAO-RBB(3B).csv",
    "1B": "SAO-RBB(1B).csv",
}

# Streamlit側クリック画像サイズ
REC_WIDTH, REC_HEIGHT = 750, 750

# ヒストグラム
X_BINS = 80
Y_BINS = 80

# 元画像座標（あなたのコードと同じ）
home = (633.0, 1071.0)
ray_origin = (633.0, 428.0)
ray_origin_near = (633.0, 737.0)
left_pt = (240.0, 682.0)
left_pt_near = (466.0, 904.0)
right_pt = (1033.0, 682.0)
right_pt_near = (800.0, 904.0)

# 初期守備位置（画像座標：y下向き）
infield_positions_img = {
    "1B": (869, 741),
    "2B": (742, 600),
    "SS": (526, 600),
    "3B": (397, 741),
    "LF": (313, 414),
    "CF": (633, 296),
    "RF": (953, 414),
}

# 回転（反時計回り＋）
ROT_DEG = {"SS": +35, "3B": +35, "2B": -35, "1B": -35}


# ------------------- helper -------------------
def read_csv_flexible(path):
    for enc in ("cp932", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)


def angle_from_home_deg(x, y, home_x=home[0], home_y=home[1]):
    dx = x - home_x
    dy = home_y - y
    return math.degrees(math.atan2(dy, dx))


def dir_vector_from_home_angle(angle_deg):
    rad = math.radians(angle_deg)
    ux = math.cos(rad)
    uy = -math.sin(rad)
    return (ux, uy)


def intersect_lines(p1, v1, p2, v2, eps=1e-9):
    A = np.array([[v1[0], -v2[0]], [v1[1], -v2[1]]], dtype=float)
    b = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)
    det = np.linalg.det(A)
    if abs(det) < eps:
        return None
    t, s = la.solve(A, b)
    return (p1[0] + t * v1[0], p1[1] + t * v1[1])


def precompute_rotated_uv(xs, ys, cx, cy, theta_deg):
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    dx = xs - cx
    dy = ys - cy
    u = c * dx + s * dy
    v = -s * dx + c * dy
    return u, v, c, s


def shift_to_du_dv(dx_shift, dy_shift, c, s):
    # シフト(dx,dy)が回転座標(u,v)でどれだけの平行移動になるか
    du = c * dx_shift + s * dy_shift
    dv = -s * dx_shift + c * dy_shift
    return du, dv


@st.cache_data
def load_main_data():
    df = pd.read_csv(CSV_PATH, encoding="cp932")

    # 投手名（あなたの列名：pitchername）
    df["pitchername"] = df["pitchername"].astype(str)
    df = df[df["pitchername"].notna()]
    # df = df[df["pitchername"] != "なし"]  # 必要なら復活

    # フィルタ列を文字列化（値の揺れ対策）
    for c in [
        "opponents",
        "pitch_course",
        "pitch_height",
        "pitch_type",
        "player_batLR",
    ]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    return df


@st.cache_data
def load_ellipse_tables():
    tables = {}
    for pos, path in ELLIPSE_CSV_PATHS.items():
        d = read_csv_flexible(path).copy()
        for c in ["center_x", "center_y", "major-axis", "minor-axis"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        tables[pos] = d
    return tables


def build_heatmap(df_filtered, img, w_img, h_img, use_projection=True):
    # クリック→元画像座標
    scale_x = w_img / REC_WIDTH
    scale_y = h_img / REC_HEIGHT

    df_filtered = df_filtered.copy()
    df_filtered["x_click"] = pd.to_numeric(df_filtered["x_coord"], errors="coerce")
    df_filtered["y_click"] = pd.to_numeric(df_filtered["y_coord"], errors="coerce")
    df_filtered["x_coord_img"] = df_filtered["x_click"] * scale_x
    df_filtered["y_coord_img"] = df_filtered["y_click"] * scale_y

    # ゴロのみ
    df_ground = df_filtered[
        df_filtered["hit_type"].str.contains("ゴロ", na=False)
    ].copy()
    if len(df_ground) == 0:
        return None

    df_ground["angle_deg"] = df_ground.apply(
        lambda r: angle_from_home_deg(r["x_coord_img"], r["y_coord_img"]), axis=1
    )

    # ファウル除外
    angle_left = angle_from_home_deg(left_pt[0], left_pt[1])
    angle_right = angle_from_home_deg(right_pt[0], right_pt[1])
    a_min, a_max = min(angle_left, angle_right), max(angle_left, angle_right)

    df_ground = df_ground[
        df_ground["angle_deg"].between(a_min, a_max, inclusive="both")
    ].copy()
    if len(df_ground) == 0:
        return None

    # ------------------- ホーム側ゴロの除外（第2境界線） -------------------
    v_left_near = (
        left_pt_near[0] - ray_origin_near[0],
        left_pt_near[1] - ray_origin_near[1],
    )
    v_right_near = (
        right_pt_near[0] - ray_origin_near[0],
        right_pt_near[1] - ray_origin_near[1],
    )

    keep_mask = []

    for _, row in df_ground.iterrows():
        x = row["x_coord_img"]
        y = row["y_coord_img"]

        if not np.isfinite(x) or not np.isfinite(y):
            keep_mask.append(False)
            continue

        theta = row["angle_deg"]
        v_home = dir_vector_from_home_angle(theta)

        # 左右どちらの境界を使うか
        v_bound = v_left_near if x <= ray_origin_near[0] else v_right_near

        inter = intersect_lines(home, v_home, ray_origin_near, v_bound)

        if inter is None:
            keep_mask.append(False)
            continue

        ix, iy = inter

        d_ball = math.hypot(x - home[0], y - home[1])
        d_inter = math.hypot(ix - home[0], iy - home[1])

        # ★ 境界線より「奥」にある打球だけ残す
        keep_mask.append(d_ball >= d_inter)

    df_ground = df_ground[np.array(keep_mask)].copy()

    if len(df_ground) == 0:
        return None

    # ------------------- 投影（ON/OFF切替） -------------------
    if use_projection:
        v_left = (left_pt[0] - ray_origin[0], left_pt[1] - ray_origin[1])
        v_right = (right_pt[0] - ray_origin[0], right_pt[1] - ray_origin[1])

        proj_x, proj_y = [], []
        for _, row in df_ground.iterrows():
            x = row["x_coord_img"]
            y = row["y_coord_img"]
            if not np.isfinite(x) or not np.isfinite(y):
                proj_x.append(x)
                proj_y.append(y)
                continue

            v_home = dir_vector_from_home_angle(row["angle_deg"])
            v_bound = v_left if x <= ray_origin[0] else v_right
            inter = intersect_lines(home, v_home, ray_origin, v_bound)

            if inter is None:
                proj_x.append(x)
                proj_y.append(y)
                continue

            ix, iy = inter
            d_ball = math.hypot(x - home[0], y - home[1])
            d_inter = math.hypot(ix - home[0], iy - home[1])

            if d_ball > d_inter:
                proj_x.append(ix)
                proj_y.append(iy)
            else:
                proj_x.append(x)
                proj_y.append(y)

        df_ground["x_proj_img"] = proj_x
        df_ground["y_proj_img"] = proj_y

    else:
        # 投影なし：元の座標をそのまま使う
        df_ground["x_proj_img"] = df_ground["x_coord_img"]
        df_ground["y_proj_img"] = df_ground["y_coord_img"]

    # math座標へ
    df_ground["x_math"] = df_ground["x_proj_img"]
    df_ground["y_math"] = h_img - df_ground["y_proj_img"]

    x_plot = df_ground["x_math"].values
    y_plot = df_ground["y_math"].values

    counts, xedges, yedges = np.histogram2d(
        x_plot, y_plot, bins=[X_BINS, Y_BINS], range=[[0, w_img], [0, h_img]]
    )

    counts_masked = counts.astype(float)
    counts_masked[counts_masked <= 0] = np.nan

    non_zero = counts_masked[~np.isnan(counts_masked)]
    if non_zero.size > 0:
        low_p = 10
        high_p = 99.5
        vmin = np.percentile(non_zero, low_p)
        vmax = np.percentile(non_zero, high_p)
        counts_masked[counts_masked < vmin] = np.nan
        vmin = max(vmin, 1e-3)
        vmax = max(vmax, vmin * 1.01)
    else:
        vmin, vmax = 1e-3, 1.0

    norm = LogNorm(vmin=vmin, vmax=vmax)
    return counts_masked, xedges, yedges, norm, len(df_ground), df_ground


def add_ellipse(ax, pos, row, h_img, dx_extra=0.0, dy_extra=0.0, **style):
    base_x_img, base_y_img = infield_positions_img[pos]
    angle = ROT_DEG[pos]

    name = str(row["NAME"])

    dx = float(row["center_x"]) + float(dx_extra)
    dy = float(row["center_y"]) + float(dy_extra)

    a = float(row["major-axis"])
    b = float(row["minor-axis"])

    cx_img = base_x_img + dx
    cy_img = base_y_img + dy

    cx = cx_img
    cy = h_img - cy_img

    e = Ellipse(
        xy=(cx, cy),
        width=2 * a,
        height=2 * b,
        angle=angle,
        fill=False,
        **({"linewidth": 2} | style),
    )
    ax.add_patch(e)

    # 名前は通常描画のときだけ出したいなら、ここは呼び出し側で制御してOK
    ax.text(cx + 6, cy + 6, name, fontsize=9)


def precompute_uv(xs, ys, cx, cy, theta_deg):
    """回転楕円判定のためのu,vを前計算"""
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    dx = xs - cx
    dy = ys - cy
    u = c * dx + s * dy
    v = -s * dx + c * dy
    return u, v, c, s


def shift_to_du_dv(dx_shift, dy_shift, c, s):
    """(dx,dy)のシフトを回転座標(u,v)の平行移動(du,dv)に変換"""
    du = c * dx_shift + s * dy_shift
    dv = -s * dx_shift + c * dy_shift
    return du, dv


def ellipse_mask_from_uv(u, v, a, b, du=0.0, dv=0.0):
    """(u-du, v-dv)が楕円内か（全点一括）"""
    uu = (u - du) / a
    vv = (v - dv) / b
    return (uu * uu + vv * vv) <= 1.0


def make_ellipse_params(pos, row, h_img, dx_extra=0.0, dy_extra=0.0):
    """
    pos: "SS","2B","3B","1B"
    row: SAO-RBBの該当行（NAME, center_x, center_y, major-axis, minor-axis）
    dx_extra, dy_extra: 追加で動かす量（画像座標系のpx）
    戻り: dict(cx, cy, a, b, theta)
    """
    base_x_img, base_y_img = infield_positions_img[pos]
    theta = ROT_DEG[pos]

    dx = float(row["center_x"]) + float(dx_extra)
    dy = float(row["center_y"]) + float(dy_extra)

    a = float(row["major-axis"])
    b = float(row["minor-axis"])

    cx_img = base_x_img + dx
    cy_img = base_y_img + dy

    # math座標へ（あなたの描画座標に合わせる）
    cx = cx_img
    cy = h_img - cy_img

    return {"cx": cx, "cy": cy, "a": a, "b": b, "theta": theta}


def inside_rotated_ellipse(x, y, cx, cy, a, b, theta_deg):
    """
    (x,y) が中心(cx,cy), 半径a,b, 角度theta(反時計回り+)の回転楕円の内側ならTrue
    """
    th = np.deg2rad(theta_deg)
    dx = x - cx
    dy = y - cy
    xr = np.cos(th) * dx + np.sin(th) * dy
    yr = -np.sin(th) * dx + np.cos(th) * dy
    return (xr / a) ** 2 + (yr / b) ** 2 <= 1.0


def calc_out_rate(df_ground, ellipse_list):
    """
    df_ground: build_heatmapで作った点群（x_math,y_math列がある）
    ellipse_list: make_ellipse_paramsで作ったdictのlist
    戻り: out_rate (0~1)
    """
    if df_ground is None or len(df_ground) == 0:
        return np.nan

    xs = df_ground["x_math"].to_numpy(dtype=float)
    ys = df_ground["y_math"].to_numpy(dtype=float)

    out = np.zeros(len(xs), dtype=np.int8)

    # 1点ずつでもOKだが、まずは確実に動く実装
    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        hit = True
        for e in ellipse_list:
            if inside_rotated_ellipse(
                x, y, e["cx"], e["cy"], e["a"], e["b"], e["theta"]
            ):
                hit = False
                break
        out[i] = 0 if hit else 1

    return out.mean()


def out_rate_from_points(xs, ys, ellipse_list):
    """
    xs, ys: 1D numpy arrays (math座標)
    ellipse_list: make_ellipse_params の dict のlist
    戻り: out_rate (0~1)
    """
    if xs.size == 0:
        return np.nan

    out_mask = np.zeros(xs.shape[0], dtype=bool)

    for e in ellipse_list:
        cx, cy, a, b, th = e["cx"], e["cy"], e["a"], e["b"], e["theta"]
        th = np.deg2rad(th)
        c, s = np.cos(th), np.sin(th)

        dx = xs - cx
        dy = ys - cy

        # 回転座標
        u = c * dx + s * dy
        v = -s * dx + c * dy

        inside = (u / a) ** 2 + (v / b) ** 2 <= 1.0
        out_mask |= inside

    return out_mask.mean()


# =========================
# UI
# =========================
st.title("Pitcher Heatmap + Defensive Range (Filters)")

df = load_main_data()
ellipse_tables = load_ellipse_tables()

# 投手名
pitchers = sorted(df["pitchername"].dropna().unique().tolist())
pitcher = st.selectbox("投手名を選択", pitchers, index=0)

# フィルタ候補を投手で絞った範囲から作る
df_p = df[df["pitchername"] == pitcher].copy()

st.subheader("投球条件フィルタ（複数選択可・未選択なら全て）")


def multiselect_filter(label, col):
    values = (
        sorted(df_p[col].dropna().astype(str).unique().tolist())
        if col in df_p.columns
        else []
    )
    return st.multiselect(label, options=values, default=[])


sel_opponents = multiselect_filter("対戦相手", "opponents")
sel_course = multiselect_filter("pitch_course（コース）", "pitch_course")
sel_height = multiselect_filter("pitch_height（高さ）", "pitch_height")
sel_type = multiselect_filter("pitch_type（球種）", "pitch_type")
sel_lr = multiselect_filter("player_batLR（打者左右）", "player_batLR")

# ★ 追加：投影ON/OFF
use_projection = st.checkbox(
    "境界線への投影を使う（深いゴロを境界線に落とし込む）", value=False
)

# 守備範囲（選手）
st.subheader("守備範囲を表示する選手（各ポジション）")
selected_player = {}
for pos in ["SS", "2B", "3B", "1B"]:
    t = ellipse_tables[pos]
    names = sorted(t["NAME"].dropna().astype(str).unique().tolist())
    selected_player[pos] = st.selectbox(
        f"{pos} 選手名", ["（表示しない）"] + names, index=0
    )

# =========================
# フィルタ適用
# =========================
df_f = df_p.copy()
if sel_opponents:
    df_f = df_f[df_f["opponents"].astype(str).isin(sel_opponents)]
if sel_course:
    df_f = df_f[df_f["pitch_course"].astype(str).isin(sel_course)]
if sel_height:
    df_f = df_f[df_f["pitch_height"].astype(str).isin(sel_height)]
if sel_type:
    df_f = df_f[df_f["pitch_type"].astype(str).isin(sel_type)]
if sel_lr:
    df_f = df_f[df_f["player_batLR"].astype(str).isin(sel_lr)]

st.caption(
    f"フィルタ後の打球データ件数：{len(df_f)}（この中からゴロのみ抽出してヒートマップ化します）"
)


# =========================
# 描画
# =========================

img = mpimg.imread(IMAGE_PATH)
h_img, w_img = img.shape[0], img.shape[1]
img_flipped = np.flipud(img)

res = build_heatmap(df_f, img, w_img, h_img, use_projection=use_projection)

if res is None:
    st.warning("フィルタ後のゴロデータが0件です。条件を緩めてください。")
else:
    opt_result = None
    opt_delta = None
    opt_best_out = None
    counts_masked, xedges, yedges, norm, n_ground, df_ground = res
    st.caption(f"ヒートマップに使ったゴロ数：{n_ground}")

    fig, ax = plt.subplots(figsize=(6, 8))

    # 背景
    ax.imshow(img_flipped, extent=[0, w_img, 0, h_img], origin="lower")

    # ヒートマップ
    cmap = plt.cm.get_cmap("inferno", 1024)
    hm = ax.imshow(
        counts_masked.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap=cmap,
        norm=norm,
        alpha=0.6,
    )
    fig.colorbar(hm, ax=ax, label="Hit density (log scale)")

    # 守備位置（参考点）
    for name, (px, py_img) in infield_positions_img.items():
        py_math = h_img - py_img
        ax.scatter(px, py_math, s=60)
        ax.text(px + 6, py_math + 6, name)

    # 楕円（選ばれた選手だけ）
    for pos in ["SS", "2B", "3B", "1B"]:
        pick = selected_player[pos]
        if pick == "（表示しない）":
            continue
        t = ellipse_tables[pos]
        row = t[t["NAME"].astype(str) == pick].iloc[0]
        add_ellipse(ax, pos, row, h_img)

    # =========================
    # Out% / ΔOut% 計算
    # =========================
    st.subheader("Out%（楕円モデル）と シフト価値 ΔOut%（簡易探索）")

    # 選択された選手の楕円をパラメータ化（通常守備）
    ellipse_params_normal = []
    rows_by_pos = {}  # 後で探索用に使う

    for pos in ["SS", "2B", "3B", "1B"]:
        pick = selected_player[pos]
        if pick == "（表示しない）":
            continue
        t = ellipse_tables[pos]
        row = t[t["NAME"].astype(str) == pick].iloc[0]
        rows_by_pos[pos] = row
        ellipse_params_normal.append(make_ellipse_params(pos, row, h_img))

    out_normal = calc_out_rate(df_ground, ellipse_params_normal)
    st.write(f"通常守備 Out%（モデル）: **{out_normal:.3f}**")
    st.write(f"通常守備 BA換算（≒1-Out%）: **{(1-out_normal):.3f}**")

    # 探索設定（軽め）

    do_opt = st.checkbox(
        "SS・2Bの位置を動かして最適Out%を探す（簡易グリッド探索）", value=False
    )

    if do_opt:
        max_shift = st.slider(
            "探索幅（±px）", min_value=0, max_value=60, value=20, step=5
        )
        step = st.slider("刻み（px）", min_value=1, max_value=20, value=10, step=1)

        move_ss = st.checkbox("SSを動かす", value=True)
        move_2b = st.checkbox("2Bを動かす", value=True)

        if (move_ss and "SS" not in rows_by_pos) or (
            move_2b and "2B" not in rows_by_pos
        ):
            st.warning(
                "探索するポジションの選手が未選択です（SS/2Bを選んでください）。"
            )
        else:
            # ---------------------------
            # データ点（math座標）
            # ---------------------------
            xs = df_ground["x_math"].to_numpy(dtype=float)
            ys = df_ground["y_math"].to_numpy(dtype=float)
            n = len(xs)
            if n == 0:
                st.warning("ゴロデータが0件です。")
            else:
                # ---------------------------
                # ② 固定ポジション（1B/3B）のmaskを1回だけ計算して使い回し
                # ---------------------------
                fixed_mask = np.zeros(n, dtype=bool)

                for pos in ["1B", "3B"]:
                    if pos not in rows_by_pos:
                        continue
                    e = make_ellipse_params(
                        pos, rows_by_pos[pos], h_img, dx_extra=0, dy_extra=0
                    )
                    u, v, c, s = precompute_uv(xs, ys, e["cx"], e["cy"], e["theta"])
                    fixed_mask |= ellipse_mask_from_uv(
                        u, v, e["a"], e["b"], du=0.0, dv=0.0
                    )

                # ---------------------------
                # ① SS/2B は前計算（u,v,c,s）を作り、シフトごとは(du,dv)だけで判定
                # ---------------------------
                ss_pack = None
                b2_pack = None

                if move_ss and "SS" in rows_by_pos:
                    e0 = make_ellipse_params(
                        "SS", rows_by_pos["SS"], h_img, dx_extra=0, dy_extra=0
                    )
                    u, v, c, s = precompute_uv(xs, ys, e0["cx"], e0["cy"], e0["theta"])
                    ss_pack = (e0, u, v, c, s)

                if move_2b and "2B" in rows_by_pos:
                    e0 = make_ellipse_params(
                        "2B", rows_by_pos["2B"], h_img, dx_extra=0, dy_extra=0
                    )
                    u, v, c, s = precompute_uv(xs, ys, e0["cx"], e0["cy"], e0["theta"])
                    b2_pack = (e0, u, v, c, s)

                # ---------------------------
                # 探索関数（ベクトル化）
                # ---------------------------
                def search_best(shifts_ss, shifts_b2):
                    best_out = -1.0
                    best_cfg = {"SS_dx": 0, "SS_dy": 0, "2B_dx": 0, "2B_dy": 0}

                    # SS側の候補
                    ss_dx_list = shifts_ss if ss_pack is not None else [0]
                    ss_dy_list = shifts_ss if ss_pack is not None else [0]
                    b2_dx_list = shifts_b2 if b2_pack is not None else [0]
                    b2_dy_list = shifts_b2 if b2_pack is not None else [0]

                    for ss_dx in ss_dx_list:
                        for ss_dy in ss_dy_list:
                            ss_mask = None
                            if ss_pack is not None:
                                e0, u, v, c, s = ss_pack
                                du, dv = shift_to_du_dv(ss_dx, ss_dy, c, s)
                                ss_mask = ellipse_mask_from_uv(
                                    u, v, e0["a"], e0["b"], du=du, dv=dv
                                )

                            for b2_dx in b2_dx_list:
                                for b2_dy in b2_dy_list:
                                    b2_mask = None
                                    if b2_pack is not None:
                                        e0, u, v, c, s = b2_pack
                                        du, dv = shift_to_du_dv(b2_dx, b2_dy, c, s)
                                        b2_mask = ellipse_mask_from_uv(
                                            u, v, e0["a"], e0["b"], du=du, dv=dv
                                        )

                                    # ★固定maskにORするだけ（copyしない）
                                    out_mask = fixed_mask
                                    if ss_mask is not None:
                                        out_mask = out_mask | ss_mask
                                    if b2_mask is not None:
                                        out_mask = out_mask | b2_mask

                                    out_rate = out_mask.mean()

                                    if out_rate > best_out:
                                        best_out = out_rate
                                        best_cfg = {
                                            "SS_dx": int(ss_dx),
                                            "SS_dy": int(ss_dy),
                                            "2B_dx": int(b2_dx),
                                            "2B_dy": int(b2_dy),
                                        }
                    return best_out, best_cfg

                # ---------------------------
                # ③ 粗→細探索（機能は同じ、速い）
                # ---------------------------
                coarse_step = max(1, step)  # UIのstepを粗探索に使う
                fine_step = max(1, step // 3)  # 細探索は3倍くらい細かく（好みで調整可）
                shifts_coarse = list(range(-max_shift, max_shift + 1, coarse_step))

                with st.spinner("粗探索中..."):
                    best_out_c, best_cfg_c = search_best(shifts_coarse, shifts_coarse)

                # 粗探索の最良点の近傍だけ細探索
                def around(center, radius, step_):
                    start = int(center - radius)
                    end = int(center + radius)
                    return list(range(start, end + 1, step_))

                radius = coarse_step  # 近傍幅（粗刻み1つ分）
                ss_dx0, ss_dy0 = best_cfg_c["SS_dx"], best_cfg_c["SS_dy"]
                b2_dx0, b2_dy0 = best_cfg_c["2B_dx"], best_cfg_c["2B_dy"]

                # SS/2Bそれぞれの近傍を作る（同じリストで回してもOKだが、ここは範囲を詰める）
                shifts_fine_ss = sorted(
                    set(
                        around(ss_dx0, radius, fine_step)
                        + around(ss_dy0, radius, fine_step)
                    )
                )
                shifts_fine_b2 = sorted(
                    set(
                        around(b2_dx0, radius, fine_step)
                        + around(b2_dy0, radius, fine_step)
                    )
                )

                with st.spinner("細探索中..."):
                    best_out, best_cfg = search_best(shifts_fine_ss, shifts_fine_b2)

                delta = best_out - out_normal
                opt_result = best_cfg
                opt_delta = delta
                opt_best_out = best_out

                st.write(f"最適Out%（探索）: **{best_out:.3f}**")
                st.write(f"ΔOut%（最適−通常）: **{delta:+.3f}**")
                st.caption(f"最適シフト（追加移動量, px）: {best_cfg}")

                # =========================
                # ΔOut% の不確実性（軽量ブートストラップ：best_cfg固定）
                # =========================
                st.subheader("ΔOut% の不確実性（ブートストラップ）※軽量版")

                do_ci = st.checkbox(
                    "ΔOut%の信頼区間を出す（best_cfg固定）", value=False
                )

                if do_ci:
                    B = st.slider("反復回数（多いほど安定・重くなる）", 20, 200, 60, 10)
                    seed = st.number_input("乱数seed（再現用）", value=0, step=1)

                    xs_all = df_ground["x_math"].to_numpy(dtype=float)
                    ys_all = df_ground["y_math"].to_numpy(dtype=float)
                    n_all = xs_all.size

                    if n_all < 5:
                        st.warning("ゴロ数が少なすぎて信頼区間が安定しません。")
                    else:
                        # 通常守備の楕円（そのまま）
                        ell_normal = []
                        for pos in ["SS", "2B", "3B", "1B"]:
                            if pos not in rows_by_pos:
                                continue
                            ell_normal.append(
                                make_ellipse_params(pos, rows_by_pos[pos], h_img)
                            )

                        # 最適シフト適用（SS/2Bだけ動かす。best_cfg固定）
                        ell_shift = []
                        for pos in ["SS", "2B", "3B", "1B"]:
                            if pos not in rows_by_pos:
                                continue

                            dx_extra = 0.0
                            dy_extra = 0.0
                            if opt_result is not None and pos in ["SS", "2B"]:
                                dx_extra = float(opt_result.get(f"{pos}_dx", 0))
                                dy_extra = float(opt_result.get(f"{pos}_dy", 0))

                            ell_shift.append(
                                make_ellipse_params(
                                    pos,
                                    rows_by_pos[pos],
                                    h_img,
                                    dx_extra=dx_extra,
                                    dy_extra=dy_extra,
                                )
                            )

                        # 元データでの値（best_cfg固定で再計算：表示と整合させる）
                        out_n0 = out_rate_from_points(xs_all, ys_all, ell_normal)
                        out_s0 = out_rate_from_points(xs_all, ys_all, ell_shift)
                        delta0 = out_s0 - out_n0

                        rng = np.random.default_rng(int(seed))
                        deltas = np.empty(B, dtype=float)

                        # ブートストラップ（復元抽出）
                        for i in range(B):
                            idx = rng.integers(0, n_all, size=n_all)
                            out_n = out_rate_from_points(
                                xs_all[idx], ys_all[idx], ell_normal
                            )
                            out_s = out_rate_from_points(
                                xs_all[idx], ys_all[idx], ell_shift
                            )
                            deltas[i] = out_s - out_n

                        lo, hi = np.percentile(deltas, [2.5, 97.5])

                        st.write(f"通常 Out%（再計算）: **{out_n0:.3f}**")
                        st.write(f"シフト Out%（best_cfg固定）: **{out_s0:.3f}**")
                        st.write(f"ΔOut%（best_cfg固定）: **{delta0:+.3f}**")
                        st.write(
                            f"ΔOut% 95%CI（ブートストラップ）: **[{lo:+.3f}, {hi:+.3f}]**"
                        )
                        st.caption(
                            "※この信頼区間は『最適化の不確実性』ではなく、『データ有限による不確実性』です（best_cfgは固定）。"
                        )

    # ……（ここまでで背景・ヒートマップ・通常楕円は描画済み）

    # =========================
    # 最適配置のオーバーレイ（探索したときだけ）
    # =========================
    show_opt = st.checkbox("最適配置（探索結果）を図に重ねる", value=True)

    if show_opt and (opt_result is not None):
        for pos in ["SS", "2B"]:
            pick = selected_player[pos]
            if pick == "（表示しない）":
                continue

            t = ellipse_tables[pos]
            row = t[t["NAME"].astype(str) == pick].iloc[0]

            dx = float(opt_result.get(f"{pos}_dx", 0))
            dy = float(opt_result.get(f"{pos}_dy", 0))

            ax.scatter(
                infield_positions_img[pos][0] + float(row["center_x"]) + dx,
                h_img - (infield_positions_img[pos][1] + float(row["center_y"]) + dy),
                s=140,
                marker="x",
            )

            add_ellipse(
                ax,
                pos,
                row,
                h_img,
                dx_extra=dx,
                dy_extra=dy,
                linewidth=3,
                linestyle="--",
            )

        ax.text(
            0.02,
            0.98,
            f"Best Out%={opt_best_out:.3f}, ΔOut%={opt_delta:+.3f}",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
        )

    # ★ axis設定は毎回同じに固定（ここで統一）
    ax.set_xlim(0, w_img)
    ax.set_ylim(0, h_img)

    ax.set_aspect("equal", adjustable="box")

    # ★表示は最後に1回だけ
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
