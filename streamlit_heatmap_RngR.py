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

    # =========================
    # ★ pitch_type を3分類にまとめる
    # =========================
    def map_pitch_group(pt: str) -> str:
        pt = str(pt).strip()
        if pt in ["ストレート", "ツーシーム"]:
            return "ストレート系"
        elif pt in ["カーブ", "スライダー", "カットボール"]:
            return "スライダー系"
        elif pt in ["チェンジアップ", "フォーク"]:
            return "フォーク系"
        else:
            return "その他"

    df["pitch_type_group"] = df["pitch_type"].apply(map_pitch_group)

    return df


def show_dist(df_base, col, title=None, topn=30):
    if col not in df_base.columns:
        return
    n = len(df_base)
    if n == 0:
        st.caption(f"{title or col}: データ0件")
        return

    s = df_base[col].astype(str).fillna("NaN")
    vc = s.value_counts(dropna=False)
    pct = (vc / n * 100).round(1)

    out = pd.DataFrame({"値": vc.index, "件数": vc.values, "割合(%)": pct.values})

    st.caption(f"{title or col}（フィルタ後 n={n} の中での割合。※ゴロ以外も含む）")
    st.dataframe(out.head(topn), use_container_width=True)


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


def to_polar_from_home(df_ground, home_math):
    dx = df_ground["x_math"].to_numpy(float) - home_math[0]
    dy = df_ground["y_math"].to_numpy(float) - home_math[1]
    theta = np.arctan2(dy, dx)  # -pi..pi
    r = np.hypot(dx, dy)
    return theta, r


def build_training_ground(df_all, w_img, h_img, use_projection_for_train=False):
    """
    フェーズ3用：
    全データからゴロのみ抽出し、(x_math,y_math,条件列)を返す
    """
    scale_x = w_img / REC_WIDTH
    scale_y = h_img / REC_HEIGHT

    df = df_all.copy()
    df["x_click"] = pd.to_numeric(df["x_coord"], errors="coerce")
    df["y_click"] = pd.to_numeric(df["y_coord"], errors="coerce")
    df["x_img"] = df["x_click"] * scale_x
    df["y_img"] = df["y_click"] * scale_y

    # ゴロのみ
    df = df[df["hit_type"].str.contains("ゴロ", na=False)].copy()
    if len(df) == 0:
        return None

    # math座標
    df["x_math"] = df["x_img"]
    df["y_math"] = h_img - df["y_img"]

    return df


def xy_to_theta_r(xs, ys, home_math):
    dx = xs - home_math[0]
    dy = ys - home_math[1]
    theta = np.arctan2(dy, dx)  # [-π,π]
    r = np.hypot(dx, dy)
    return theta, r


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


def fit_conditional_hist(
    df_ground,
    home_math,
    feat_cols,
    n_theta=36,
    n_r=18,
    laplace=1.0,
):
    """
    条件→(theta,r) 2Dヒストを作る。
    __ALL__ を必ず作り、キーが無いときはそこへバックオフ。
    """
    xs = df_ground["x_math"].to_numpy(float)
    ys = df_ground["y_math"].to_numpy(float)
    theta, r = xy_to_theta_r(xs, ys, home_math)

    r_max = np.percentile(r, 99.5)
    r_max = max(r_max, 1.0)

    th_edges = np.linspace(-np.pi, np.pi, n_theta + 1)
    r_edges = np.linspace(0, r_max, n_r + 1)

    model = {
        "feat_cols": feat_cols,
        "th_edges": th_edges,
        "r_edges": r_edges,
        "tables": {},
    }

    # ---- 全体（バックオフ） ----
    H_all, _, _ = np.histogram2d(theta, r, bins=[th_edges, r_edges])
    H_all = H_all + laplace
    model["tables"]["__ALL__"] = H_all / H_all.sum()

    # ---- 条件ごと ----
    keys = df_ground[feat_cols].astype(str).agg("|".join, axis=1).to_numpy()
    for k in np.unique(keys):
        m = keys == k
        if m.sum() < 2:
            continue
        H, _, _ = np.histogram2d(theta[m], r[m], bins=[th_edges, r_edges])
        H = H + laplace
        model["tables"][k] = H / H.sum()

    return model


def pick_one_or_blank(sel_list):
    # 未選択→""（バックオフ方向）
    if not sel_list:
        return ""
    # 複数なら先頭1つ（まずはこれでOK）
    return str(sel_list[0])


def generate_ground_xy_from_model(model, cond_dict, home_math, n_samples, seed=0):
    feat_cols = model["feat_cols"]
    key = "|".join([str(cond_dict.get(c, "")) for c in feat_cols])

    H = model["tables"].get(key, model["tables"]["__ALL__"])
    th_edges = model["th_edges"]
    r_edges = model["r_edges"]

    rng = np.random.default_rng(int(seed))

    flat = H.ravel()
    idx = rng.choice(flat.size, size=n_samples, p=flat)

    it = idx // (len(r_edges) - 1)
    ir = idx % (len(r_edges) - 1)

    th0 = th_edges[it]
    th1 = th_edges[it + 1]
    rr0 = r_edges[ir]
    rr1 = r_edges[ir + 1]

    theta = rng.uniform(th0, th1)
    rr = rng.uniform(rr0, rr1)

    x = home_math[0] + rr * np.cos(theta)
    y = home_math[1] + rr * np.sin(theta)

    return x, y, key


def out_rate_weighted_mask(out_mask: np.ndarray, ws: np.ndarray) -> float:
    # out_mask: bool配列（楕円に入った=アウト想定）
    # ws: 重み
    denom = ws.sum()
    if denom <= 0:
        return np.nan
    return (ws * out_mask.astype(float)).sum() / denom


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


def angle_deg_from_home_math(x, y, home_math):
    dx = x - home_math[0]
    dy = y - home_math[1]
    return math.degrees(math.atan2(dy, dx))


def project_points_img(xs_img, ys_img, home, ray_origin, left_pt, right_pt):
    """
    画像座標(y下)の点群を、あなたのロジックで境界線へ投影して返す
    xs_img, ys_img: 1D numpy array
    return: (xp_img, yp_img)
    """
    xs_img = np.asarray(xs_img, dtype=float)
    ys_img = np.asarray(ys_img, dtype=float)

    v_left = (left_pt[0] - ray_origin[0], left_pt[1] - ray_origin[1])
    v_right = (right_pt[0] - ray_origin[0], right_pt[1] - ray_origin[1])

    xp = xs_img.copy()
    yp = ys_img.copy()

    for i in range(xs_img.size):
        x = xs_img[i]
        y = ys_img[i]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue

        # ホーム→打球方向の角度（画像座標）
        theta = angle_from_home_deg(x, y, home_x=home[0], home_y=home[1])
        v_home = dir_vector_from_home_angle(theta)

        v_bound = v_left if x <= ray_origin[0] else v_right
        inter = intersect_lines(home, v_home, ray_origin, v_bound)
        if inter is None:
            continue

        ix, iy = inter
        d_ball = math.hypot(x - home[0], y - home[1])
        d_inter = math.hypot(ix - home[0], iy - home[1])

        # 境界より外（深い）なら交点へ投影
        if d_ball > d_inter:
            xp[i] = ix
            yp[i] = iy

    return xp, yp


def project_points_math(xs_math, ys_math, h_img, home, ray_origin, left_pt, right_pt):
    """
    math座標(y上) → img座標(y下)に戻して投影 → mathに戻す
    """
    xs_math = np.asarray(xs_math, dtype=float)
    ys_math = np.asarray(ys_math, dtype=float)

    xs_img = xs_math
    ys_img = h_img - ys_math

    xp_img, yp_img = project_points_img(
        xs_img, ys_img, home, ray_origin, left_pt, right_pt
    )

    xp_math = xp_img
    yp_math = h_img - yp_img
    return xp_math, yp_math


def filter_generated_points(
    xg,
    yg,
    home_math,
    w_img,
    h_img,
    left_pt_math,
    right_pt_math,
    ray_origin_near_math,
    left_pt_near_math,
    right_pt_near_math,
):
    xg = np.asarray(xg, dtype=float)
    yg = np.asarray(yg, dtype=float)

    # ①画面外除外
    mask = (xg >= 0) & (xg <= w_img) & (yg >= 0) & (yg <= h_img)

    # ②ファウル除外（角度）
    ang_left = angle_deg_from_home_math(left_pt_math[0], left_pt_math[1], home_math)
    ang_right = angle_deg_from_home_math(right_pt_math[0], right_pt_math[1], home_math)
    a_min, a_max = (min(ang_left, ang_right), max(ang_left, ang_right))

    ang = np.array([angle_deg_from_home_math(x, y, home_math) for x, y in zip(xg, yg)])
    mask &= (ang >= a_min) & (ang <= a_max)

    # ③ホーム寄り（第2境界線より手前）を除外
    v_left_near = (
        left_pt_near_math[0] - ray_origin_near_math[0],
        left_pt_near_math[1] - ray_origin_near_math[1],
    )
    v_right_near = (
        right_pt_near_math[0] - ray_origin_near_math[0],
        right_pt_near_math[1] - ray_origin_near_math[1],
    )

    keep2 = np.zeros_like(mask, dtype=bool)

    for i in range(len(xg)):
        if not mask[i]:
            continue

        x, y = xg[i], yg[i]

        dx = x - home_math[0]
        dy = y - home_math[1]
        d_ball = math.hypot(dx, dy)
        if d_ball <= 0:
            continue

        v_home = (dx / d_ball, dy / d_ball)
        v_bound = v_left_near if x <= ray_origin_near_math[0] else v_right_near

        inter = intersect_lines(home_math, v_home, ray_origin_near_math, v_bound)
        if inter is None:
            continue

        ix, iy = inter
        d_inter = math.hypot(ix - home_math[0], iy - home_math[1])

        keep2[i] = d_ball >= d_inter

    mask &= keep2
    return mask


def generate_ground_xy_filtered(
    model,
    cond_dict,
    home_math,
    n_samples,
    seed,
    w_img,
    h_img,
    left_pt_math,
    right_pt_math,
    ray_origin_near_math,
    left_pt_near_math,
    right_pt_near_math,
    oversample=4,
    max_rounds=10,
    use_projection=False,
    h_img_for_projection=None,
    home_img=None,
    ray_origin_img=None,
    left_pt_img=None,
    right_pt_img=None,
):
    """
    生成→除外条件で落とす→必要数に満たなければ追加生成
    ※use_projection=Trueなら最後に生成点も境界線へ投影する
    """
    need = int(n_samples)
    if need <= 0:
        return None, None, None

    xs_list, ys_list = [], []
    used_key_final = None

    for r in range(max_rounds):
        n_try = max(need * oversample, 50)

        xg, yg, used_key = generate_ground_xy_from_model(
            model, cond_dict, home_math, n_samples=n_try, seed=int(seed) + r
        )
        used_key_final = used_key

        # まずフィルタで落とす（投影前座標で）
        m = filter_generated_points(
            xg,
            yg,
            home_math,
            w_img,
            h_img,
            left_pt_math,
            right_pt_math,
            ray_origin_near_math,
            left_pt_near_math,
            right_pt_near_math,
        )

        xk = np.asarray(xg)[m]
        yk = np.asarray(yg)[m]

        if xk.size == 0:
            continue

        # ★必要ならここで投影（フィルタ通過後にやるのが効率良い）
        if use_projection:
            if h_img_for_projection is None:
                raise ValueError(
                    "use_projection=True なのに h_img_for_projection が None です"
                )
            if None in (home_img, ray_origin_img, left_pt_img, right_pt_img):
                raise ValueError(
                    "use_projection=True なのに画像座標の境界パラメータが不足しています"
                )

            xk, yk = project_points_math(
                xk,
                yk,
                h_img_for_projection,
                home_img,
                ray_origin_img,
                left_pt_img,
                right_pt_img,
            )

        take = min(xk.size, need)
        xs_list.append(xk[:take])
        ys_list.append(yk[:take])
        need -= take

        if need <= 0:
            break

    if len(xs_list) == 0:
        return None, None, used_key_final

    return np.concatenate(xs_list), np.concatenate(ys_list), used_key_final


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


def out_rate_weighted(xs, ys, ws, ellipse_list):
    if xs.size == 0:
        return np.nan
    out_mask = np.zeros(xs.shape[0], dtype=bool)

    for e in ellipse_list:
        cx, cy, a, b, th = e["cx"], e["cy"], e["a"], e["b"], e["theta"]
        th = np.deg2rad(th)
        c, s = np.cos(th), np.sin(th)

        dx = xs - cx
        dy = ys - cy
        u = c * dx + s * dy
        v = -s * dx + c * dy
        inside = (u / a) ** 2 + (v / b) ** 2 <= 1.0
        out_mask |= inside

    return (ws * out_mask.astype(float)).sum() / ws.sum()


def out_prob_from_uv(u, v, a, b, du=0.0, dv=0.0, alpha=2.197):
    """
    q = ((u-du)/a)^2 + ((v-dv)/b)^2
    p_out = 1/(1+exp(alpha*(q-1)))
      - 楕円縁(q=1) -> 0.5
      - 中心(q=0)  -> 約0.9（alpha≈2.197のとき）
    """
    q = ((u - du) / a) ** 2 + ((v - dv) / b) ** 2
    z = alpha * (q - 1.0)
    z = np.clip(z, -60, 60)  # overflow防止
    return 1.0 / (1.0 + np.exp(z))


def combine_out_probs(probs_list, n=None):
    """
    複数ポジションのアウト確率を合成（独立近似）
    p_total = 1 - Π(1 - p_i)
    空なら p_out=0 を返す
    """
    if len(probs_list) == 0:
        return None

    p_not = np.ones_like(probs_list[0], dtype=float)
    for p in probs_list:
        p_not *= 1.0 - p
    return 1.0 - p_not


def out_rate_weighted_prob(p_out, ws):
    if p_out is None:
        return np.nan

    denom = ws.sum()
    if denom <= 0:
        return np.nan
    return float((ws * p_out).sum() / denom)


def eval_out_rate_prob_points(xs, ys, ws, rows_by_pos, opt_cfg, h_img, alpha):
    """
    xs, ys, ws : 評価対象点（bootstrap後）
    rows_by_pos: {pos: row}
    opt_cfg    : {"SS_dx":..., "SS_dy":..., ...}
    """
    probs_list = []

    for pos, row in rows_by_pos.items():
        dx = float(opt_cfg.get(f"{pos}_dx", 0.0))
        dy = float(opt_cfg.get(f"{pos}_dy", 0.0))

        e = make_ellipse_params(pos, row, h_img, dx_extra=dx, dy_extra=dy)

        u, v, c, s = precompute_uv(xs, ys, e["cx"], e["cy"], e["theta"])

        p = out_prob_from_uv(u, v, e["a"], e["b"], du=0.0, dv=0.0, alpha=alpha)
        probs_list.append(p)

    p_total = combine_out_probs(probs_list)
    return out_rate_weighted_prob(p_total, ws)


def eval_out_rate_hard_points(xs, ys, ws, rows_by_pos, cfg, h_img):
    n = len(xs)
    out_mask = np.zeros(n, dtype=bool)

    for pos in ["SS", "2B", "3B", "1B"]:
        if pos not in rows_by_pos:
            continue

        dx = float(cfg.get(f"{pos}_dx", 0.0))
        dy = float(cfg.get(f"{pos}_dy", 0.0))

        e = make_ellipse_params(pos, rows_by_pos[pos], h_img, dx_extra=dx, dy_extra=dy)
        u, v, c, s = precompute_uv(xs, ys, e["cx"], e["cy"], e["theta"])
        out_mask |= ellipse_mask_from_uv(u, v, e["a"], e["b"], du=0.0, dv=0.0)

    return out_rate_weighted_mask(out_mask, ws)


# =========================
# UI
# =========================
st.title("Pitcher Heatmap + Defensive Range (Filters)")

df = load_main_data()
ellipse_tables = load_ellipse_tables()

# 投手名（全投手オプション付き）
pitchers = sorted(df["pitchername"].dropna().unique().tolist())
ALL_PITCHERS_LABEL = "（全投手）"

pitcher = st.selectbox(
    "投手名を選択",
    options=[ALL_PITCHERS_LABEL] + pitchers,
    index=0,
)

# フィルタ候補を投手で絞った範囲から作る
if pitcher == ALL_PITCHERS_LABEL:
    df_p = df.copy()
else:
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
sel_type_group = st.multiselect(
    "pitch_type（球種グループ）",
    options=["ストレート系", "スライダー系", "フォーク系"],
    default=[],
)
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
if sel_type_group:
    df_f = df_f[df_f["pitch_type_group"].isin(sel_type_group)]
if sel_lr:
    df_f = df_f[df_f["player_batLR"].astype(str).isin(sel_lr)]

st.subheader("フィルタ後データ内の構成比（ゴロ以外も含む）")

col1, col2 = st.columns(2)
with col1:
    show_dist(df_f, "pitch_course", "コース")
    show_dist(df_f, "pitch_height", "高さ")
with col2:
    show_dist(df_f, "pitch_type_group", "球種グループ")
    show_dist(df_f, "player_batLR", "打者左右")

st.caption(
    f"フィルタ後の打球データ件数：{len(df_f)}（この中からゴロのみ抽出してヒートマップ化します）"
)


# =========================
# 描画
# =========================

img = mpimg.imread(IMAGE_PATH)
h_img, w_img = img.shape[0], img.shape[1]
img_flipped = np.flipud(img)

# --- home / 境界点を math 座標へ（origin="lower" 描画に合わせる） ---
home_math = (home[0], h_img - home[1])

left_pt_math = (left_pt[0], h_img - left_pt[1])
right_pt_math = (right_pt[0], h_img - right_pt[1])

ray_origin_near_math = (ray_origin_near[0], h_img - ray_origin_near[1])
left_pt_near_math = (left_pt_near[0], h_img - left_pt_near[1])
right_pt_near_math = (right_pt_near[0], h_img - right_pt_near[1])

home_math = (home[0], h_img - home[1])  # homeは画像座標(y下)なのでmathに変換

# =========================
# フェーズ3：条件付き分布モデル（学習）
# =========================
use_ai = st.checkbox("AI生成ゴロでデータ不足を補完（フェーズ3）", value=False)

n_gen = st.slider("生成するゴロ点数（不足分の目安）", 0, 500, 150, 25)
gen_weight = st.slider("生成点の重み（実測=1.0）", 0.1, 1.0, 0.3, 0.05)
gen_seed = st.number_input("生成seed（再現用）", value=0, step=1)

# 学習に使う特徴量（ここが重要：pitch_type_group を使う）
feat_cols = [
    "opponents",
    "pitch_course",
    "pitch_height",
    "pitch_type_group",
    "player_batLR",
]

df_ground_train = build_training_ground(
    df, w_img, h_img, use_projection_for_train=False
)

model = None
if use_ai and (df_ground_train is not None) and (len(df_ground_train) > 0):
    model = fit_conditional_hist(
        df_ground_train,
        home_math=home_math,
        feat_cols=feat_cols,
        n_theta=36,
        n_r=18,
        laplace=1.0,
    )

res = build_heatmap(df_f, img, w_img, h_img, use_projection=use_projection)

if res is None:
    st.warning("フィルタ後のゴロデータが0件です。条件を緩めてください。")
else:
    opt_result = None
    opt_delta = None
    opt_best_out = None
    counts_masked, xedges, yedges, norm, n_ground, df_ground = res

    # =========================
    # フェーズ3：生成点を評価に混ぜる（重み付き）
    # （※このブロックは1回だけ置く）
    # =========================
    xs_obs = df_ground["x_math"].to_numpy(float)
    ys_obs = df_ground["y_math"].to_numpy(float)
    ws_obs = np.ones(xs_obs.size, dtype=float)

    xs_all = xs_obs
    ys_all = ys_obs
    ws_all = ws_obs

    used_key = None

    if use_ai and (model is not None) and (n_gen > 0):
        cond_dict = {
            "opponents": pick_one_or_blank(sel_opponents),  # "京大"/"京大以外"
            "pitch_course": pick_one_or_blank(sel_course),
            "pitch_height": pick_one_or_blank(sel_height),
            "pitch_type_group": pick_one_or_blank(sel_type_group),
            "player_batLR": pick_one_or_blank(sel_lr),
        }

        xg, yg, used_key = generate_ground_xy_filtered(
            model=model,
            cond_dict=cond_dict,
            home_math=home_math,
            n_samples=n_gen,
            seed=gen_seed,
            w_img=w_img,
            h_img=h_img,
            left_pt_math=(left_pt[0], h_img - left_pt[1]),
            right_pt_math=(right_pt[0], h_img - right_pt[1]),
            ray_origin_near_math=(ray_origin_near[0], h_img - ray_origin_near[1]),
            left_pt_near_math=(left_pt_near[0], h_img - left_pt_near[1]),
            right_pt_near_math=(right_pt_near[0], h_img - right_pt_near[1]),
            oversample=4,
            max_rounds=10,
            # 投影を使うならここもON
            use_projection=use_projection,
            h_img_for_projection=h_img,
            home_img=home,
            ray_origin_img=ray_origin,
            left_pt_img=left_pt,
            right_pt_img=right_pt,
        )

        if xg is not None:
            gw = float(gen_weight)
            xs_all = np.concatenate([xs_obs, xg])
            ys_all = np.concatenate([ys_obs, yg])
            ws_all = np.concatenate([ws_obs, np.full(len(xg), gw, dtype=float)])

        st.caption(
            f"AI生成点 {n_gen}点（重み={gw}）を追加 / key='{used_key if used_key else 'N/A'}'"
        )
    st.caption(f"ヒートマップに使ったゴロ数：{n_ground}")

    fig, ax = plt.subplots(figsize=(6, 8))

    show_gen_points = st.checkbox(
        "AI生成点をヒートマップに重ねる（薄く表示）", value=True
    )
    gen_point_alpha = st.slider("生成点の透明度", 0.01, 0.5, 0.08, 0.01)
    gen_point_size = st.slider("生成点の大きさ", 1, 20, 6, 1)

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

    # =========================
    # AI生成点をヒートマップに重ねる（薄い点）
    # =========================
    if show_gen_points and use_ai and (model is not None) and (n_gen > 0):
        # 生成点がこのスコープに無い場合に備えて安全に取得
        if (
            "xg" in locals()
            and "yg" in locals()
            and (xg is not None)
            and (yg is not None)
        ):
            x_plot_gen = xg
            y_plot_gen = yg
        else:
            # 万一 xg/yg を消してしまった場合の保険：ここで再生成
            cond_dict = {
                "opponents": pick_one_or_blank(sel_opponents),
                "pitch_course": pick_one_or_blank(sel_course),
                "pitch_height": pick_one_or_blank(sel_height),
                "pitch_type_group": pick_one_or_blank(sel_type_group),
                "player_batLR": pick_one_or_blank(sel_lr),
            }
            x_plot_gen, y_plot_gen, _ = generate_ground_xy_from_model(
                model, cond_dict, home_math, n_samples=n_gen, seed=gen_seed
            )

        ax.scatter(
            x_plot_gen,
            y_plot_gen,
            s=gen_point_size,
            alpha=gen_point_alpha,
            marker="o",
            linewidths=0,
            rasterized=True,  # ★軽くなる（重要）
            label="AI generated",
        )

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

    has_fielders = len(rows_by_pos) > 0
    if not has_fielders:
        st.info("守備範囲（SS/2B/3B/1B）を選ぶと、Out%と最適化が表示されます。")
    else:
        # ==================================
        # 評価関数を統一（通常も最適もここを使う）
        # ==================================

        xs = xs_all
        ys = ys_all
        ws = ws_all
        n = len(xs)

        use_prob = st.checkbox("Out%を確率モデルで評価（中心0.9/縁0.5）", value=True)
        alpha_prob = st.slider(
            "確率の鋭さ alpha（大きいほど0/1に近づく）", 0.5, 8.0, 2.2, 0.1
        )

        # 事前計算（確率用）: u,v,c,s は毎回作らない
        packs_all = {}
        for pos in ["SS", "2B", "3B", "1B"]:
            if pos not in rows_by_pos:
                continue
            e0 = make_ellipse_params(
                pos, rows_by_pos[pos], h_img, dx_extra=0, dy_extra=0
            )
            u0, v0, c0, s0 = precompute_uv(xs, ys, e0["cx"], e0["cy"], e0["theta"])
            packs_all[pos] = (e0, u0, v0, c0, s0)

        def eval_out_rate_prob(cfg):
            probs_list = []
            for pos, (e0, u, v, c, s) in packs_all.items():
                dx = float(cfg.get(f"{pos}_dx", 0.0))
                dy = float(cfg.get(f"{pos}_dy", 0.0))
                du, dv = shift_to_du_dv(dx, dy, c, s)
                p = out_prob_from_uv(
                    u, v, e0["a"], e0["b"], du=du, dv=dv, alpha=alpha_prob
                )
                probs_list.append(p)

            p_total = combine_out_probs(probs_list)  # ← n渡す版にしておく
            return out_rate_weighted_prob(p_total, ws)

        def eval_out_rate_hard(cfg):
            return eval_out_rate_hard_points(xs, ys, ws, rows_by_pos, cfg, h_img)

        def eval_out(cfg):
            return eval_out_rate_prob(cfg) if use_prob else eval_out_rate_hard(cfg)

        # =========================
        # 通常守備 Out%（重み付き）
        # =========================
        # ---------------------------
        # データ点（math座標）
        # ---------------------------
        # ここを生成込みに差し替え

        sum_w = ws.sum()

        out_normal = eval_out({})
        st.write(f"通常守備 Out%（モデル）: **{out_normal:.3f}**")
        st.write(f"通常守備 BA換算（≒1-Out%）: **{(1-out_normal):.3f}**")

        # 探索設定（軽め）

        do_opt = st.checkbox(
            "SS・2B・3B・1Bの位置を動かして最適Out%を探す（簡易グリッド探索）",
            value=False,
        )

        if do_opt:
            max_shift = st.slider(
                "探索幅（±px）", min_value=0, max_value=100, value=20, step=5
            )
            step = st.slider("刻み（px）", min_value=1, max_value=20, value=10, step=1)

            move_ss = st.checkbox("SSを動かす", value=True)
            move_2b = st.checkbox("2Bを動かす", value=True)
            move_3b = st.checkbox("3Bを動かす", value=True)
            move_1b = st.checkbox("1Bを動かす", value=True)

            move_flags = {"SS": move_ss, "2B": move_2b, "3B": move_3b, "1B": move_1b}

            missing = [
                p
                for p in ["SS", "2B", "3B", "1B"]
                if move_flags.get(p, False) and (p not in rows_by_pos)
            ]
            if missing:
                st.warning(f"探索するポジションの選手が未選択です: {missing}")
            else:

                if n == 0:
                    st.warning("ゴロデータが0件です。")
                else:
                    shifts = list(range(-max_shift, max_shift + 1, step))

                    # 動かすポジション（選択されている & 選手が選ばれているものだけ）
                    move_positions = []
                    for pos in ["SS", "2B", "3B", "1B"]:
                        if move_flags.get(pos, False) and (pos in rows_by_pos):
                            move_positions.append(pos)

                    # 1つも動かさないなら終了
                    if len(move_positions) == 0:
                        st.warning(
                            "動かすポジションがありません（選手未選択 or チェックOFF）。"
                        )
                    else:
                        # ---------------------------
                        # 固定ポジ（動かさないポジ）のmaskを先に作る
                        # ---------------------------
                        fixed_mask = np.zeros(n, dtype=bool)

                        for pos in ["SS", "2B", "3B", "1B"]:
                            if pos not in rows_by_pos:
                                continue
                            if pos in move_positions:
                                continue  # 動かすので固定には入れない

                            e = make_ellipse_params(
                                pos, rows_by_pos[pos], h_img, dx_extra=0, dy_extra=0
                            )
                            u, v, c, s = precompute_uv(
                                xs, ys, e["cx"], e["cy"], e["theta"]
                            )
                            fixed_mask |= ellipse_mask_from_uv(
                                u, v, e["a"], e["b"], du=0.0, dv=0.0
                            )

                        # ---------------------------
                        # 座標降下：1ポジずつ最適にしていく（軽い）
                        # ---------------------------
                        n_iters = st.slider("最適化の反復回数（軽い）", 1, 8, 3, 1)

                        cur0 = {}
                        for pos in move_positions:
                            cur0[f"{pos}_dx"] = 0
                            cur0[f"{pos}_dy"] = 0

                        out_normal_eval = eval_out(cur0)
                        best_out = eval_out(cur0)
                        cur = dict(cur0)

                        for it in range(n_iters):
                            improved = False

                            for pos in move_positions:
                                best_local = best_out
                                best_dx = cur[f"{pos}_dx"]
                                best_dy = cur[f"{pos}_dy"]

                                # このposだけ(dx,dy)を総当たり（他posは固定）
                                for dx in shifts:
                                    for dy in shifts:
                                        # 一時的に上書き
                                        cur[f"{pos}_dx"] = int(dx)
                                        cur[f"{pos}_dy"] = int(dy)

                                        out_rate = eval_out(cur)

                                        if out_rate > best_local:
                                            best_local = out_rate
                                            best_dx, best_dy = int(dx), int(dy)

                                # 探索が終わったら、見つかった最良値をセット
                                cur[f"{pos}_dx"] = best_dx
                                cur[f"{pos}_dy"] = best_dy

                                if best_local > best_out:
                                    best_out = best_local
                                    improved = True

                            if not improved:
                                break

                        opt_best_out = best_out
                        opt_result = dict(cur)
                        opt_delta = opt_best_out - out_normal_eval

                        st.write(f"最適Out%（探索）: **{opt_best_out:.3f}**")
                        st.write(f"ΔOut%（最適−通常）: **{opt_delta:+.3f}**")
                        st.caption(f"最適シフト（追加移動量, px）: {opt_result}")

                    # =========================
                    # ΔOut% の不確実性（軽量ブートストラップ：best_cfg固定）
                    # =========================
                    st.subheader("ΔOut% の不確実性（ブートストラップ）※軽量版")

                    do_ci = st.checkbox(
                        "ΔOut%の信頼区間を出す（best_cfg固定）", value=False
                    )

                    if do_ci:
                        B = st.slider(
                            "反復回数（多いほど安定・重くなる）", 20, 400, 100, 10
                        )
                        seed = st.number_input("乱数seed（再現用）", value=0, step=1)

                        # ★CIは「評価に使っている点群」と同じものを使う（生成点・重みも含む）
                        xs_ci = np.asarray(xs, dtype=float)
                        ys_ci = np.asarray(ys, dtype=float)
                        ws_ci = np.asarray(ws, dtype=float)
                        n_all = xs_ci.size

                        if (opt_result is None) or (len(opt_result) == 0):
                            st.warning(
                                "最適配置(opt_result)が未計算です。先に探索(do_opt)をONにしてください。"
                            )
                        elif n_all < 5:
                            st.warning("ゴロ数が少なすぎて信頼区間が安定しません。")
                        else:
                            # ===== 通常（ゼロシフト）cfg =====
                            cfg_normal = {}
                            for pos in rows_by_pos.keys():
                                cfg_normal[f"{pos}_dx"] = 0.0
                                cfg_normal[f"{pos}_dy"] = 0.0

                            # ===== 点推定（同じ点群上で）=====
                            out_n0 = eval_out_rate_prob_points(
                                xs_ci,
                                ys_ci,
                                ws_ci,
                                rows_by_pos,
                                cfg_normal,
                                h_img,
                                alpha_prob,
                            )
                            out_s0 = eval_out_rate_prob_points(
                                xs_ci,
                                ys_ci,
                                ws_ci,
                                rows_by_pos,
                                opt_result,
                                h_img,
                                alpha_prob,
                            )
                            delta0 = out_s0 - out_n0

                            # ===== paired bootstrap（同じidxで通常とシフトを両方評価）=====
                            rng = np.random.default_rng(int(seed))
                            deltas = np.empty(B, dtype=float)
                            outs_n = np.empty(B, dtype=float)
                            outs_s = np.empty(B, dtype=float)

                            for i in range(B):
                                idx = rng.integers(0, n_all, size=n_all)

                                xs_b = xs_ci[idx]
                                ys_b = ys_ci[idx]
                                ws_b = ws_ci[idx]

                                out_n = eval_out_rate_prob_points(
                                    xs_b,
                                    ys_b,
                                    ws_b,
                                    rows_by_pos,
                                    cfg_normal,
                                    h_img,
                                    alpha_prob,
                                )
                                out_s = eval_out_rate_prob_points(
                                    xs_b,
                                    ys_b,
                                    ws_b,
                                    rows_by_pos,
                                    opt_result,
                                    h_img,
                                    alpha_prob,
                                )

                                outs_n[i] = out_n
                                outs_s[i] = out_s
                                deltas[i] = out_s - out_n

                            lo, hi = np.percentile(deltas, [2.5, 97.5])
                            n_lo, n_hi = np.percentile(outs_n, [2.5, 97.5])
                            s_lo, s_hi = np.percentile(outs_s, [2.5, 97.5])

                            st.write(f"通常 Out%（確率, 点推定）: **{out_n0:.3f}**")
                            st.write(f"シフト Out%（確率, 点推定）: **{out_s0:.3f}**")
                            st.write(f"ΔOut%（点推定）: **{delta0:+.3f}**")

                            st.write(
                                f"通常 Out% 95%CI（bootstrap）: **[{n_lo:.3f}, {n_hi:.3f}]**"
                            )
                            st.write(
                                f"シフト Out% 95%CI（bootstrap）: **[{s_lo:.3f}, {s_hi:.3f}]**"
                            )
                            st.write(
                                f"ΔOut% 95%CI（paired bootstrap）: **[{lo:+.3f}, {hi:+.3f}]**"
                            )

                            st.caption(
                                "※paired bootstrap：同じ再標本サンプル上で通常とシフトを比較するので、差(Δ)の解釈が自然になります。"
                            )

        # ……（ここまでで背景・ヒートマップ・通常楕円は描画済み）

        # =========================
        # 最適配置のオーバーレイ（探索したときだけ）
        # =========================
        show_opt = st.checkbox("最適配置（探索結果）を図に重ねる", value=True)

        if show_opt and (opt_result is not None):
            for pos in ["SS", "2B", "3B", "1B"]:
                pick = selected_player[pos]
                if pick == "（表示しない）":
                    continue
                if f"{pos}_dx" not in opt_result:
                    continue

                t = ellipse_tables[pos]
                row = t[t["NAME"].astype(str) == pick].iloc[0]

                dx = float(opt_result.get(f"{pos}_dx", 0))
                dy = float(opt_result.get(f"{pos}_dy", 0))

                # 最適位置の点
                base_x_img, base_y_img = infield_positions_img[pos]
                cx_img = base_x_img + float(row["center_x"]) + dx
                cy_img = base_y_img + float(row["center_y"]) + dy
                cx = cx_img
                cy = h_img - cy_img
                ax.scatter(cx, cy, s=140, marker="x")

                # 破線の楕円
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
