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
    return counts_masked, xedges, yedges, norm, len(df_ground)


def add_ellipse(ax, pos, row, h_img):
    base_x_img, base_y_img = infield_positions_img[pos]
    angle = ROT_DEG[pos]

    name = str(row["NAME"])
    dx = float(row["center_x"])
    dy = float(row["center_y"])
    a = float(row["major-axis"])
    b = float(row["minor-axis"])

    cx_img = base_x_img + dx
    cy_img = base_y_img + dy

    cx = cx_img
    cy = h_img - cy_img

    e = Ellipse(
        xy=(cx, cy), width=2 * a, height=2 * b, angle=angle, fill=False, linewidth=2
    )
    ax.add_patch(e)
    ax.text(cx + 6, cy + 6, name, fontsize=9)


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
    counts_masked, xedges, yedges, norm, n_ground = res
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

    ax.set_xlim(0, w_img)
    ax.set_ylim(0, h_img)

    proj_text = "Projected" if use_projection else "No Projection"
    ax.set_title(f"Heatmap + Ranges | {pitcher} | {proj_text}")

    ax.set_xlabel("x")
    ax.set_ylabel("y (math coords: up is +)")

    st.pyplot(fig)
