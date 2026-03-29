import streamlit as st
import numpy as np
import plotly.graph_objects as go
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

# ================= 1. UI 配置与参数 =================
st.set_page_config(page_title="艏部双平面相切干涉分析", layout="wide")
st.title("🛡️ 艏部刚体正撞蜂窝工程干涉分析 (V6.14 纯净视图版)")

with st.sidebar:
    st.header("📐 1. 基础物理尺寸")
    LB = st.number_input("船舶总长 LB (m)", value=90.0, step=1.0)
    BM = st.number_input("船艏型宽 BM (m)", value=14.8, step=0.1)
    DL = st.number_input("吃水深度 DL (m)", value=2.6, step=0.1)
    BT = st.number_input("迎水面定宽 BT (m)", value=4.5, step=0.1, disabled=True)

    st.markdown("---")
    st.header("🎛️ 2. 无量纲比例系数")
    alpha = st.number_input("α = RL / LB", value=0.1555556, step=0.0000001, format="%.7f")
    beta = st.number_input("β = HL / DL", value=0.242, step=0.001, format="%.3f")
    gamma = st.number_input("γ = DB / DL", value=1.600, step=0.001, format="%.3f")

    st.markdown("---")
    st.header("🚀 3. 几何过渡控制")
    RL = alpha * LB
    ZT = st.number_input("过渡曲线段纵深 ZT (m)", value=RL * 0.8, min_value=0.1, max_value=float(RL), step=0.1)
    n_power = st.number_input("双平面曲线饱满度 n (指数)", min_value=1.1, max_value=5.0, value=2.0, step=0.1)

    st.markdown("---")
    st.header("🕸️ 4. 仿真环境控制")
    hex_side = st.number_input("蜂窝边长 (m)", value=0.5, step=0.05)

    st.markdown("---")
    st.markdown("### 🔥 **核心工况输入**")
    z_d = st.number_input(f"精确侵入深度 Z (0 ~ {RL:.3f}m)", min_value=0.0, max_value=float(RL), value=RL / 2,
                          step=0.05, format="%.3f")

# --- 核心尺寸衍生 ---
HL = beta * DL
DB = gamma * DL
bottom_half_w = (0.75 * BM) / 2


# ================= 2. 几何核心逻辑 =================

def get_contour_pts(z, BT, BM, DB, HL, ZT, n_power):
    if z <= ZT:
        pct = 1.0 - (1.0 - z / ZT) ** n_power
        y_scaled = (-HL) - (DB - HL) * pct
        xt_scaled = BT / 2 + ((BM - BT) / 2) * pct
        xb_scaled = BT / 2 + (bottom_half_w - BT / 2) * pct
    else:
        y_scaled = -DB
        xt_scaled = BM / 2
        xb_scaled = bottom_half_w

    pts = []
    pts.extend([[-xt_scaled, 0], [xt_scaled, 0]])
    pts.append([xt_scaled, -(DB - (DB - HL))])

    for v in np.linspace(0, 1, 12):
        pts.append([xt_scaled - (xt_scaled - xb_scaled) * (v ** 4),
                    ((DB - HL) - ((DB - HL) - (y_scaled + DB)) * v) - DB])
    pts.extend([[-xb_scaled, y_scaled], [-xb_scaled, y_scaled]])

    for v in np.linspace(1, 0, 12):
        pts.append([-(xt_scaled - (xt_scaled - xb_scaled) * (v ** 4)),
                    ((DB - HL) - ((DB - HL) - (y_scaled + DB)) * v) - DB])
    pts.append([-xt_scaled, -(DB - (DB - HL))])
    pts.append([-xt_scaled, 0])

    pts_arr = np.array(pts)
    pts_arr[:, 1] -= 0.001
    return pts_arr


# 💥 彻底抛弃了俯视图的假蜂窝代码，现在只算XY截面，速度更快！
@st.cache_data(show_spinner=False)
def get_static_backgrounds(side):
    h = side * np.sqrt(3)
    shapely_lines = []
    all_x, all_y = [], []

    for c in range(-25, 25):
        for r in range(-20, 20):
            cx, cy = c * 1.5 * side, r * h + (c % 2) * (h / 2)
            v = []
            for i in range(6):
                angle = np.deg2rad(i * 60)
                v.append([np.round(cx + side * np.cos(angle), 5), np.round(cy + side * np.sin(angle), 5)])
            for i in range(6):
                p1, p2 = v[i], v[(i + 1) % 6]
                shapely_lines.append(LineString([p1, p2]))
                all_x.extend([p1[0], p2[0], None])
                all_y.extend([p1[1], p2[1], None])

    return unary_union(shapely_lines), all_x, all_y


h_shapely, all_x, all_y = get_static_backgrounds(hex_side)

current_contour = get_contour_pts(z_d, BT, BM, DB, HL, ZT, n_power)

# 1. 周界交点计算
contour_line = LineString(current_contour)
tree = STRtree(h_shapely.geoms if hasattr(h_shapely, 'geoms') else [h_shapely])
relevant_indices = tree.query(contour_line)
inter_coords = []
for i in relevant_indices:
    inter = contour_line.intersection(h_shapely.geoms[i] if hasattr(h_shapely, 'geoms') else h_shapely)
    if not inter.is_empty:
        if inter.geom_type == 'Point':
            inter_coords.append((inter.x, inter.y))
        elif inter.geom_type == 'MultiPoint':
            for p in inter.geoms: inter_coords.append((p.x, p.y))
        elif inter.geom_type == 'LineString':
            inter_coords.extend(list(inter.coords))

if inter_coords:
    inter_coords = np.unique(np.array(inter_coords), axis=0)
else:
    inter_coords = np.array([])
n_count = len(inter_coords)

# 2. 截面积精确计算
contour_poly = Polygon(current_contour)
current_area = contour_poly.area

# ================= 3. UI 布局与视图生成 =================

col_l, col_r = st.columns([1.2, 1])

with col_l:
    st.subheader("🌐 3D 艏部复合几何预览")
    fig_3d = go.Figure()

    zs_3d = np.linspace(0, RL, 30)
    for i in range(len(zs_3d) - 1):
        c1 = get_contour_pts(zs_3d[i], BT, BM, DB, HL, ZT, n_power)
        c2 = get_contour_pts(zs_3d[i + 1], BT, BM, DB, HL, ZT, n_power)
        color = 'rgba(255, 0, 0, 0.6)' if zs_3d[i] < z_d else 'rgba(0, 0, 139, 0.8)'
        fig_3d.add_trace(go.Surface(
            x=[c1[:, 0], c2[:, 0]], y=[c1[:, 1], c2[:, 1]], z=[[zs_3d[i]] * len(c1), [zs_3d[i + 1]] * len(c2)],
            colorscale=[[0, color], [1, color]], showscale=False
        ))

    # 3D虚拟障碍面保留，起指示作用
    fig_3d.add_trace(go.Surface(
        x=[[-BM / 2 - 1, BM / 2 + 1], [-BM / 2 - 1, BM / 2 + 1]], y=[[-DB - 1, -DB - 1], [1, 1]],
        z=[[z_d, z_d], [z_d, z_d]],
        colorscale=[[0, 'rgba(255, 0, 0, 0.2)'], [1, 'rgba(255, 0, 0, 0.2)']], showscale=False, name='蜂窝障碍面'
    ))

    fig_3d.update_layout(scene=dict(aspectmode='data', camera=dict(projection=dict(type='orthographic')),
                                    xaxis_title="X (宽)", yaxis_title="Y (高)", zaxis_title="Z (深)"), height=450,
                         margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig_3d, use_container_width=True)
    st.markdown("---")

    # 💥 俯视图回归纯净！
    st.subheader("🛰️ 俯视图 (ZX平面)")
    fig_top = go.Figure()

    # 画一个代表蜂窝障碍块的浅灰色线框，表示它的物理边界
    fig_top.add_trace(go.Scatter(
        x=[0, 0, -RL - 2, -RL - 2, 0],
        y=[BM / 2 + 1, -BM / 2 - 1, -BM / 2 - 1, BM / 2 + 1, BM / 2 + 1],
        mode='lines', fill='toself', fillcolor='rgba(150,150,150,0.1)', line=dict(color='gray', width=1, dash='dot'),
        name='蜂窝区边界'
    ))

    z_mesh = np.linspace(0, RL, 60)
    x_half = []
    for z in z_mesh:
        if z <= ZT:
            x_half.append(BT / 2 + ((BM - BT) / 2) * (1.0 - (1.0 - z / ZT) ** n_power))
        else:
            x_half.append(BM / 2)

    plot_z = [-z for z in z_mesh] + [-z for z in reversed(z_mesh)] + [0]
    plot_x = x_half + [-x for x in reversed(x_half)] + [x_half[0]]
    fig_top.add_trace(go.Scatter(x=plot_z, y=plot_x, mode='lines', fill='toself', fillcolor='rgba(0,0,139,0.3)',
                                 line=dict(color='blue', width=3), name='完整艏部'))

    if z_d > 0:
        z_crush_mesh = np.linspace(0, z_d, 25)
        x_crush = []
        for z in z_crush_mesh:
            if z <= ZT:
                x_crush.append(BT / 2 + ((BM - BT) / 2) * (1.0 - (1.0 - z / ZT) ** n_power))
            else:
                x_crush.append(BM / 2)
        cz = [-z for z in z_crush_mesh] + [-z for z in reversed(z_crush_mesh)] + [0]
        cx = x_crush + [-x for x in reversed(x_crush)] + [x_crush[0]]
        fig_top.add_trace(go.Scatter(x=cz, y=cx, mode='lines', fill='toself', fillcolor='rgba(255,0,0,0.5)',
                                     line=dict(color='red', width=2), name='已压溃区'))

    fig_top.add_trace(go.Scatter(x=[-z_d, -z_d], y=[-BM / 2 - 1, BM / 2 + 1], mode='lines',
                                 line=dict(color='red', width=3, dash='dash')))
    fig_top.add_trace(go.Scatter(x=[-ZT, -ZT], y=[-BM / 2 - 1, BM / 2 + 1], mode='lines',
                                 line=dict(color='green', width=2, dash='dot')))

    fig_top.update_layout(xaxis=dict(title="Z (纵深 m)", range=[-RL - 1, 1]),
                          yaxis=dict(title="X (宽度 m)", range=[-BM / 2 - 1, BM / 2 + 1], scaleanchor="x",
                                     scaleratio=1), height=400, margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig_top, use_container_width=True)

with col_r:
    st.subheader("📐 侧视图 (ZY平面)")
    fig_s = go.Figure()

    # 侧视图也加一个灰色框表示障碍块位置
    fig_s.add_trace(go.Scatter(
        x=[0, 0, RL + 2, RL + 2, 0],
        y=[0.5, -DB - 0.5, -DB - 0.5, 0.5, 0.5],
        mode='lines', fill='toself', fillcolor='rgba(150,150,150,0.1)', line=dict(color='gray', width=1, dash='dot'),
        name='蜂窝区边界'
    ))

    zs_s = np.linspace(0, RL, 60)
    ys_s = []
    for z in zs_s:
        if z <= ZT:
            ys_s.append((-HL) - (DB - HL) * (1.0 - (1.0 - z / ZT) ** n_power))
        else:
            ys_s.append(-DB)

    s_x = [0, RL, RL] + list(np.flip(zs_s)) + [0]
    total_y = [0, 0, -DB] + list(np.flip(ys_s)) + [0]
    fig_s.add_trace(go.Scatter(x=s_x, y=total_y, mode='lines', fill='toself', fillcolor='rgba(0,0,139,0.1)',
                               line=dict(color='rgba(0,0,0,0)'), showlegend=False))

    fig_s.add_trace(go.Scatter(x=[0, RL], y=[0, 0], mode='lines', line=dict(color='blue', width=2), name='甲板线'))
    fig_s.add_trace(go.Scatter(x=[0, 0], y=[0, -HL], mode='lines', line=dict(color='blue', width=2), showlegend=False))
    fig_s.add_trace(go.Scatter(x=zs_s, y=ys_s, mode='lines', line=dict(color='#00008B', width=4), name='船底剖面线'))

    if z_d > 0:
        crush_zs = np.linspace(0, z_d, 25)
        crush_ys = []
        for z in crush_zs:
            if z <= ZT:
                crush_ys.append((-HL) - (DB - HL) * (1.0 - (1.0 - z / ZT) ** n_power))
            else:
                crush_ys.append(-DB)
        fig_s.add_trace(
            go.Scatter(x=crush_zs, y=crush_ys, mode='lines', line=dict(color='red', width=4), name='已压溃剖面'))

    fig_s.add_trace(
        go.Scatter(x=[z_d, z_d], y=[0.5, -DB - 0.5], mode='lines', line=dict(color='red', width=3, dash='dash')))
    fig_s.add_trace(
        go.Scatter(x=[ZT, ZT], y=[0.5, -DB - 0.5], mode='lines', line=dict(color='green', width=2, dash='dot')))

    fig_s.update_layout(xaxis=dict(title="Z (深 m)", range=[-1, RL + 1]),
                        yaxis=dict(title="Y (高 m)", range=[-DB - 1, 1.5], scaleanchor="x", scaleratio=1), height=400,
                        margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig_s, use_container_width=True)

    st.subheader("🎯 正截面干涉分析视图 (Z={:.3f}m处)".format(z_d))
    fig_xy = go.Figure()
    fig_xy.add_trace(
        go.Scattergl(x=all_x, y=all_y, mode='lines', line=dict(color='#888888', width=1), showlegend=False))
    fig_xy.add_trace(
        go.Scatter(x=current_contour[:, 0], y=current_contour[:, 1], mode='lines', line=dict(color='#00008B', width=4),
                   fill='toself', fillcolor='rgba(0,0,139,0.1)', name='当前截面'))
    if n_count > 0: fig_xy.add_trace(go.Scattergl(x=inter_coords[:, 0], y=inter_coords[:, 1], mode='markers',
                                                  marker=dict(color='red', size=10, line=dict(color='white', width=1)),
                                                  name='交点'))
    fig_xy.add_trace(
        go.Scatter(x=[0], y=[0], mode='markers', marker=dict(color='#32CD32', size=15, symbol='cross'), name='中心点'))
    fig_xy.update_layout(xaxis=dict(range=[-BM / 2 - 1, BM / 2 + 1], scaleanchor="y"), yaxis=dict(range=[-DB - 1, 1.5]),
                         height=500)
    st.plotly_chart(fig_xy, use_container_width=True)

# ================= 5. 数据仪表盘 (全新分组排版) =================
st.markdown("---")
st.markdown("### 📊 实时数据监控")

c1, c2, c3, c4 = st.columns(4)
c1.metric("📐 船艏长度 RL", f"{RL:.3f} m")
c2.metric("📐 船头平面高 HL", f"{HL:.3f} m")
c3.metric("📐 吃水深度 DL", f"{DL:.3f} m")
c4.metric("📐 船艏总高度 DB", f"{DB:.3f} m")

st.markdown("<br>", unsafe_allow_html=True)

c5, c6 = st.columns(2)
c5.info(f"#### 📦 当前截面积 Area:  **{current_area:.3f}** m²")
c6.error(f"#### 🔥 精确交点数 N:  **{n_count}**")