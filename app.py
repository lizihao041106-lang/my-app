import streamlit as st
import numpy as np
import plotly.graph_objects as go
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

# ================= 1. UI 配置与参数 =================
st.set_page_config(page_title="艏部双平面相切干涉分析-极限版", layout="wide")
st.title("🛡️ 艏部刚体正撞蜂窝工程干涉分析 (V6.17 0.01m 极限精度版)")

with st.sidebar:
    st.header("📐 1. 基础物理尺寸")
    LB = st.number_input("船舶总长 LB (m)", value=90.0, step=1.0)
    BM = st.number_input("船艏型宽 BM (m)", value=14.8, step=0.1)
    DL = st.number_input("吃水深度 DL (m)", value=2.6, step=0.1)
    BT = st.number_input("迎水面定宽 BT (m)", value=4.5, min_value=0.1, max_value=float(BM-0.1), step=0.1)
    
    st.markdown("---")
    st.header("🎛️ 2. 无量纲比例系数")
    alpha = st.number_input("α = RL / LB", value=0.1555556, step=0.0000001, format="%.7f")
    beta  = st.number_input("β = HL / DL", value=0.242, step=0.001, format="%.3f")
    gamma = st.number_input("γ = DB / DL", value=1.600, step=0.001, format="%.3f")
    
    st.markdown("---")
    st.header("🚀 3. 几何过渡控制")
    RL = alpha * LB
    ZT = st.number_input("过渡曲线段纵深 ZT (m)", value=RL*0.8, min_value=0.1, max_value=float(RL), step=0.1)
    n_power = st.number_input("双平面曲线饱满度 n (指数)", min_value=1.1, max_value=5.0, value=2.0, step=0.1)

    st.markdown("---")
    st.header("🕸️ 4. 仿真环境控制")
    # 💥 极限开关：min_value 已解锁至 0.01m (1厘米)
    hex_side = st.number_input("蜂窝边长 (m)", value=0.5, min_value=0.01, step=0.01)
    
    st.markdown("---")
    st.markdown("### 🔥 **核心工况输入**")
    z_d = st.number_input(f"精确侵入深度 Z (0 ~ {RL:.3f}m)", min_value=0.0, max_value=float(RL), value=RL/2, step=0.05, format="%.3f")

# --- 核心尺寸衍生 ---
HL = beta * DL      
DB = gamma * DL     
bottom_half_w = (0.75 * BM) / 2  

# ================= 2. 几何核心逻辑 =================

def get_contour_pts(z, BT, BM, DB, HL, ZT, n_power):
    if z <= ZT:
        pct = 1.0 - (1.0 - z/ZT)**n_power
        y_scaled = (-HL) - (DB - HL) * pct
        xt_scaled = BT/2 + ((BM - BT)/2) * pct
        xb_scaled = BT/2 + (bottom_half_w - BT/2) * pct
    else:
        y_scaled = -DB
        xt_scaled = BM / 2
        xb_scaled = bottom_half_w

    pts = []
    pts.extend([[-xt_scaled, 0], [xt_scaled, 0]]) 
    pts.append([xt_scaled, -(DB - (DB - HL))])    
    
    for v in np.linspace(0, 1, 15): # 略微增加采样率以匹配高精度
        pts.append([xt_scaled - (xt_scaled - xb_scaled)*(v**4), 
                    ( (DB-HL) - ((DB-HL) - (y_scaled+DB))*v) - DB ])
    pts.extend([[-xb_scaled, y_scaled], [-xb_scaled, y_scaled]])
    
    for v in np.linspace(1, 0, 15): 
        pts.append([-(xt_scaled - (xt_scaled - xb_scaled)*(v**4)), 
                    ( (DB-HL) - ((DB-HL) - (y_scaled+DB))*v) - DB ])
    pts.append([-xt_scaled, -(DB - (DB - HL))])
    pts.append([-xt_scaled, 0])
    
    return np.array(pts)

@st.cache_data(show_spinner=False)
def get_dynamic_backgrounds(side, max_w, max_d):
    h = side * np.sqrt(3)
    sx, sy = 1.5 * side, h
    req_w, req_d = max_w * 1.2, max_d + 1.0
    
    num_cols = int(np.ceil(req_w / sx))
    c_min, c_max = - (num_cols // 2) - 1, (num_cols // 2) + 1
    num_rows = int(np.ceil(req_d / sy))
    r_offset = int(np.round(-(max_d / 2) / sy))
    r_min, r_max = r_offset - (num_rows // 2) - 1, r_offset + (num_rows // 2) + 1
    
    # 💥 针对 0.01m 的极致优化：预分配内存
    shapely_lines = []
    all_x, all_y = [], []
    
    for c in range(c_min, c_max + 1):
        for r in range(r_min, r_max + 1):
            cx, cy = c * sx, r * sy + (c % 2) * (h/2)
            v = []
            for i in range(6):
                angle = np.deg2rad(i * 60)
                v.append([np.round(cx + side * np.cos(angle), 5), np.round(cy + side * np.sin(angle), 5)])
            for i in range(6):
                p1, p2 = v[i], v[(i+1)%6]
                shapely_lines.append(LineString([p1, p2]))
                all_x.extend([p1[0], p2[0], None])
                all_y.extend([p1[1], p2[1], None])
                
    return unary_union(shapely_lines), all_x, all_y

# 背景生成
h_shapely, all_x, all_y = get_dynamic_backgrounds(hex_side, BM, DB)
current_contour = get_contour_pts(z_d, BT, BM, DB, HL, ZT, n_power)

# 1. 高速 STRtree 空间查询
contour_line = LineString(current_contour)
tree = STRtree(h_shapely.geoms if hasattr(h_shapely, 'geoms') else [h_shapely])
relevant_indices = tree.query(contour_line)
inter_coords = []
for i in relevant_indices:
    inter = contour_line.intersection(h_shapely.geoms[i] if hasattr(h_shapely, 'geoms') else h_shapely)
    if not inter.is_empty:
        if inter.geom_type == 'Point': inter_coords.append((inter.x, inter.y))
        elif inter.geom_type == 'MultiPoint':
            for p in inter.geoms: inter_coords.append((p.x, p.y))
        elif inter.geom_type == 'LineString': inter_coords.extend(list(inter.coords))

if inter_coords: inter_coords = np.unique(np.array(inter_coords), axis=0)
else: inter_coords = np.array([])
n_count = len(inter_coords)
current_area = Polygon(current_contour).area

# ================= 3. 渲染视图 (GPU 加速) =================

col_l, col_r = st.columns([1, 1])

with col_l:
    st.subheader("🌐 3D 预览 (低采样率以保持流畅)")
    fig_3d = go.Figure()
    zs_3d = np.linspace(0, RL, 20) # 💥 降低 3D 采样防止 WebGL 崩溃
    for i in range(len(zs_3d)-1):
        c1, c2 = get_contour_pts(zs_3d[i],BT,BM,DB,HL,ZT,n_power), get_contour_pts(zs_3d[i+1],BT,BM,DB,HL,ZT,n_power)
        color = 'rgba(255, 0, 0, 0.4)' if zs_3d[i] < z_d else 'rgba(0, 0, 139, 0.6)'
        fig_3d.add_trace(go.Surface(x=[c1[:,0], c2[:,0]], y=[c1[:,1], c2[:,1]], z=[[zs_3d[i]]*len(c1), [zs_3d[i+1]]*len(c2)], colorscale=[[0, color], [1, color]], showscale=False))
    fig_3d.update_layout(scene=dict(aspectmode='data', xaxis_title="X", yaxis_title="Y", zaxis_title="Z"), height=400, margin=dict(l=0,r=0,b=0,t=0))
    st.plotly_chart(fig_3d, use_container_width=True)

with col_r:
    st.subheader(f"🎯 正截面干涉分析 (Z={z_d:.3f}m)")
    fig_xy = go.Figure()
    
    # 💥 核心优化：使用 Scattergl 而不是 Scatter 以启用 GPU 渲染 30 万根线条
    fig_xy.add_trace(go.Scattergl(x=all_x, y=all_y, mode='lines', line=dict(color='#888888', width=0.5), showlegend=False))
    fig_xy.add_trace(go.Scatter(x=current_contour[:,0], y=current_contour[:,1], mode='lines', line=dict(color='#00008B', width=3), fill='toself', fillcolor='rgba(0,0,139,0.1)', name='当前截面'))
    
    # 动态红点大小
    m_size = max(1.5, min(8, int(hex_side * 20)))
    if n_count > 0: 
        fig_xy.add_trace(go.Scattergl(x=inter_coords[:,0], y=inter_coords[:,1], mode='markers', marker=dict(color='red', size=m_size), name='交点'))
        
    fig_xy.update_layout(
        xaxis=dict(range=[-BM/2-0.5, BM/2+0.5], scaleanchor="y", scaleratio=1), 
        yaxis=dict(range=[-DB-0.5, 1]), 
        height=600,
        hovermode=False, # 💥 极其重要：0.01m 下开启悬停感应会导致浏览器瞬间卡死
        dragmode='pan'   # 默认改为平移模式
    )
    st.plotly_chart(fig_xy, use_container_width=True)

# ================= 4. 数据看板 =================
st.markdown("---")
c1, c2, c3 = st.columns(3)
c1.metric("📦 当前截面积 Area", f"{current_area:.4f} m²")
c2.error(f"🔥 精确交点数 N: {n_count}")
c3.warning(f"💡 蜂窝密度: 每平方米约 {int(1/(2.598*hex_side**2))} 个")
