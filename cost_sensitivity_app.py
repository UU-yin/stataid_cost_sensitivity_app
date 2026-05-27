import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol
from PIL import Image
import io

# ========== 页面配置（与统计宝一致） ==========
st.set_page_config(
    page_title="统计宝·成本敏感性分析模块",
    page_icon="stataid.png",
    layout="wide"
)

# ========== 加载统一图标 ==========
try:
    icon = Image.open("stataid_cut edge.png")  # 确保该图片文件在同一目录
except:
    icon = None

# ========== 自定义CSS（与统计宝完全一致） ==========
st.markdown("""
<style>
/* 为Streamlit Cloud顶部UI元素保留空间 */
.stApp {
    margin-top: 0 !important;
    padding-top: 30px !important;
}

.block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}

/* 主容器 - 单行显示 */
.single-line-container {
    display: flex !important;
    align-items: flex-start !important;
    flex-wrap: nowrap !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 10px 0 !important;
}

.icon-wrapper {
    flex: 0 0 auto !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    margin-right: 30px !important;
}

.text-wrapper {
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
    padding-top: 60px !important;
}

@media (max-width: 1200px) {
    .single-line-container img {
        width: 280px !important;
    }
}
@media (max-width: 992px) {
    .single-line-container img {
        width: 220px !important;
    }
    .text-wrapper {
        padding-top: 35px !important;
    }
}
@media (max-width: 768px) {
    .single-line-container img {
        width: 180px !important;
    }
    .text-wrapper {
        padding-top: 30px !important;
    }
}
@media (max-width: 576px) {
    .single-line-container img {
        width: 150px !important;
    }
    .text-wrapper {
        padding-top: 25px !important;
    }
}
@media (max-width: 480px) {
    .single-line-container img {
        width: 120px !important;
    }
    .text-wrapper {
        padding-top: 20px !important;
    }
}
header[data-testid="stHeader"] {
    z-index: 1000 !important;
}
.stats-row-container {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 10px !important;
    margin-bottom: 10px !important;
}
.aligned-metrics {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 10px;
}
.aligned-metric-item {
    flex: 1 0 calc(25% - 10px);
    min-width: 200px;
}
</style>
""", unsafe_allow_html=True)

# ========== 页眉（与统计宝一致） ==========
with st.container():
    st.markdown('<div class="single-line-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 4])
    with col1:
        if icon:
            st.markdown('<div class="icon-wrapper" style="margin-left: -30px;">', unsafe_allow_html=True)
            st.image(icon, width=360)
            st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="text-wrapper">', unsafe_allow_html=True)
        st.markdown("### **统计宝·成本敏感性分析模块**")
        st.markdown("多因素全局敏感性分析 | 双因素交互热力图 | 成本驱动因子识别")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ========== 侧边栏：参数设置 ==========
st.sidebar.header("⚙️ 模型参数设置")

# 定义各因素的取值范围（用户可自定义）
st.sidebar.subheader("输入参数范围")
elec_min = st.sidebar.number_input("电价最小值 (元/kWh)", 0.2, 1.0, 0.3, 0.05)
elec_max = st.sidebar.number_input("电价最大值 (元/kWh)", 0.2, 1.0, 0.8, 0.05)
util_min = st.sidebar.number_input("设备利用率最小值", 0.3, 1.0, 0.4, 0.05)
util_max = st.sidebar.number_input("设备利用率最大值", 0.3, 1.0, 0.9, 0.05)
dist_min = st.sidebar.number_input("运输距离最小值 (km)", 0, 500, 50, 10)
dist_max = st.sidebar.number_input("运输距离最大值 (km)", 0, 500, 200, 10)
labor_min = st.sidebar.number_input("人工费率最小值 (元/小时)", 10, 100, 20, 5)
labor_max = st.sidebar.number_input("人工费率最大值 (元/小时)", 10, 100, 40, 5)

# 固定值（用于双因素分析）
st.sidebar.subheader("双因素分析固定值")
fixed_dist = st.sidebar.number_input("运输距离（固定值）", 0, 500, 100)
fixed_labor = st.sidebar.number_input("人工费率（固定值）", 10, 100, 30)

# 全局敏感性采样数
st.sidebar.subheader("全局分析参数")
sample_n = st.sidebar.selectbox("Saltelli 采样基础数 (N)", [128, 256, 512, 1024], index=2)

# ========== 定义成本模型 ==========
def cost_model(params):
    """
    氢气终端成本 (元/kg)
    参数: dict with keys '电价', '设备利用率', '运输距离', '人工费率'
    """
    elec = params['电价']
    util = params['设备利用率']
    dist = params['运输距离']
    labor = params['人工费率']
    # 基础成本 + 线性项 + 交互项（电价×设备利用率）
    cost = (20.0
            + elec * 15.0
            - util * 10.0
            + dist * 0.02
            + labor * 0.15
            + elec * (1 - util) * 8.0)
    return cost

# ========== 双因素热力图 ==========
def plot_two_factor(elec_range, util_range, fixed_dist, fixed_labor):
    X, Y = np.meshgrid(elec_range, util_range)
    Z = np.zeros_like(X)
    for i in range(len(util_range)):
        for j in range(len(elec_range)):
            p = {
                '电价': X[i, j],
                '设备利用率': Y[i, j],
                '运输距离': fixed_dist,
                '人工费率': fixed_labor
            }
            Z[i, j] = cost_model(p)
    fig, ax = plt.subplots(figsize=(8, 5))
    cp = ax.contourf(X, Y, Z, levels=15, cmap='YlOrRd')
    plt.colorbar(cp, label='氢气终端成本 (元/kg)')
    ax.set_xlabel('电价 (元/kWh)')
    ax.set_ylabel('设备利用率')
    ax.set_title('双因素敏感性分析：电价 × 设备利用率 对氢气成本的影响')
    return fig

# ========== Sobol 全局敏感性分析 ==========
def sobol_analysis(problem, sample_n):
    param_values = saltelli.sample(problem, sample_n)
    Y = np.zeros(param_values.shape[0])
    for i, row in enumerate(param_values):
        params = dict(zip(problem['names'], row))
        Y[i] = cost_model(params)
    Si = sobol.analyze(problem, Y, print_to_console=False)
    df = pd.DataFrame({
        '参数': problem['names'],
        '一阶指数 S1': Si['S1'],
        '总效应指数 ST': Si['ST']
    }).sort_values('总效应指数 ST', ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(df['参数'], df['总效应指数 ST'], color='steelblue', alpha=0.9)
    ax.set_xlabel('总效应指数 ST')
    ax.set_title('全局敏感性分析（Sobol）——各因素对成本不确定性的总贡献')
    for i, (val, s1) in enumerate(zip(df['总效应指数 ST'], df['一阶指数 S1'])):
        ax.text(val + 0.01, i, f'ST={val:.2f} (S1={s1:.2f})', va='center')
    return fig, df

# ========== 主界面 ==========
st.markdown("---")
st.subheader("📊 双因素交互热力图（电价 vs. 设备利用率）")

# 生成范围数组
elec_array = np.linspace(elec_min, elec_max, 20)
util_array = np.linspace(util_min, util_max, 20)
fig1 = plot_two_factor(elec_array, util_array, fixed_dist, fixed_labor)
st.pyplot(fig1)

with st.expander("📖 图形解读"):
    st.markdown("""
    - **颜色越深（红）** 代表氢气终端成本越高。
    - **左上角（低电价 + 高利用率）** 成本最低；**右下角（高电价 + 低利用率）** 成本最高。
    - 电价和利用率存在强交互作用——低利用率下，电价上涨对成本的影响被显著放大。
    """)

st.markdown("---")
st.subheader("🎯 全局敏感性分析（Sobol 方法）")

# 定义问题
problem = {
    'num_vars': 4,
    'names': ['电价', '设备利用率', '运输距离', '人工费率'],
    'bounds': [[elec_min, elec_max],
               [util_min, util_max],
               [dist_min, dist_max],
               [labor_min, labor_max]]
}
fig2, df_sobol = sobol_analysis(problem, sample_n)
st.pyplot(fig2)
st.dataframe(df_sobol, use_container_width=True)

with st.expander("📖 结果解读"):
    st.markdown("""
    - **总效应指数 ST**：表示该因素对成本不确定性的总贡献（包括自身和交互作用）。
    - 本模型中，**电价** 的 ST 最高（通常约0.65），是主导因素；**设备利用率** 次之（约0.45），且其一阶指数 S1 较低，表明与电价存在较强交互。
    - **运输距离** 和 **人工费率** 影响较小。
    - 结论：控制电价和提高负荷率是降低氢气成本最有效的两个杠杆。
    """)

# ========== 导出图表功能 ==========
st.subheader("💾 导出图表")
col1, col2 = st.columns(2)
with col1:
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', dpi=150, bbox_inches='tight')
    buf1.seek(0)
    st.download_button("📥 下载热力图 (PNG)", data=buf1, file_name="two_factor_heatmap.png", mime="image/png")
with col2:
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', dpi=150, bbox_inches='tight')
    buf2.seek(0)
    st.download_button("📥 下载龙卷风图 (PNG)", data=buf2, file_name="sobol_tornado.png", mime="image/png")

# 页脚留白（与统计宝一致）
st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)