import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import warnings

# 尝试导入财务函数库
try:
    import numpy_financial as npf
    IRR_FUNC = npf.irr
except ImportError:
    from scipy.optimize import newton
    def IRR_FUNC(cashflows):
        """使用牛顿法计算IRR"""
        cashflows = np.asarray(cashflows)
        if np.all(cashflows >= 0) or np.all(cashflows <= 0):
            return np.nan
        # 猜测值
        guess = 0.1
        try:
            return newton(lambda r: np.npv(r, cashflows), guess, maxiter=100)
        except:
            return np.nan

# 全局敏感性分析库
try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False

# ---------------------------- 辅助函数 ----------------------------
def crf(r, n):
    """资本回收系数 Capital Recovery Factor"""
    if r == 0:
        return 1 / n
    return (r * (1 + r)**n) / ((1 + r)**n - 1)

def compute_npv(I, cf_series, r, n):
    """计算NPV: Σ CF_t/(1+r)^t - I"""
    t = np.arange(1, n + 1)
    # 如果cf_series是标量，扩展为每年相同现金流
    if np.isscalar(cf_series):
        cfs = np.full(n, cf_series)
    else:
        cfs = np.asarray(cf_series)[:n]
    pv_factors = (1 + r) ** t
    npv_val = np.sum(cfs / pv_factors) - I
    return npv_val

def compute_irr(I, cf_series, n):
    """计算IRR"""
    if np.isscalar(cf_series):
        cfs = np.full(n, cf_series)
    else:
        cfs = np.asarray(cf_series)[:n]
    cashflow_seq = np.insert(cfs, 0, -I)
    return IRR_FUNC(cashflow_seq)

def compute_lcoh(I, r, n, C_op, Q):
    """计算LCOH: (I*CRF + C_op) / Q"""
    if Q == 0:
        return np.nan
    ann_cap = I * crf(r, n)
    return (ann_cap + C_op) / Q

# ---------------------------- 页面设置 ----------------------------
st.set_page_config(page_title="氢能项目经济性分析平台", layout="wide")
st.title("🔋 氢能项目 NPV / IRR / LCOH 在线计算与敏感性分析")

# ---------------------------- 侧边栏：数据输入 ----------------------------
st.sidebar.header("📥 数据输入方式")
input_mode = st.sidebar.radio("选择输入方式", ["手动输入", "上传文件 (CSV/Excel)"])

# 初始化session_state存储参数
if 'params' not in st.session_state:
    st.session_state.params = {}

# 手动输入模式
if input_mode == "手动输入":
    st.sidebar.subheader("基本参数")
    I = st.sidebar.number_input("初始投资 I (万元)", value=1000.0, step=100.0)
    r_base = st.sidebar.number_input("基准折现率 r (如0.08=8%)", value=0.08, step=0.01, format="%.3f")
    n_base = st.sidebar.number_input("项目寿命期 n (年)", value=20, min_value=1, step=1)
    Q = st.sidebar.number_input("年制氢量 Q (kg/年)", value=50000.0, step=1000.0)
    C_op = st.sidebar.number_input("年运营成本 C_op (万元/年)", value=200.0, step=10.0)

    st.sidebar.subheader("净现金流设置")
    cf_mode = st.sidebar.radio("现金流类型", ["等额年金（各年相同）", "逐年输入"])
    if cf_mode == "等额年金（各年相同）":
        cf_val = st.sidebar.number_input("年均净现金流 (万元)", value=300.0, step=10.0)
        cf_series = cf_val
    else:
        cf_str = st.sidebar.text_area("请输入各年净现金流，用逗号分隔（万元）", value="300,300,300,300,300")
        try:
            cf_series = [float(x.strip()) for x in cf_str.split(",") if x.strip() != ""]
        except:
            st.sidebar.error("现金流格式错误，请使用逗号分隔数字")
            cf_series = [300]

    # 高级：折旧与税率（可选）
    with st.sidebar.expander("高级：折旧与税率（构建现金流时自动计算）"):
        use_dep_tax = st.checkbox("启用折旧和税率计算净现金流", value=False)
        if use_dep_tax:
            revenue = st.number_input("年收入 (万元)", value=500.0)
            op_cost = C_op  # 沿用运营成本
            depreciation = st.number_input("年折旧 (万元)", value=50.0)
            tax_rate = st.number_input("税率 (如0.25)", value=0.25)
            # 重新计算净现金流: (收入 - 运营成本 - 折旧)*(1-税率) + 折旧
            cf_val = (revenue - op_cost - depreciation) * (1 - tax_rate) + depreciation
            st.info(f"根据公式计算得到年均净现金流 = {cf_val:.2f} 万元")
            cf_series = cf_val  # 覆盖

    # 多场景参数（额外折现率和寿命）
    st.sidebar.subheader("多场景计算（可选）")
    multi_scenario = st.sidebar.checkbox("启用多组折现率/寿命期计算")
    if multi_scenario:
        r_list_str = st.sidebar.text_input("折现率列表，逗号分隔", value=f"{r_base},0.10,0.12")
        n_list_str = st.sidebar.text_input("寿命期列表，逗号分隔（整数）", value=f"{n_base},15,25")
        try:
            r_list = [float(x.strip()) for x in r_list_str.split(",")]
            n_list = [int(x.strip()) for x in n_list_str.split(",")]
        except:
            st.sidebar.error("列表格式错误")
            r_list, n_list = [r_base], [n_base]
    else:
        r_list, n_list = [r_base], [n_base]

    # 存储到session
    st.session_state.params = {
        'I': I, 'r_base': r_base, 'n_base': n_base, 'Q': Q, 'C_op': C_op,
        'cf_series': cf_series, 'r_list': r_list, 'n_list': n_list
    }

# 文件上传模式
elif input_mode == "上传文件 (CSV/Excel)":
    uploaded_file = st.sidebar.file_uploader("上传文件", type=['csv', 'xlsx'])
    st.sidebar.markdown("文件需包含列: `I`, `r`, `n`, `Q`, `C_op`, `cf_type`, `cf_values`")
    st.sidebar.markdown("- `cf_type`: 填 `uniform` 表示各年相同, 则 `cf_values` 填一个数值；填 `custom` 表示逐年，`cf_values` 用逗号分隔各年现金流")
    st.sidebar.download_button(
        "📥 下载标准模板",
        data="I,r,n,Q,C_op,cf_type,cf_values\n1000,0.08,20,50000,200,uniform,300\n1200,0.10,25,60000,250,uniform,350",
        file_name="template.csv"
    )
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            # 验证必要列
            required = {'I', 'r', 'n', 'Q', 'C_op', 'cf_type', 'cf_values'}
            if not required.issubset(df.columns):
                st.sidebar.error("文件缺少必要的列")
            else:
                st.session_state.uploaded_df = df
                st.sidebar.success("文件上传成功！")
        except Exception as e:
            st.sidebar.error(f"读取文件出错: {e}")
    else:
        st.session_state.uploaded_df = None

# ---------------------------- 主区域：数据展示与计算 ----------------------------
st.header("📊 输入数据总览")
if input_mode == "手动输入":
    p = st.session_state.params
    # 构建三列数据表
    data_table = []
    data_table.append(["初始投资 I", f"{p['I']} 万元", "万元"])
    data_table.append(["基准折现率 r", f"{p['r_base']*100:.2f}%", "-"])
    data_table.append(["项目寿命期 n", f"{p['n_base']}", "年"])
    data_table.append(["年制氢量 Q", f"{p['Q']}", "kg/年"])
    data_table.append(["年运营成本 C_op", f"{p['C_op']}", "万元/年"])
    if np.isscalar(p['cf_series']):
        cf_display = f"年均 {p['cf_series']} 万元"
    else:
        cf_display = f"{p['cf_series']} 万元 (逐年)"
    data_table.append(["各年净现金流", cf_display, "万元"])
    df_input = pd.DataFrame(data_table, columns=["项目", "数值", "单位"])
    st.table(df_input)

    # 多场景表
    if multi_scenario:
        st.subheader("多场景参数组合")
        scenario_list = [(rr, nn) for rr in p['r_list'] for nn in p['n_list']]
        st.caption(f"共 {len(scenario_list)} 个场景")
        results = []
        for rr, nn in scenario_list:
            npv_val = compute_npv(p['I'], p['cf_series'], rr, nn)
            irr_val = compute_irr(p['I'], p['cf_series'], nn)
            lcoh_val = compute_lcoh(p['I'], rr, nn, p['C_op'], p['Q'])
            results.append({
                "折现率 r": f"{rr*100:.2f}%",
                "寿命期 n": nn,
                "NPV (万元)": f"{npv_val:.2f}",
                "IRR (%)": f"{irr_val*100:.2f}" if not np.isnan(irr_val) else "无解",
                "LCOH (元/kg)": f"{lcoh_val:.4f}"
            })
        st.dataframe(pd.DataFrame(results))
elif input_mode == "上传文件 (CSV/Excel)":
    if st.session_state.uploaded_df is not None:
        df = st.session_state.uploaded_df
        st.dataframe(df)
    else:
        st.info("请在侧边栏上传文件")

# ---------------------------- 基准计算结果 ----------------------------
st.header("💰 基准计算结果")
if input_mode == "手动输入":
    p = st.session_state.params
    I, r_base, n_base, Q, C_op = p['I'], p['r_base'], p['n_base'], p['Q'], p['C_op']
    cf = p['cf_series']
    npv_base = compute_npv(I, cf, r_base, n_base)
    irr_base = compute_irr(I, cf, n_base)
    lcoh_base = compute_lcoh(I, r_base, n_base, C_op, Q)

    col1, col2, col3 = st.columns(3)
    col1.metric("NPV (万元)", f"{npv_base:.2f}")
    col2.metric("IRR (%)", f"{irr_base*100:.2f}" if not np.isnan(irr_base) else "无解")
    col3.metric("LCOH (元/kg)", f"{lcoh_base:.4f}")
elif input_mode == "上传文件 (CSV/Excel)":
    if st.session_state.uploaded_df is not None:
        df = st.session_state.uploaded_df
        all_results = []
        for idx, row in df.iterrows():
            cf_type = row['cf_type']
            cf_raw = str(row['cf_values'])
            if cf_type == 'uniform':
                cf_val = float(cf_raw)
                cf_series = cf_val
            else:
                cf_series = [float(x.strip()) for x in cf_raw.split(',')]
            npv_val = compute_npv(row['I'], cf_series, row['r'], int(row['n']))
            irr_val = compute_irr(row['I'], cf_series, int(row['n']))
            lcoh_val = compute_lcoh(row['I'], row['r'], int(row['n']), row['C_op'], row['Q'])
            all_results.append({
                "项目序号": idx+1,
                "NPV (万元)": npv_val,
                "IRR (%)": irr_val*100 if not np.isnan(irr_val) else np.nan,
                "LCOH (元/kg)": lcoh_val
            })
        st.dataframe(pd.DataFrame(all_results))
        st.success("批量计算完成")

# ---------------------------- 敏感性分析模块 ----------------------------
st.header("📈 敏感性分析")
tab1, tab2, tab3 = st.tabs(["单因素分析", "双因素分析", "全局敏感性 (Sobol)"])

# ----- 通用参数选择函数（针对手动输入模式）-----
def get_param_dict():
    """返回可用于敏感性分析的参数名-基准值字典"""
    p = st.session_state.params
    if input_mode == "手动输入":
        cf_base = np.mean(p['cf_series']) if not np.isscalar(p['cf_series']) else p['cf_series']
        return {
            "初始投资 I (万元)": p['I'],
            "年净现金流 (万元)": cf_base,
            "折现率 r": p['r_base'],
            "项目寿命 n (年)": p['n_base'],
            "年运营成本 C_op (万元)": p['C_op'],
            "年制氢量 Q (kg)": p['Q']
        }
    else:
        return None

def compute_sensitivity_measures(param_name, base_val, change_type, change_level, base_params):
    """
    计算变动后的NPV和IRR。
    change_type: 'relative' 或 'absolute'
    change_level: 百分比如10表示+10%，或绝对变化数值
    """
    new_val = base_val * (1 + change_level/100) if change_type == 'relative' else base_val + change_level
    # 复制基准参数
    I = base_params['I']
    r = base_params['r']
    n = base_params['n']
    C_op = base_params['C_op']
    Q = base_params['Q']
    cf = base_params['cf_base']
    # 根据参数名更新
    if "初始投资" in param_name:
        I = new_val
    elif "年净现金流" in param_name:
        cf = new_val
    elif "折现率" in param_name:
        r = new_val
    elif "项目寿命" in param_name:
        n = int(max(1, round(new_val)))
    elif "年运营成本" in param_name:
        C_op = new_val
    elif "年制氢量" in param_name:
        Q = new_val
    else:
        pass
    npv = compute_npv(I, cf, r, n)
    irr = compute_irr(I, cf, n)
    lcoh = compute_lcoh(I, r, n, C_op, Q) if '制氢' in param_name or '运营' in param_name or '初始投资' in param_name or '折现率' in param_name else None
    return npv, irr, lcoh

# 单因素分析
with tab1:
    st.subheader("单因素敏感性分析（龙卷风图）")
    if input_mode == "手动输入":
        param_dict = get_param_dict()
        base_params = {
            'I': st.session_state.params['I'],
            'r': st.session_state.params['r_base'],
            'n': st.session_state.params['n_base'],
            'C_op': st.session_state.params['C_op'],
            'Q': st.session_state.params['Q'],
            'cf_base': np.mean(st.session_state.params['cf_series']) if not np.isscalar(st.session_state.params['cf_series']) else st.session_state.params['cf_series']
        }
        selected_params = st.multiselect("选择要分析的参数", list(param_dict.keys()), default=list(param_dict.keys())[:3])
        range_options = ["±10%", "±20%", "±30%", "自定义"]
        range_choice = st.selectbox("变动范围", range_options, index=1)
        if range_choice == "自定义":
            custom_range = st.number_input("变动百分比（绝对值）", value=20.0, step=1.0)
            levels = [-custom_range, custom_range]
        else:
            pct = int(range_choice.replace("±", "").replace("%", ""))
            levels = [-pct, pct]

        # 绝对变动选项
        use_absolute = st.checkbox("启用绝对数值变动 (替代百分比)", value=False)
        abs_changes = {}
        if use_absolute:
            st.caption("为每个选定参数输入绝对增减量")
            for param in selected_params:
                c1, c2 = st.columns(2)
                with c1:
                    down = st.number_input(f"{param} 减少量", value=0.0, step=1.0, key=f"abs_down_{param}")
                with c2:
                    up = st.number_input(f"{param} 增加量", value=0.0, step=1.0, key=f"abs_up_{param}")
                abs_changes[param] = (down, up)

        if st.button("运行单因素分析"):
            change_type = 'absolute' if use_absolute else 'relative'
            results_npv = []
            results_irr = []
            for param in selected_params:
                base_val = param_dict[param]
                if change_type == 'relative':
                    low_change, high_change = levels[0], levels[1]
                else:
                    low_change, high_change = -abs_changes[param][0], abs_changes[param][1]
                # 低变动
                npv_low, irr_low, lcoh_low = compute_sensitivity_measures(param, base_val, change_type, low_change, base_params)
                # 高变动
                npv_high, irr_high, lcoh_high = compute_sensitivity_measures(param, base_val, change_type, high_change, base_params)
                results_npv.append({
                    "参数": param,
                    "变动低": f"{low_change:+.1f}{'%' if change_type=='relative' else ''}",
                    "NPV (低)": npv_low,
                    "变动高": f"{high_change:+.1f}{'%' if change_type=='relative' else ''}",
                    "NPV (高)": npv_high,
                    "NPV 变化范围": npv_high - npv_low
                })
                results_irr.append({
                    "参数": param,
                    "IRR (低)": irr_low * 100 if not np.isnan(irr_low) else np.nan,
                    "IRR (高)": irr_high * 100 if not np.isnan(irr_high) else np.nan,
                    "IRR 变化范围": (irr_high - irr_low)*100 if not np.isnan(irr_high) and not np.isnan(irr_low) else np.nan
                })
            df_npv = pd.DataFrame(results_npv)
            df_irr = pd.DataFrame(results_irr)
            st.write("**NPV 敏感性结果**")
            st.dataframe(df_npv.style.format("{:.2f}", subset=["NPV (低)", "NPV (高)", "NPV 变化范围"]))
            st.write("**IRR 敏感性结果**")
            st.dataframe(df_irr.style.format("{:.2f}", subset=["IRR (低)", "IRR (高)", "IRR 变化范围"]))

            # 龙卷风图（NPV）
            fig, ax = plt.subplots(figsize=(10, 5))
            param_names = [r['参数'] for r in results_npv]
            npv_ranges = [r['NPV 变化范围'] for r in results_npv]
            colors = ['red' if x > 0 else 'green' for x in npv_ranges]
            ax.barh(param_names, npv_ranges, color=colors)
            ax.set_xlabel("NPV 变化范围 (万元)")
            ax.set_title("单因素敏感性 - NPV 龙卷风图")
            st.pyplot(fig)

            # 相对测定法与绝对测定法简单对比说明
            st.markdown("""**分析解读**：  
- 龙卷风图的条形越长，表示该因素对NPV的影响越大。  
- 若采用百分比变动（相对测定），结果展示了参数弹性；若采用绝对变动（绝对测定），则反映了实际数值变动对结果的冲击。建议结合两种方式，识别关键风险因素。""")
            # 下载功能
            csv_npv = df_npv.to_csv(index=False).encode('utf-8')
            st.download_button("下载 NPV 敏感性表 (CSV)", csv_npv, "npv_sensitivity.csv")
    else:
        st.info("单因素分析目前仅支持手动输入模式")

# 双因素分析
with tab2:
    st.subheader("双因素敏感性分析（热力图）")
    if input_mode == "手动输入":
        param_dict = get_param_dict()
        base_params = {
            'I': st.session_state.params['I'],
            'r': st.session_state.params['r_base'],
            'n': st.session_state.params['n_base'],
            'C_op': st.session_state.params['C_op'],
            'Q': st.session_state.params['Q'],
            'cf_base': np.mean(st.session_state.params['cf_series']) if not np.isscalar(st.session_state.params['cf_series']) else st.session_state.params['cf_series']
        }
        col_a, col_b = st.columns(2)
        with col_a:
            param_x = st.selectbox("X轴参数", list(param_dict.keys()), index=0)
            x_range = st.slider("X轴变化范围 (%)", -50, 50, (-20, 20), step=5)
        with col_b:
            param_y = st.selectbox("Y轴参数", list(param_dict.keys()), index=1)
            y_range = st.slider("Y轴变化范围 (%)", -50, 50, (-20, 20), step=5)
        target = st.selectbox("分析目标", ["NPV", "IRR", "LCOH"])
        if st.button("生成热力图"):
            x_vals = np.linspace(x_range[0]/100, x_range[1]/100, 10)
            y_vals = np.linspace(y_range[0]/100, y_range[1]/100, 10)
            grid = np.zeros((len(y_vals), len(x_vals)))
            for i, dy in enumerate(y_vals):
                for j, dx in enumerate(x_vals):
                    # 更新参数值
                    I_tmp = base_params['I']
                    r_tmp = base_params['r']
                    n_tmp = base_params['n']
                    C_op_tmp = base_params['C_op']
                    Q_tmp = base_params['Q']
                    cf_tmp = base_params['cf_base']
                    for pname, chg in zip([param_x, param_y], [dx, dy]):
                        base_val = param_dict[pname]
                        new_val = base_val * (1 + chg)
                        if "初始投资" in pname:
                            I_tmp = new_val
                        elif "年净现金流" in pname:
                            cf_tmp = new_val
                        elif "折现率" in pname:
                            r_tmp = new_val
                        elif "项目寿命" in pname:
                            n_tmp = int(max(1, round(new_val)))
                        elif "年运营成本" in pname:
                            C_op_tmp = new_val
                        elif "年制氢量" in pname:
                            Q_tmp = new_val
                    if target == "NPV":
                        grid[i,j] = compute_npv(I_tmp, cf_tmp, r_tmp, n_tmp)
                    elif target == "IRR":
                        irr_tmp = compute_irr(I_tmp, cf_tmp, n_tmp)
                        grid[i,j] = irr_tmp*100 if not np.isnan(irr_tmp) else np.nan
                    else:
                        grid[i,j] = compute_lcoh(I_tmp, r_tmp, n_tmp, C_op_tmp, Q_tmp)
            fig, ax = plt.subplots()
            c = ax.contourf(x_vals*100, y_vals*100, grid, levels=15, cmap='RdYlGn')
            ax.set_xlabel(f"{param_x} 变化(%)")
            ax.set_ylabel(f"{param_y} 变化(%)")
            ax.set_title(f"{target} 双因素热力图")
            fig.colorbar(c, ax=ax, label=target)
            st.pyplot(fig)
            st.markdown("**技术讲解**：热力图显示两个参数协同变动时目标值的变化趋势，颜色越深表示变化越剧烈，可直观识别交互效应。")
    else:
        st.info("双因素分析目前仅支持手动输入模式")

# 全局敏感性分析 (Sobol)
with tab3:
    st.subheader("全局敏感性分析 (Sobol方法)")
    if not SALIB_AVAILABLE:
        st.error("需要安装SALib库，请在requirements.txt中添加 SALib。")
    elif input_mode != "手动输入":
        st.info("Sobol分析仅支持手动输入模式")
    else:
        param_dict = get_param_dict()
        base_params = {
            'I': st.session_state.params['I'],
            'r': st.session_state.params['r_base'],
            'n': st.session_state.params['n_base'],
            'C_op': st.session_state.params['C_op'],
            'Q': st.session_state.params['Q'],
            'cf_base': np.mean(st.session_state.params['cf_series']) if not np.isscalar(st.session_state.params['cf_series']) else st.session_state.params['cf_series']
        }
        # 将参数名与代码内变量映射
        param_mapping = {
            "初始投资 I (万元)": 'I',
            "年净现金流 (万元)": 'cf',
            "折现率 r": 'r',
            "项目寿命 n (年)": 'n',
            "年运营成本 C_op (万元)": 'C_op',
            "年制氢量 Q (kg)": 'Q'
        }
        selected_sobol = st.multiselect("选择参与全局敏感性分析的参数", list(param_dict.keys()), default=["初始投资 I (万元)", "年净现金流 (万元)", "折现率 r"])
        target_sobol = st.selectbox("分析输出指标", ["NPV", "IRR", "LCOH"], key="sobol_target")
        sample_N = st.number_input("每个参数的样本数 (N)，总样本 N*(2D+2) 建议不小于1000", value=512, min_value=64, step=64)
        if st.button("运行 Sobol 分析"):
            if len(selected_sobol) < 2:
                st.warning("请至少选择两个参数")
            else:
                # 定义问题
                problem = {
                    'num_vars': len(selected_sobol),
                    'names': selected_sobol,
                    'bounds': []
                }
                for p in selected_sobol:
                    base_v = param_dict[p]
                    # 上下界设为基准值的±30%
                    low = base_v * 0.7 if base_v != 0 else -1
                    high = base_v * 1.3 if base_v != 0 else 1
                    if "折现率" in p or "税率" in p:  # 比率限制在0~1
                        low = max(0.01, base_v * 0.5)
                        high = min(0.5, base_v * 1.5)
                    if "项目寿命" in p:
                        low = max(1, base_v - 10)
                        high = base_v + 10
                    problem['bounds'].append([low, high])
                # 生成样本
                param_values = saltelli.sample(problem, sample_N, calc_second_order=False)
                # 运行模型
                Y = np.zeros(param_values.shape[0])
                for i, row in enumerate(param_values):
                    # 映射到计算参数
                    I_tmp = base_params['I']
                    r_tmp = base_params['r']
                    n_tmp = base_params['n']
                    C_op_tmp = base_params['C_op']
                    Q_tmp = base_params['Q']
                    cf_tmp = base_params['cf_base']
                    for j, pname in enumerate(selected_sobol):
                        val = row[j]
                        code = param_mapping[pname]
                        if code == 'I':
                            I_tmp = val
                        elif code == 'cf':
                            cf_tmp = val
                        elif code == 'r':
                            r_tmp = val
                        elif code == 'n':
                            n_tmp = int(max(1, round(val)))
                        elif code == 'C_op':
                            C_op_tmp = val
                        elif code == 'Q':
                            Q_tmp = val
                    if target_sobol == "NPV":
                        Y[i] = compute_npv(I_tmp, cf_tmp, r_tmp, n_tmp)
                    elif target_sobol == "IRR":
                        irr_tmp = compute_irr(I_tmp, cf_tmp, n_tmp)
                        Y[i] = irr_tmp * 100 if not np.isnan(irr_tmp) else 0
                    else:
                        Y[i] = compute_lcoh(I_tmp, r_tmp, n_tmp, C_op_tmp, Q_tmp)
                # Sobol分析
                Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)
                # 绘图
                fig, ax = plt.subplots(figsize=(8, 5))
                indices = [Si['S1'], Si['ST']]
                labels = ['一阶效应 S1', '总效应 ST']
                x_pos = np.arange(len(selected_sobol))
                width = 0.35
                ax.bar(x_pos - width/2, indices[0], width, label=labels[0])
                ax.bar(x_pos + width/2, indices[1], width, label=labels[1])
                ax.set_xticks(x_pos)
                ax.set_xticklabels(selected_sobol, rotation=45, ha='right')
                ax.set_ylabel("敏感指数")
                ax.set_title(f"{target_sobol} 的 Sobol 敏感指数")
                ax.legend()
                st.pyplot(fig)
                # 显示数值
                st.write("**一阶敏感指数 (S1)**", dict(zip(selected_sobol, Si['S1'])))
                st.write("**总效应指数 (ST)**", dict(zip(selected_sobol, Si['ST'])))
                st.markdown("""
                **分析解读**：  
                - S1 表示该参数单独对输出方差的贡献；  
                - ST 包含参数自身以及与其他参数交互作用的贡献。  
                若 ST >> S1，说明该参数与其他因素有较强的交互效应。
                """)

# ---------------------------- 页脚 ----------------------------
st.sidebar.markdown("---")
st.sidebar.caption("氢能项目经济性分析平台 v1.0 | 部署于 Streamlit Cloud")
