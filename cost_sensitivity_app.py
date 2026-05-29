import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# 财务函数导入
try:
    import numpy_financial as npf
    IRR_FUNC = npf.irr
except ImportError:
    from scipy.optimize import newton
    def IRR_FUNC(cashflows):
        cashflows = np.asarray(cashflows)
        if np.all(cashflows >= 0) or np.all(cashflows <= 0):
            return np.nan
        try:
            return newton(lambda r: np.npv(r, cashflows), 0.1, maxiter=100)
        except:
            return np.nan

try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False

# ---------------------------- 核心计算 ----------------------------
def crf(r, n):
    return (r * (1 + r)**n) / ((1 + r)**n - 1) if r != 0 else 1/n

def compute_npv(I, cf, r, n):
    t = np.arange(1, n + 1)
    cfs = np.full(n, cf) if np.isscalar(cf) else np.asarray(cf)[:n]
    return np.sum(cfs / (1 + r)**t) - I

def compute_irr(I, cf, n):
    cfs = np.full(n, cf) if np.isscalar(cf) else np.asarray(cf)[:n]
    return IRR_FUNC(np.insert(cfs, 0, -I))

def compute_lcoh(I, r, n, C_op, Q):
    if Q == 0:
        return np.nan
    return (I * crf(r, n) + C_op) / Q

def get_relevant_params(target):
    if target in ["NPV", "IRR"]:
        return ["初始投资 I (万元)", "年净现金流 (万元)", "折现率 r", "项目寿命 n (年)"]
    else:
        return ["初始投资 I (万元)", "折现率 r", "项目寿命 n (年)", "年运营成本 C_op (万元)", "年制氢量 Q (kg)"]

# 中英文参数名映射（用于图表）
PARAM_EN = {
    "初始投资 I (万元)": "Initial Investment I",
    "年净现金流 (万元)": "Annual Net CF",
    "折现率 r": "Discount Rate r",
    "项目寿命 n (年)": "Project Life n",
    "年运营成本 C_op (万元)": "Annual OPEX",
    "年制氢量 Q (kg)": "Annual H2 Output Q"
}

# ---------------------------- 页面配置 ----------------------------
st.set_page_config(page_title="项目经济性分析平台", layout="wide")
st.title("项目成本计算与敏感性分析平台")

# ---------------------------- 侧边栏：指标选择 ----------------------------
st.sidebar.header("📌 分析方法选择")
analysis_scope = st.sidebar.selectbox("选择指标数量", ["单个方法", "两种方法", "三种方法"], index=1)
all_targets = ["NPV", "IRR", "LCOH"]
if analysis_scope == "单个方法":
    selected_targets = [st.sidebar.selectbox("选择指标", all_targets, index=0)]
elif analysis_scope == "两种方法":
    selected_targets = st.sidebar.multiselect("选择两个指标", all_targets, default=["NPV", "IRR"], max_selections=2)
    if len(selected_targets) != 2:
        st.sidebar.warning("请选择恰好两个指标")
else:
    selected_targets = st.sidebar.multiselect("选择指标 (默认全选)", all_targets, default=all_targets)
    if not selected_targets:
        st.sidebar.warning("请至少选择一个指标")

st.session_state['selected_targets'] = selected_targets if selected_targets else all_targets

# ---------------------------- 主界面：数据输入 ----------------------------
st.header("📥 数据输入")
input_mode = st.radio("选择输入方式", ["手动输入", "上传文件 (CSV/Excel)"], horizontal=True)

if 'params' not in st.session_state:
    st.session_state.params = {}

if input_mode == "手动输入":
    # 根据所选指标动态显示输入字段
    has_lcoh = "LCOH" in st.session_state.selected_targets
    col1, col2, col3 = st.columns(3)
    with col1:
        I = st.number_input("初始投资 I (万元)", value=1000.0, step=100.0)
    with col2:
        r_base = st.number_input("基准折现率 r (如0.08=8%)", value=0.08, step=0.01, format="%.3f")
    with col3:
        n_base = st.number_input("项目寿命期 n (年)", value=20, min_value=1, step=1)

    if has_lcoh:
        col4, col5 = st.columns(2)
        with col4:
            Q = st.number_input("年制氢量 Q (kg/年)", value=50000.0, step=1000.0)
        with col5:
            C_op = st.number_input("年运营成本 C_op (万元/年)", value=200.0, step=10.0)
    else:
        Q = 1.0   # 安全默认值，不影响 NPV/IRR
        C_op = 0.0

    st.subheader("净现金流设置")
    cf_mode = st.radio("现金流类型", ["等额年金（各年相同）", "逐年输入"], horizontal=True)
    if cf_mode == "等额年金（各年相同）":
        cf_val = st.number_input("年均净现金流 (万元)", value=300.0, step=10.0)
        cf_series = cf_val
    else:
        cf_str = st.text_area("各年净现金流，逗号分隔（万元）", value="300,300,300,300,300")
        try:
            cf_series = [float(x.strip()) for x in cf_str.split(",") if x.strip() != ""]
        except:
            st.error("格式错误，请用逗号分隔")
            cf_series = [300]

    with st.expander("高级：由收入/成本/折旧/税率计算净现金流"):
        use_dep_tax = st.checkbox("启用联动计算")
        if use_dep_tax:
            c1, c2 = st.columns(2)
            with c1:
                revenue = st.number_input("年收入 (万元)", value=500.0)
                depreciation = st.number_input("年折旧 (万元)", value=50.0)
            with c2:
                tax_rate = st.number_input("税率 (如0.25)", value=0.25)
            cf_val = (revenue - C_op - depreciation) * (1 - tax_rate) + depreciation
            st.info(f"计算得到的年均净现金流 = {cf_val:.2f} 万元")
            cf_series = cf_val

    multi_scenario = st.checkbox("启用多组折现率/寿命期计算")
    if multi_scenario:
        c_r, c_n = st.columns(2)
        with c_r:
            r_list_str = st.text_input("折现率列表，逗号分隔", value=f"{r_base},0.10,0.12")
        with c_n:
            n_list_str = st.text_input("寿命期列表，逗号分隔", value=f"{n_base},15,25")
        try:
            r_list = [float(x.strip()) for x in r_list_str.split(",")]
            n_list = [int(x.strip()) for x in n_list_str.split(",")]
        except:
            st.error("列表格式错误")
            r_list, n_list = [r_base], [n_base]
    else:
        r_list, n_list = [r_base], [n_base]

    st.session_state.params = {
        'I': I, 'r_base': r_base, 'n_base': n_base, 'Q': Q, 'C_op': C_op,
        'cf_series': cf_series, 'r_list': r_list, 'n_list': n_list
    }

elif input_mode == "上传文件 (CSV/Excel)":
    uploaded_file = st.file_uploader("上传文件", type=['csv', 'xlsx'])
    st.markdown("文件需包含列: `I`, `r`, `n`, `Q`, `C_op`, `cf_type`, `cf_values`")
    st.markdown("- `cf_type`: `uniform` 各年相同, `custom` 逐年(逗号分隔)")
    st.download_button("📥 下载标准模板", data="I,r,n,Q,C_op,cf_type,cf_values\n1000,0.08,20,50000,200,uniform,300", file_name="template.csv")
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            if {'I', 'r', 'n', 'Q', 'C_op', 'cf_type', 'cf_values'}.issubset(df.columns):
                st.session_state.uploaded_df = df
                st.success("文件上传成功！")
            else:
                st.error("文件缺少必要列")
        except Exception as e:
            st.error(f"读取文件出错: {e}")
    else:
        st.session_state.uploaded_df = None

# ---------------------------- 基准计算结果 ----------------------------
st.header("📊 基准计算结果")
targets_to_show = st.session_state.get('selected_targets', ["NPV", "IRR", "LCOH"])

if input_mode == "手动输入":
    p = st.session_state.params
    I, r_base, n_base, Q, C_op = p['I'], p['r_base'], p['n_base'], p['Q'], p['C_op']
    cf = p['cf_series']
    npv_val = compute_npv(I, cf, r_base, n_base)
    irr_val = compute_irr(I, cf, n_base)
    lcoh_val = compute_lcoh(I, r_base, n_base, C_op, Q)

    cols = st.columns(len(targets_to_show))
    for idx, target in enumerate(targets_to_show):
        if target == "NPV":
            cols[idx].metric("NPV (万元)", f"{npv_val:.2f}")
        elif target == "IRR":
            cols[idx].metric("IRR (%)", f"{irr_val*100:.2f}" if not np.isnan(irr_val) else "无解")
        elif target == "LCOH":
            cols[idx].metric("LCOH (元/kg)", f"{lcoh_val:.4f}")

    with st.expander("📋 查看输入数据明细"):
        data = [
            ["初始投资 I", f"{I} 万元"],
            ["基准折现率 r", f"{r_base*100:.2f}%"],
            ["项目寿命期 n", f"{n_base} 年"],
        ]
        if "LCOH" in targets_to_show:
            data += [["年制氢量 Q", f"{Q} kg/年"], ["年运营成本 C_op", f"{C_op} 万元/年"]]
        if np.isscalar(cf):
            data.append(["各年净现金流", f"年均 {cf} 万元"])
        else:
            data.append(["各年净现金流", f"{cf} (逐年)"])
        st.table(pd.DataFrame(data, columns=["项目", "数值"]))

    if multi_scenario:
        st.subheader("多场景结果")
        results = []
        for rr, nn in [(r, n) for r in r_list for n in n_list]:
            row = {"折现率 r": f"{rr*100:.2f}%", "寿命 n": nn}
            if "NPV" in targets_to_show: row["NPV (万元)"] = f"{compute_npv(I, cf, rr, nn):.2f}"
            if "IRR" in targets_to_show:
                irr = compute_irr(I, cf, nn)
                row["IRR (%)"] = f"{irr*100:.2f}" if not np.isnan(irr) else "无解"
            if "LCOH" in targets_to_show: row["LCOH (元/kg)"] = f"{compute_lcoh(I, rr, nn, C_op, Q):.4f}"
            results.append(row)
        st.dataframe(pd.DataFrame(results))

elif input_mode == "上传文件 (CSV/Excel)":
    if st.session_state.get('uploaded_df') is not None:
        df = st.session_state.uploaded_df
        st.subheader("上传数据预览")
        st.dataframe(df)
        all_results = []
        for _, row in df.iterrows():
            cf_raw = str(row['cf_values'])
            cf_series = float(cf_raw) if row['cf_type'] == 'uniform' else [float(x.strip()) for x in cf_raw.split(',')]
            res = {"项目": _+1}
            if "NPV" in targets_to_show: res["NPV (万元)"] = compute_npv(row['I'], cf_series, row['r'], int(row['n']))
            if "IRR" in targets_to_show:
                irr = compute_irr(row['I'], cf_series, int(row['n']))
                res["IRR (%)"] = irr*100 if not np.isnan(irr) else np.nan
            if "LCOH" in targets_to_show:
                res["LCOH (元/kg)"] = compute_lcoh(row['I'], row['r'], int(row['n']), row['C_op'], row['Q'])
            all_results.append(res)
        st.subheader("批量计算结果")
        st.dataframe(pd.DataFrame(all_results))
    else:
        st.info("请上传文件")

# ---------------------------- 敏感性分析 ----------------------------
st.header("📈 敏感性分析")
tab1, tab2, tab3 = st.tabs(["单因素分析", "双因素分析", "全局敏感性 (Sobol)"])

def get_base_params():
    p = st.session_state.params
    cf_base = np.mean(p['cf_series']) if not np.isscalar(p['cf_series']) else p['cf_series']
    return {'I': p['I'], 'r': p['r_base'], 'n': p['n_base'], 'C_op': p['C_op'], 'Q': p['Q'], 'cf': cf_base}

def update_param(pname, new_val, base):
    I, r, n, C_op, Q, cf = base['I'], base['r'], base['n'], base['C_op'], base['Q'], base['cf']
    if "初始投资" in pname: I = new_val
    elif "年净现金流" in pname: cf = new_val
    elif "折现率" in pname: r = new_val
    elif "项目寿命" in pname: n = int(max(1, round(new_val)))
    elif "年运营成本" in pname: C_op = new_val
    elif "年制氢量" in pname: Q = new_val
    return I, r, n, C_op, Q, cf

# 通用图表设置
plt.rcParams.update({'figure.dpi': 100, 'figure.figsize': (10, 5)})

# ---------- 单因素分析 ----------
with tab1:
    st.subheader("单因素敏感性分析 (Tornado Chart & Table)")
    if input_mode != "手动输入":
        st.info("仅支持手动输入模式")
    else:
        target = st.selectbox("选择分析目标", targets_to_show, key="single_target")
        base = get_base_params()
        param_dict = {
            "初始投资 I (万元)": base['I'], "年净现金流 (万元)": base['cf'],
            "折现率 r": base['r'], "项目寿命 n (年)": base['n'],
            "年运营成本 C_op (万元)": base['C_op'], "年制氢量 Q (kg)": base['Q']
        }
        relevant = get_relevant_params(target)
        selected = st.multiselect(f"选择参数", relevant, default=relevant[:3])
        range_choice = st.selectbox("变动范围", ["±10%", "±20%", "±30%", "自定义"], index=1)
        if range_choice == "自定义":
            pct = st.number_input("变动百分比 (%)", value=20.0, step=1.0)
            levels = [-pct, pct]
        else:
            pct = int(range_choice.replace("±", "").replace("%", ""))
            levels = [-pct, pct]

        use_abs = st.checkbox("启用绝对数值变动", value=False)
        abs_dict = {}
        if use_abs:
            for pname in selected:
                c1, c2 = st.columns(2)
                with c1: down = st.number_input(f"{pname} 减少量", 0.0, key=f"abs_d_{pname}")
                with c2: up = st.number_input(f"{pname} 增加量", 0.0, key=f"abs_u_{pname}")
                abs_dict[pname] = (down, up)

        if st.button("运行单因素分析", key="run_single"):
            change_type = 'absolute' if use_abs else 'relative'
            results = []
            for pname in selected:
                base_val = param_dict[pname]
                low_ch, high_ch = levels if change_type == 'relative' else (-abs_dict[pname][0], abs_dict[pname][1])
                for tag, ch in [("Low", low_ch), ("High", high_ch)]:
                    new_val = base_val * (1 + ch/100) if change_type == 'relative' else base_val + ch
                    I_t, r_t, n_t, C_t, Q_t, cf_t = update_param(pname, new_val, base)
                    if target == "NPV": val = compute_npv(I_t, cf_t, r_t, n_t)
                    elif target == "IRR":
                        val = compute_irr(I_t, cf_t, n_t)
                        val = val*100 if not np.isnan(val) else np.nan
                    else: val = compute_lcoh(I_t, r_t, n_t, C_t, Q_t)
                    results.append({"Parameter": PARAM_EN.get(pname, pname), "Direction": tag, "Change": f"{ch:+.1f}%", target: val})
            df = pd.DataFrame(results)
            pivot = df.pivot(index="Parameter", columns="Direction", values=target).reset_index()
            pivot["Range"] = pivot["High"] - pivot["Low"]
            st.subheader("📋 Sensitivity Table")
            # 修复：只对数值列应用格式化，避免字符串列报错
            st.dataframe(pivot.style.format(subset=["Low", "High", "Range"], formatter="{:.4f}"))

            # Tornado chart (English)
            fig, ax = plt.subplots(figsize=(10, 6))
            params_en = pivot["Parameter"].tolist()
            ranges = pivot["Range"].tolist()
            colors = ['#d62728' if x > 0 else '#2ca02c' for x in ranges]
            ax.barh(params_en, ranges, color=colors, edgecolor='white')
            ax.set_xlabel(f"{target} Change Range", fontsize=12)
            ax.set_title(f"Single-factor Sensitivity for {target}", fontsize=14)
            ax.axvline(0, color='black', linewidth=0.8)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("*Bar length indicates impact; red = positive change, green = negative.*")

            csv = pivot.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, f"{target}_single.csv")

# ---------- 双因素分析 ----------
with tab2:
    st.subheader("双因素敏感性分析 (Heatmap & Grid Table)")
    if input_mode != "手动输入":
        st.info("仅支持手动输入模式")
    else:
        target2 = st.selectbox("分析目标", targets_to_show, key="dual_target")
        base2 = get_base_params()
        param_dict2 = {
            "初始投资 I (万元)": base2['I'], "年净现金流 (万元)": base2['cf'],
            "折现率 r": base2['r'], "项目寿命 n (年)": base2['n'],
            "年运营成本 C_op (万元)": base2['C_op'], "年制氢量 Q (kg)": base2['Q']
        }
        relevant2 = get_relevant_params(target2)
        col_a, col_b = st.columns(2)
        with col_a:
            param_x = st.selectbox("X-axis parameter", relevant2, index=0, format_func=lambda x: PARAM_EN.get(x, x))
            x_range = st.slider("X change (%)", -50, 50, (-20, 20), 5)
        with col_b:
            param_y = st.selectbox("Y-axis parameter", relevant2, index=min(1, len(relevant2)-1), format_func=lambda x: PARAM_EN.get(x, x))
            y_range = st.slider("Y change (%)", -50, 50, (-20, 20), 5)

        if st.button("Generate Heatmap & Table", key="run_dual"):
            xs = np.linspace(x_range[0]/100, x_range[1]/100, 10)
            ys = np.linspace(y_range[0]/100, y_range[1]/100, 10)
            grid = np.zeros((len(ys), len(xs)))
            for i, dy in enumerate(ys):
                for j, dx in enumerate(xs):
                    I_t, r_t, n_t, C_t, Q_t, cf_t = base2['I'], base2['r'], base2['n'], base2['C_op'], base2['Q'], base2['cf']
                    for pname, chg in [(param_x, dx), (param_y, dy)]:
                        new_val = param_dict2[pname] * (1 + chg)
                        I_t, r_t, n_t, C_t, Q_t, cf_t = update_param(pname, new_val, {'I':I_t,'r':r_t,'n':n_t,'C_op':C_t,'Q':Q_t,'cf':cf_t})
                    if target2 == "NPV": grid[i,j] = compute_npv(I_t, cf_t, r_t, n_t)
                    elif target2 == "IRR":
                        irr_t = compute_irr(I_t, cf_t, n_t)
                        grid[i,j] = irr_t*100 if not np.isnan(irr_t) else np.nan
                    else: grid[i,j] = compute_lcoh(I_t, r_t, n_t, C_t, Q_t)
            # Data table (所有单元格均为数值，可直接格式化)
            df_grid = pd.DataFrame(grid,
                                   index=[f"{y*100:+.0f}%" for y in ys],
                                   columns=[f"{x*100:+.0f}%" for x in xs])
            df_grid.index.name = f"{PARAM_EN.get(param_y, param_y)} change"
            df_grid.columns.name = f"{PARAM_EN.get(param_x, param_x)} change"
            st.subheader("📋 Grid Table")
            st.dataframe(df_grid.style.format("{:.4f}"))

            # Heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            c = ax.contourf(xs*100, ys*100, grid, levels=15, cmap='RdYlGn')
            ax.set_xlabel(f"{PARAM_EN.get(param_x, param_x)} change (%)", fontsize=11)
            ax.set_ylabel(f"{PARAM_EN.get(param_y, param_y)} change (%)", fontsize=11)
            ax.set_title(f"Two-factor Sensitivity for {target2}", fontsize=14)
            fig.colorbar(c, ax=ax, label=target2)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("*Darker colors indicate stronger changes in the target value.*")

# ---------- Sobol 分析 ----------
with tab3:
    st.subheader("Global Sensitivity Analysis (Sobol) – Chart & Table")
    if not SALIB_AVAILABLE:
        st.error("SALib library is required. Add SALib to requirements.txt")
    elif input_mode != "手动输入":
        st.info("仅支持手动输入模式")
    else:
        target3 = st.selectbox("Output indicator", targets_to_show, key="sobol_target")
        base3 = get_base_params()
        relevant3 = get_relevant_params(target3)
        selected_sobol = st.multiselect("Select parameters", relevant3, default=relevant3[:3])
        sample_N = st.number_input("Base samples N", value=256, min_value=64, step=64, help="Total runs = N*(2D+2)")
        if st.button("Run Sobol Analysis", key="run_sobol"):
            if len(selected_sobol) < 2:
                st.warning("Select at least two parameters")
            else:
                bounds = []
                for pname in selected_sobol:
                    if "初始投资" in pname: base_v = base3['I']
                    elif "净现金流" in pname: base_v = base3['cf']
                    elif "折现率" in pname: base_v = base3['r']
                    elif "寿命" in pname: base_v = base3['n']
                    elif "运营成本" in pname: base_v = base3['C_op']
                    else: base_v = base3['Q']
                    low = base_v * 0.7
                    high = base_v * 1.3
                    if "折现率" in pname:
                        low, high = max(0.01, low), min(0.5, high)
                    if "寿命" in pname:
                        low, high = max(1, int(low)), max(2, int(high))
                    bounds.append([low, high])
                problem = {'num_vars': len(selected_sobol), 'names': selected_sobol, 'bounds': bounds}
                param_values = saltelli.sample(problem, sample_N, calc_second_order=False)
                Y = np.zeros(param_values.shape[0])
                for i, row in enumerate(param_values):
                    I_t, r_t, n_t, C_t, Q_t, cf_t = base3['I'], base3['r'], base3['n'], base3['C_op'], base3['Q'], base3['cf']
                    for j, pname in enumerate(selected_sobol):
                        val = row[j]
                        if "初始投资" in pname: I_t = val
                        elif "净现金流" in pname: cf_t = val
                        elif "折现率" in pname: r_t = val
                        elif "寿命" in pname: n_t = int(max(1, round(val)))
                        elif "运营成本" in pname: C_t = val
                        else: Q_t = val
                    if target3 == "NPV": Y[i] = compute_npv(I_t, cf_t, r_t, n_t)
                    elif target3 == "IRR":
                        irr = compute_irr(I_t, cf_t, n_t)
                        Y[i] = irr*100 if not np.isnan(irr) else 0
                    else: Y[i] = compute_lcoh(I_t, r_t, n_t, C_t, Q_t)
                Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)

                # Table
                df_si = pd.DataFrame({
                    "Parameter": [PARAM_EN.get(p, p) for p in selected_sobol],
                    "First-order (S1)": Si['S1'],
                    "Total-effect (ST)": Si['ST']
                })
                st.subheader("📋 Sobol Indices Table")
                # 修复：只格式化数值列
                st.dataframe(df_si.style.format(subset=["First-order (S1)", "Total-effect (ST)"], formatter="{:.4f}"))

                # Bar chart
                fig, ax = plt.subplots(figsize=(8, 5))
                x = np.arange(len(selected_sobol))
                w = 0.35
                ax.bar(x - w/2, Si['S1'], w, label='S1 (Main effect)', color='#1f77b4', edgecolor='white')
                ax.bar(x + w/2, Si['ST'], w, label='ST (Total effect)', color='#ff7f0e', edgecolor='white')
                ax.set_xticks(x)
                ax.set_xticklabels([PARAM_EN.get(p, p) for p in selected_sobol], rotation=30, ha='right')
                ax.set_ylabel("Sensitivity Index", fontsize=12)
                ax.set_title(f"Sobol Sensitivity for {target3}", fontsize=14)
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                st.markdown("*S1 measures individual contribution; ST includes interactions. Large gap = strong interaction.*")

st.sidebar.markdown("---")
st.sidebar.caption("氢能项目经济性分析平台 v1.0")
