import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from io import BytesIO

# ---------- matplotlib 中文支持 ----------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 财务函数 ----------
try:
    import numpy_financial as npf
    IRR_FUNC = npf.irr
except ImportError:
    from scipy.optimize import newton
    def IRR_FUNC(cashflows):
        cashflows = np.asarray(cashflows, dtype=float)
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

# ---------- 核心公式 ----------
def crf(r, n):
    return (r * (1 + r)**n) / ((1 + r)**n - 1) if r != 0 else 1/n

def npv(I, cf, r, n):
    t = np.arange(1, n + 1)
    if np.isscalar(cf):
        cfs = np.full(n, cf, dtype=float)
    else:
        cfs = np.asarray(cf, dtype=float).ravel()
        if len(cfs) < n:
            cfs = np.pad(cfs, (0, n - len(cfs)), constant_values=0)
        else:
            cfs = cfs[:n]
    return np.sum(cfs / (1 + r)**t) - I

def irr(I, cf, n):
    if np.isscalar(cf):
        cfs = np.full(n, cf, dtype=float)
    else:
        cfs = np.asarray(cf, dtype=float).ravel()
        if len(cfs) < n:
            cfs = np.pad(cfs, (0, n - len(cfs)), constant_values=0)
        else:
            cfs = cfs[:n]
    return IRR_FUNC(np.insert(cfs, 0, -I))

def lcoh(I, r, n, C_op, Q):
    if Q == 0:
        return np.nan
    return (I * crf(r, n) + C_op) / Q

def lcoe(I, r, n, C_op, Q_gen):
    if Q_gen == 0:
        return np.nan
    return (I * crf(r, n) + C_op) / Q_gen

def loan_schedule(principal, annual_rate, years, method='等额本息'):
    if method == '等额本息':
        if annual_rate == 0:
            annual_payment = principal / years
            interests = [0] * years
        else:
            annual_payment = principal * (annual_rate * (1 + annual_rate)**years) / ((1 + annual_rate)**years - 1)
            interests = []
            balance = principal
            for _ in range(years):
                interest = balance * annual_rate
                principal_paid = annual_payment - interest
                balance -= principal_paid
                interests.append(interest)
        return [annual_payment] * years, interests
    else:
        annual_principal = principal / years
        payments = []
        interests = []
        balance = principal
        for _ in range(years):
            interest = balance * annual_rate
            payments.append(annual_principal + interest)
            interests.append(interest)
            balance -= annual_principal
        return payments, interests

# ---------- 完整项目现金流计算 ----------
def compute_full_project(params):
    I = params['I']
    r = params['r_base']
    n = params['n_base']
    Q = params['Q']
    C_op = params['C_op']
    unit_scale = params.get('unit_scale', 1.0)

    # 现金流基础序列
    if params.get('use_advanced_cf') and params.get('rev_items') is not None:
        rev_items = params['rev_items']
        cost_items = params['cost_items']
        total_rev = np.zeros(n)
        total_cost = np.zeros(n)
        for item in rev_items:
            amt = item['amount']
            for t in range(n):
                total_rev[t] += amt * (1 + item['growth'])**t
        for item in cost_items:
            amt = item['amount']
            for t in range(n):
                total_cost[t] += amt * (1 + item['growth'])**t
        cf_series = total_rev - total_cost
    else:
        cf_val = params.get('cf_series', 300.0)
        if np.isscalar(cf_val):
            cf_series = np.full(n, cf_val, dtype=float)
        else:
            cf_series = np.asarray(cf_val, dtype=float).ravel()
            if len(cf_series) < n:
                cf_series = np.pad(cf_series, (0, n - len(cf_series)), constant_values=0)
            else:
                cf_series = cf_series[:n]

    # 融资利息扣除
    if params.get('use_finance'):
        loan_ratio = params.get('loan_ratio', 0.0)
        loan_rate = params.get('loan_rate', 0.0)
        loan_years = params.get('loan_years', 0)
        loan_amount = I * loan_ratio
        if loan_amount > 0:
            _, interest_list = loan_schedule(loan_amount, loan_rate, loan_years,
                                             method=params.get('repay_method', '等额本息'))
            full_interests = np.zeros(n)
            full_interests[:loan_years] = interest_list
            cf_series = cf_series - full_interests

    # 替换成本
    if params.get('use_replacement') and params.get('replacements'):
        for yr, cost in params['replacements']:
            if 0 < yr <= n:
                cf_series[yr-1] -= cost

    # 碳收益 (carbon_params 现在为列表 [ef, cp, gcp, agg])
    if params.get('use_carbon') and params.get('carbon_params'):
        ef, cp, gcp, agg = params['carbon_params']
        carbon_rev = agg * ef / 1000 * cp + agg * gcp / 10000
        cf_series = cf_series + carbon_rev

    cf_series = np.asarray(cf_series, dtype=float).ravel()
    if len(cf_series) < n:
        cf_series = np.pad(cf_series, (0, n - len(cf_series)), constant_values=0)

    return I, r, n, Q, C_op, cf_series[:n]

# ---------- 页面设置 ----------
st.set_page_config(page_title="项目经济性分析平台", layout="wide")
st.title("项目成本计算与敏感性分析平台")

# ---------- 侧边栏 ----------
st.sidebar.header("📌 分析方法选择")
analysis_scope = st.sidebar.selectbox("选择指标数量", ["单个方法", "两种方法", "三种方法"], index=1)
all_targets = ["NPV", "IRR", "LCOH"]
if analysis_scope == "单个方法":
    selected_targets = [st.sidebar.selectbox("选择指标", all_targets, index=0)]
elif analysis_scope == "两种方法":
    selected_targets = st.sidebar.multiselect("选择两个指标", all_targets, default=["NPV", "IRR"], max_selections=2)
else:
    selected_targets = st.sidebar.multiselect("选择指标", all_targets, default=all_targets)
st.session_state['selected_targets'] = selected_targets if selected_targets else all_targets

# ========== 新增：全局单位选择 ==========
unit_choice = st.sidebar.selectbox("💲 金额单位", ["万元", "亿元"], index=0)
UNIT_SCALE = 10000.0 if unit_choice == "亿元" else 1.0
unit_label = "亿元" if unit_choice == "亿元" else "万元"
# =====================================

st.sidebar.header("⚙️ 高级功能开关")
use_advanced_cf = st.sidebar.checkbox("现金流分项构建器（收益/支出明细）", value=False)
use_carbon = st.sidebar.checkbox("碳排放与碳收益计算", value=False)
use_finance = st.sidebar.checkbox("融资结构（贷款利息）", value=False)
use_replacement = st.sidebar.checkbox("大修/替换成本时间线", value=False)
use_lcoe = st.sidebar.checkbox("计算LCOE（度电成本）", value=False)
use_multi_scenario = st.sidebar.checkbox("多方案对比", value=False)
use_breakeven = st.sidebar.checkbox("盈亏平衡分析（需启用双方案对比）", value=False)
use_matrix = st.sidebar.checkbox("多场景矩阵分析", value=False)

# ---------- 主界面：数据输入 ----------
st.header("📥 数据输入")
input_mode = st.radio("输入方式", ["手动输入", "上传文件 (CSV/Excel)"], horizontal=True)

if 'params' not in st.session_state:
    st.session_state.params = {}

if input_mode == "手动输入":
    has_lcoh = "LCOH" in st.session_state.selected_targets
    col1, col2, col3 = st.columns(3)
    with col1:
        I_raw = st.number_input(f"初始投资 I ({unit_label})", value=1000.0, step=100.0)
        I = I_raw * UNIT_SCALE
    with col2:
        r_base = st.number_input("基准折现率 r", value=0.08, step=0.01, format="%.3f")
    with col3:
        n_base = st.number_input("项目寿命期 n (年)", value=20, min_value=1, step=1)

    if has_lcoh or use_lcoe:
        c4, c5 = st.columns(2)
        with c4:
            Q = st.number_input("年制氢量 Q (kg/年)", value=50000.0, step=1000.0) if has_lcoh else st.number_input("年发电量 (万kWh)", value=5000.0, step=100.0)
        with c5:
            C_op_raw = st.number_input(f"年运营成本 ({unit_label}/年)", value=200.0, step=10.0)
            C_op = C_op_raw * UNIT_SCALE
    else:
        Q = 1.0
        C_op = 0.0

    # 现金流构建
    rev_items = None
    cost_items = None
    if not use_advanced_cf:
        st.subheader("净现金流设置")
        cf_mode = st.radio("现金流类型", ["等额年金（各年相同）", "逐年输入"], horizontal=True)
        if cf_mode == "等额年金（各年相同）":
            cf_val_raw = st.number_input(f"年均净现金流 ({unit_label})", value=300.0, step=10.0)
            cf_series = cf_val_raw * UNIT_SCALE
        else:
            cf_str = st.text_area(f"各年净现金流，逗号分隔（{unit_label}）", "300,300,300,300,300")
            try:
                cf_series = [float(x.strip()) * UNIT_SCALE for x in cf_str.split(",") if x.strip() != ""]
            except:
                st.error("现金流格式错误，将使用默认值300")
                cf_series = 300.0 * UNIT_SCALE
    else:
        st.subheader("💵 现金流分项构建器")
        st.caption("输入各项收益与支出，自动计算每年净现金流")
        st.markdown("**收益项**")
        num_rev = st.number_input("收益项数量", min_value=1, value=2, step=1)
        rev_items = []
        for i in range(num_rev):
            cols = st.columns(3)
            with cols[0]:
                name = st.text_input(f"收益{i+1}名称", f"电力销售收入" if i==0 else f"碳收益", key=f"rev_name_{i}")
            with cols[1]:
                amount_raw = st.number_input(f"年金额 ({unit_label})", value=500.0, step=10.0, key=f"rev_amt_{i}")
                amount = amount_raw * UNIT_SCALE
            with cols[2]:
                growth = st.number_input(f"年增长率 (%)", value=0.0, step=0.1, key=f"rev_growth_{i}") / 100
            rev_items.append({'name': name, 'amount': amount, 'growth': growth})

        st.markdown("**支出项**")
        num_cost = st.number_input("支出项数量", min_value=1, value=3, step=1)
        cost_items = []
        for i in range(num_cost):
            cols = st.columns(3)
            with cols[0]:
                name = st.text_input(f"支出{i+1}名称", f"运维费" if i==0 else (f"外购电费" if i==1 else f"其他"), key=f"cost_name_{i}")
            with cols[1]:
                amount_raw = st.number_input(f"年金额 ({unit_label})", value=200.0, step=10.0, key=f"cost_amt_{i}")
                amount = amount_raw * UNIT_SCALE
            with cols[2]:
                growth = st.number_input(f"年增长率 (%)", value=0.0, step=0.1, key=f"cost_growth_{i}") / 100
            cost_items.append({'name': name, 'amount': amount, 'growth': growth})

        def generate_cf_from_items(revs, costs, n_years):
            total_rev = np.zeros(n_years)
            total_cost = np.zeros(n_years)
            for item in revs:
                amt = item['amount']
                for t in range(n_years):
                    total_rev[t] += amt * (1 + item['growth'])**t
            for item in costs:
                amt = item['amount']
                for t in range(n_years):
                    total_cost[t] += amt * (1 + item['growth'])**t
            return total_rev - total_cost
        cf_series = generate_cf_from_items(rev_items, cost_items, n_base)

    # 融资参数
    loan_ratio = 0.0
    loan_rate = 0.0
    loan_years = 0
    repay_method = '等额本息'
    if use_finance:
        st.subheader("🏦 融资结构")
        loan_ratio = st.slider("贷款比例 (%)", 0, 100, 70) / 100
        loan_rate = st.number_input("贷款年利率 (%)", value=4.2, step=0.1) / 100
        loan_years = st.number_input("贷款年限", min_value=1, value=min(15, n_base))
        repay_method = st.selectbox("还款方式", ["等额本息", "等额本金"])

    # 替换成本
    replacements = []
    if use_replacement:
        st.subheader("🔧 大修/替换成本时间线")
        replace_count = st.number_input("替换事件数量", min_value=0, value=1, step=1)
        for i in range(replace_count):
            c1, c2 = st.columns(2)
            with c1:
                year = st.number_input(f"事件{i+1}年份", min_value=1, max_value=n_base, value=10, key=f"rep_year_{i}")
            with c2:
                cost_raw = st.number_input(f"事件{i+1}金额 ({unit_label})", value=500.0, step=10.0, key=f"rep_cost_{i}")
                cost = cost_raw * UNIT_SCALE
            replacements.append((year, cost))

    # 碳排放参数 (改为列表，方便后续修改)
    carbon_params = None
    if use_carbon:
        st.subheader("🌱 碳排放与碳收益")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            emission_factor = st.number_input("电网排放因子 (kgCO₂/kWh)", value=0.58, step=0.01)
        with col_c2:
            carbon_price = st.number_input("碳价 (元/tCO₂)", value=50.0, step=5.0)
        with col_c3:
            green_cert_price = st.number_input("绿证价格 (元/个)", value=7.76, step=0.5)
        annual_green_gen = st.number_input("年自发绿电量 (万kWh)", value=6000.0, step=100.0)
        carbon_params = [emission_factor, carbon_price, green_cert_price, annual_green_gen]

    # 保存完整参数
    st.session_state.params = {
        'I': I,
        'r_base': r_base,
        'n_base': n_base,
        'Q': Q,
        'C_op': C_op,
        'cf_series': cf_series if not use_advanced_cf else None,
        'use_advanced_cf': use_advanced_cf,
        'rev_items': rev_items,
        'cost_items': cost_items,
        'use_finance': use_finance,
        'loan_ratio': loan_ratio,
        'loan_rate': loan_rate,
        'loan_years': loan_years,
        'repay_method': repay_method,
        'use_replacement': use_replacement,
        'replacements': replacements,
        'use_carbon': use_carbon,
        'carbon_params': carbon_params,
        'use_lcoe': use_lcoe,
        'unit_scale': UNIT_SCALE,    # 记录单位缩放
    }

# ---------- 文件上传 ----------
elif input_mode == "上传文件 (CSV/Excel)":
    uploaded_file = st.file_uploader("上传文件", type=['csv', 'xlsx'])
    st.markdown("文件需包含列: `I`, `r`, `n`, `Q`, `C_op`, `cf_type`, `cf_values`")
    st.download_button("📥 下载标准模板", data="I,r,n,Q,C_op,cf_type,cf_values\n1000,0.08,20,50000,200,uniform,300",
                       file_name="template.csv")
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            required = {'I', 'r', 'n', 'Q', 'C_op', 'cf_type', 'cf_values'}
            if required.issubset(df.columns):
                st.session_state.uploaded_df = df
                st.success("文件上传成功！")
            else:
                st.error("文件缺少必要列")
        except Exception as e:
            st.error(f"读取文件出错: {e}")
    else:
        st.session_state.uploaded_df = None

# ---------- 基准计算结果 ----------
st.header("📊 基准计算结果")
targets_to_show = st.session_state.get('selected_targets', ["NPV", "IRR", "LCOH"])

if input_mode == "手动输入":
    base_params = deepcopy(st.session_state.params)
    I, r, n, Q, C_op, cf = compute_full_project(base_params)

    npv_val = npv(I, cf, r, n)
    irr_val = irr(I, cf, n)
    lcoh_val = lcoh(I, r, n, C_op, Q) if "LCOH" in targets_to_show else None
    lcoe_val = lcoe(I, r, n, C_op, Q) if use_lcoe else None

    cols = st.columns(len(targets_to_show) + (1 if use_lcoe else 0))
    idx = 0
    for target in targets_to_show:
        if target == "NPV":
            cols[idx].metric("NPV (万元)", f"{npv_val:.2f}")
        elif target == "IRR":
            cols[idx].metric("IRR (%)", f"{irr_val*100:.2f}" if not np.isnan(irr_val) else "无解")
        elif target == "LCOH":
            cols[idx].metric("LCOH (元/kg)", f"{lcoh_val:.4f}")
        idx += 1
    if use_lcoe:
        cols[idx].metric("LCOE (元/kWh)", f"{lcoe_val:.4f}")

    with st.expander("📋 查看输入数据明细"):
        data = [
            ["初始投资 I", f"{I} 万元"],
            ["基准折现率 r", f"{r*100:.2f}%"],
            ["项目寿命期 n", f"{n} 年"],
        ]
        if "LCOH" in targets_to_show or use_lcoe:
            data.append(["年生产量", f"{Q} kg/年" if "LCOH" in targets_to_show else f"{Q} 万kWh/年"])
            data.append(["年运营成本 C_op", f"{C_op} 万元/年"])
        if np.isscalar(cf):
            data.append(["各年净现金流", f"年均 {cf} 万元"])
        else:
            data.append(["各年净现金流", f"逐年数据"])
        st.table(pd.DataFrame(data, columns=["项目", "数值"]))

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
            if "NPV" in targets_to_show: res["NPV (万元)"] = npv(row['I'], cf_series, row['r'], int(row['n']))
            if "IRR" in targets_to_show:
                ir = irr(row['I'], cf_series, int(row['n']))
                res["IRR (%)"] = ir*100 if not np.isnan(ir) else np.nan
            if "LCOH" in targets_to_show: res["LCOH (元/kg)"] = lcoh(row['I'], row['r'], int(row['n']), row['C_op'], row['Q'])
            all_results.append(res)
        st.subheader("批量计算结果")
        st.dataframe(pd.DataFrame(all_results))
    else:
        st.info("请上传文件")

# ---------- 敏感性分析 ----------
st.header("📈 敏感性分析")

def build_all_param_specs(base_params):
    specs = []
    # 固定基本参数
    specs.append(('I', '初始投资 I (万元)', base_params['I'], lambda p, v: p.update({'I': v})))
    specs.append(('r_base', '折现率 r', base_params['r_base'], lambda p, v: p.update({'r_base': v})))
    specs.append(('n_base', '项目寿命 n (年)', base_params['n_base'], lambda p, v: p.update({'n_base': int(max(1, round(v)))})))
    specs.append(('C_op', '年运营成本 C_op (万元)', base_params['C_op'], lambda p, v: p.update({'C_op': v})))
    if base_params.get('use_lcoe'):
        specs.append(('Q', '年发电量 (万kWh)', base_params['Q'], lambda p, v: p.update({'Q': v})))
    else:
        specs.append(('Q', '年制氢量 Q (kg)', base_params['Q'], lambda p, v: p.update({'Q': v})))
    # 高级现金流分项
    if base_params.get('use_advanced_cf') and base_params.get('rev_items'):
        for i, item in enumerate(base_params['rev_items']):
            name = item['name']
            specs.append((f'rev_{i}_amount', f'收益-{name} 金额 (万元)', item['amount'],
                          lambda p, v, idx=i: p['rev_items'][idx].update({'amount': v})))
            specs.append((f'rev_{i}_growth', f'收益-{name} 增长率', item['growth'],
                          lambda p, v, idx=i: p['rev_items'][idx].update({'growth': v})))
    if base_params.get('use_advanced_cf') and base_params.get('cost_items'):
        for i, item in enumerate(base_params['cost_items']):
            name = item['name']
            specs.append((f'cost_{i}_amount', f'支出-{name} 金额 (万元)', item['amount'],
                          lambda p, v, idx=i: p['cost_items'][idx].update({'amount': v})))
            specs.append((f'cost_{i}_growth', f'支出-{name} 增长率', item['growth'],
                          lambda p, v, idx=i: p['cost_items'][idx].update({'growth': v})))
    # 融资参数
    if base_params.get('use_finance'):
        specs.append(('loan_ratio', '贷款比例', base_params['loan_ratio'], lambda p, v: p.update({'loan_ratio': v})))
        specs.append(('loan_rate', '贷款年利率', base_params['loan_rate'], lambda p, v: p.update({'loan_rate': v})))
        specs.append(('loan_years', '贷款年限', base_params['loan_years'], lambda p, v: p.update({'loan_years': int(max(1, round(v)))})))
    # 替换成本
    if base_params.get('use_replacement') and base_params.get('replacements'):
        for i, (yr, cost) in enumerate(base_params['replacements']):
            specs.append((f'replace_{i}_cost', f'替换事件{i+1}金额 (万元)', cost,
                          lambda p, v, idx=i: p['replacements'].__setitem__(idx, (p['replacements'][idx][0], v))))
    # 碳排放参数 (列表，直接索引赋值)
    if base_params.get('use_carbon') and base_params.get('carbon_params'):
        cp = base_params['carbon_params']  # [ef, cp, gcp, agg]
        specs.append(('emission_factor', '电网排放因子 (kgCO₂/kWh)', cp[0],
                      lambda p, v: p['carbon_params'].__setitem__(0, v)))
        specs.append(('carbon_price', '碳价 (元/tCO₂)', cp[1],
                      lambda p, v: p['carbon_params'].__setitem__(1, v)))
        specs.append(('green_cert_price', '绿证价格 (元/个)', cp[2],
                      lambda p, v: p['carbon_params'].__setitem__(2, v)))
        specs.append(('annual_green_gen', '年自发绿电量 (万kWh)', cp[3],
                      lambda p, v: p['carbon_params'].__setitem__(3, v)))
    return specs

if input_mode == "手动输入":
    base_params_original = deepcopy(st.session_state.params)
    all_specs = build_all_param_specs(base_params_original)
    all_display_names = [spec[1] for spec in all_specs]
    display_to_key = {spec[1]: spec[0] for spec in all_specs}
    key_to_base = {spec[0]: spec[2] for spec in all_specs}
    key_to_updater = {spec[0]: spec[3] for spec in all_specs}
else:
    all_display_names = []
    display_to_key = {}
    key_to_base = {}
    key_to_updater = {}

tab1, tab2, tab3 = st.tabs(["单因素分析", "双因素分析", "全局敏感性 (Sobol)"])

# ---------- 单因素 ----------
with tab1:
    st.subheader("单因素敏感性分析（龙卷风图）")
    if input_mode != "手动输入":
        st.info("仅支持手动输入模式")
    else:
        target = st.selectbox("分析目标", targets_to_show + (["LCOE"] if use_lcoe else []), key="single_target")
        selected_display = st.multiselect("选择参数", all_display_names,
                                          default=all_display_names[:min(3, len(all_display_names))],
                                          key="single_multiselect")
        range_choice = st.selectbox("变动范围", ["±10%", "±20%", "±30%", "自定义"], index=1)
        if range_choice == "自定义":
            pct = st.number_input("变动百分比 (%)", value=20.0, step=1.0)
            levels = [-pct, pct]
        else:
            pct = int(range_choice.replace("±", "").replace("%", ""))
            levels = [-pct, pct]

        use_abs = st.checkbox("启用绝对数值变动", False)
        abs_dict = {}
        if use_abs:
            for disp in selected_display:
                key = display_to_key[disp]
                base_val = key_to_base[key]
                c1, c2 = st.columns(2)
                with c1: down = st.number_input(f"{disp} 减少量", value=0.0, key=f"abs_d_{key}")
                with c2: up = st.number_input(f"{disp} 增加量", value=0.0, key=f"abs_u_{key}")
                abs_dict[key] = (down, up)

        if st.button("运行单因素分析", key="run_single"):
            change_type = 'absolute' if use_abs else 'relative'
            results = []
            for disp in selected_display:
                key = display_to_key[disp]
                base_val = key_to_base[key]
                if change_type == 'relative':
                    low_ch, high_ch = levels
                else:
                    down, up = abs_dict.get(key, (0.0, 0.0))
                    low_ch, high_ch = -down, up
                for tag, ch in [("Low", low_ch), ("High", high_ch)]:
                    new_val = base_val * (1 + ch/100) if change_type == 'relative' else base_val + ch
                    p = deepcopy(base_params_original)
                    updater = key_to_updater.get(key)
                    if updater:
                        updater(p, new_val)
                    I_t, r_t, n_t, Q_t, C_t, cf_t = compute_full_project(p)
                    if target == "NPV": val = npv(I_t, cf_t, r_t, n_t)
                    elif target == "IRR":
                        val = irr(I_t, cf_t, n_t)
                        val = val*100 if not np.isnan(val) else np.nan
                    elif target == "LCOH": val = lcoh(I_t, r_t, n_t, C_t, Q_t)
                    else: val = lcoe(I_t, r_t, n_t, C_t, Q_t)
                    results.append({"参数": disp, "方向": tag, "变动": f"{ch:+.1f}%", target: val})
            if results:
                df = pd.DataFrame(results)
                pivot = df.pivot(index="参数", columns="方向", values=target).reset_index()
                pivot_clean = pivot.dropna(subset=["Low", "High"])
                if not pivot_clean.empty:
                    pivot_clean["变化范围"] = pivot_clean["High"] - pivot_clean["Low"]
                    st.subheader("📋 敏感性数据表")
                    st.dataframe(pivot_clean.style.format(subset=["Low", "High", "变化范围"], formatter="{:.4f}"))
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ranges = pivot_clean["变化范围"]
                    colors = ['#d62728' if x > 0 else '#2ca02c' for x in ranges]
                    ax.barh(pivot_clean["参数"], ranges, color=colors)
                    ax.set_xlabel(f"{target} Change Range")
                    ax.set_title(f"Single-factor Sensitivity for {target}")
                    ax.axvline(0, color='black', linewidth=0.8)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("无有效数据可绘图")
            else:
                st.error("没有分析结果")

# ---------- 双因素 ----------
with tab2:
    st.subheader("双因素敏感性分析（热力图）")
    if input_mode != "手动输入":
        st.info("仅支持手动输入模式")
    else:
        target2 = st.selectbox("分析目标", targets_to_show + (["LCOE"] if use_lcoe else []), key="dual_target")
        col_a, col_b = st.columns(2)
        with col_a:
            param_x_display = st.selectbox("X轴参数", all_display_names, index=0, key="dual_param_x")
            x_range = st.slider("X变动范围 (%)", -50, 50, (-20, 20), 5)
        with col_b:
            param_y_display = st.selectbox("Y轴参数", all_display_names,
                                           index=min(1, len(all_display_names)-1) if len(all_display_names) > 1 else 0,
                                           key="dual_param_y")
            y_range = st.slider("Y变动范围 (%)", -50, 50, (-20, 20), 5)

        if st.button("生成热力图及表格", key="run_dual"):
            key_x = display_to_key[param_x_display]
            key_y = display_to_key[param_y_display]
            base_x = key_to_base[key_x]
            base_y = key_to_base[key_y]
            updater_x = key_to_updater[key_x]
            updater_y = key_to_updater[key_y]

            xs = np.linspace(x_range[0]/100, x_range[1]/100, 10)
            ys = np.linspace(y_range[0]/100, y_range[1]/100, 10)
            grid = np.zeros((len(ys), len(xs)))
            has_nan = False
            for i, dy in enumerate(ys):
                for j, dx in enumerate(xs):
                    p = deepcopy(base_params_original)
                    new_x = base_x * (1 + dx)
                    new_y = base_y * (1 + dy)
                    updater_x(p, new_x)
                    updater_y(p, new_y)
                    I_t, r_t, n_t, Q_t, C_t, cf_t = compute_full_project(p)
                    if target2 == "NPV": grid[i,j] = npv(I_t, cf_t, r_t, n_t)
                    elif target2 == "IRR":
                        irr_t = irr(I_t, cf_t, n_t)
                        grid[i,j] = irr_t*100 if not np.isnan(irr_t) else np.nan
                        if np.isnan(grid[i,j]): has_nan = True
                    elif target2 == "LCOH": grid[i,j] = lcoh(I_t, r_t, n_t, C_t, Q_t)
                    else: grid[i,j] = lcoe(I_t, r_t, n_t, C_t, Q_t)
            if has_nan:
                st.warning("部分组合的IRR无解，热力图中将NaN替换为0显示。")
                grid = np.nan_to_num(grid, nan=0.0)
            df_grid = pd.DataFrame(grid, index=[f"{y*100:+.0f}%" for y in ys], columns=[f"{x*100:+.0f}%" for x in xs])
            df_grid.index.name = f"{param_y_display} 变化"
            df_grid.columns.name = f"{param_x_display} 变化"
            st.subheader("📋 网格数据表")
            st.dataframe(df_grid.style.format("{:.4f}"))
            fig, ax = plt.subplots()
            c = ax.contourf(xs*100, ys*100, grid, cmap='RdYlGn')
            ax.set_xlabel(f"{param_x_display} change (%)")
            ax.set_ylabel(f"{param_y_display} change (%)")
            ax.set_title(f"Two-factor Sensitivity for {target2}")
            fig.colorbar(c, ax=ax)
            st.pyplot(fig)

# ---------- Sobol ----------
with tab3:
    st.subheader("全局敏感性分析 (Sobol)")
    if not SALIB_AVAILABLE:
        st.error("需要安装 SALib 库")
    elif input_mode != "手动输入":
        st.info("仅支持手动输入模式")
    else:
        target3 = st.selectbox("输出指标", targets_to_show + (["LCOE"] if use_lcoe else []), key="sobol_target")
        selected_sobol_display = st.multiselect("选择参数", all_display_names,
                                                default=all_display_names[:min(3, len(all_display_names))],
                                                key="sobol_multiselect")
        sample_N = st.number_input("基础样本数 N", value=256, min_value=64, step=64, help="总运行次数 = N*(2D+2)")
        if st.button("运行 Sobol 分析", key="run_sobol"):
            if len(selected_sobol_display) < 2:
                st.warning("至少选择两个参数")
            else:
                selected_keys = [display_to_key[d] for d in selected_sobol_display]
                bounds = []
                for key in selected_keys:
                    base_v = key_to_base[key]
                    low = base_v * 0.7
                    high = base_v * 1.3
                    if key == 'r_base':
                        low, high = max(0.01, low), min(0.5, high)
                    elif key == 'n_base' or key == 'loan_years':
                        low, high = max(1, int(low)), max(2, int(high))
                    elif key == 'loan_ratio':
                        low, high = max(0.0, low), min(1.0, high)
                    bounds.append([low, high])

                problem = {'num_vars': len(selected_keys), 'names': selected_keys, 'bounds': bounds}
                param_values = saltelli.sample(problem, sample_N, calc_second_order=False)
                Y = np.zeros(param_values.shape[0])
                for i, row in enumerate(param_values):
                    p = deepcopy(base_params_original)
                    for j, key in enumerate(selected_keys):
                        val = row[j]
                        updater = key_to_updater[key]
                        updater(p, val)
                    I_t, r_t, n_t, Q_t, C_t, cf_t = compute_full_project(p)
                    if target3 == "NPV": Y[i] = npv(I_t, cf_t, r_t, n_t)
                    elif target3 == "IRR":
                        ir = irr(I_t, cf_t, n_t)
                        Y[i] = ir*100 if not np.isnan(ir) else 0.0
                    elif target3 == "LCOH": Y[i] = lcoh(I_t, r_t, n_t, C_t, Q_t)
                    else: Y[i] = lcoe(I_t, r_t, n_t, C_t, Q_t)
                if np.any(np.isnan(Y)):
                    st.error("计算结果包含NaN，请检查参数范围或指标计算。")
                else:
                    Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)
                    df_si = pd.DataFrame({
                        "参数": selected_sobol_display,
                        "一阶指数 S1": Si['S1'],
                        "总效应指数 ST": Si['ST']
                    })
                    st.subheader("📋 Sobol 敏感指数表")
                    st.dataframe(df_si.style.format(subset=["一阶指数 S1", "总效应指数 ST"], formatter="{:.4f}"))
                    fig, ax = plt.subplots()
                    x = np.arange(len(selected_sobol_display))
                    w = 0.35
                    ax.bar(x - w/2, Si['S1'], w, label='S1', color='#1f77b4')
                    ax.bar(x + w/2, Si['ST'], w, label='ST', color='#ff7f0e')
                    ax.set_xticks(x)
                    ax.set_xticklabels(selected_sobol_display, rotation=30)
                    ax.set_ylabel("Sensitivity Index")
                    ax.set_title(f"Sobol Sensitivity for {target3}")
                    ax.legend()
                    st.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.caption("项目经济性分析平台 v3.1")
