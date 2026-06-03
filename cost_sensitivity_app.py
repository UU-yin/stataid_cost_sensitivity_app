import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# ---------- 财务函数 ----------
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

# SALib 用于全局敏感性分析
try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False

# ---------- 核心公式 ----------
def crf(r, n):
    """资本回收系数"""
    return (r * (1 + r)**n) / ((1 + r)**n - 1) if r != 0 else 1/n

def npv(I, cf, r, n):
    """净现值：Σ CF_t/(1+r)^t - I"""
    t = np.arange(1, n + 1)
    cfs = np.full(n, cf) if np.isscalar(cf) else np.asarray(cf)[:n]
    return np.sum(cfs / (1 + r)**t) - I

def irr(I, cf, n):
    """内部收益率"""
    cfs = np.full(n, cf) if np.isscalar(cf) else np.asarray(cf)[:n]
    return IRR_FUNC(np.insert(cfs, 0, -I))

def lcoh(I, r, n, C_op, Q):
    """平准化制氢成本"""
    if Q == 0:
        return np.nan
    return (I * crf(r, n) + C_op) / Q

def lcoe(I, r, n, C_op, Q_gen):
    """平准化度电成本（元/kWh）"""
    if Q_gen == 0:
        return np.nan
    return (I * crf(r, n) + C_op) / Q_gen

# ---------- 辅助：融资还款计算 ----------
def loan_schedule(principal, annual_rate, years, method='等额本息'):
    """返回每年还款额和利息列表（简化：年还款，等额本息或等额本金）"""
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
    else:  # 等额本金
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

# 存放参数
if 'params' not in st.session_state:
    st.session_state.params = {}

# ---------- 手动输入 ----------
if input_mode == "手动输入":
    # 基本参数
    has_lcoh = "LCOH" in st.session_state.selected_targets
    col1, col2, col3 = st.columns(3)
    with col1:
        I = st.number_input("初始投资 I (万元)", value=1000.0, step=100.0)
    with col2:
        r_base = st.number_input("基准折现率 r", value=0.08, step=0.01, format="%.3f")
    with col3:
        n_base = st.number_input("项目寿命期 n (年)", value=20, min_value=1, step=1)

    if has_lcoh or use_lcoe:
        c4, c5 = st.columns(2)
        with c4:
            Q = st.number_input("年制氢量 Q (kg/年)", value=50000.0, step=1000.0) if has_lcoh else st.number_input("年发电量 (万kWh)", value=5000.0, step=100.0)
        with c5:
            C_op = st.number_input("年运营成本 (万元/年)", value=200.0, step=10.0)
    else:
        Q = 1.0
        C_op = 0.0

    # ---------- 现金流构建 ----------
    if not use_advanced_cf:
        # 简单模式
        st.subheader("净现金流设置")
        cf_mode = st.radio("现金流类型", ["等额年金（各年相同）", "逐年输入"], horizontal=True)
        if cf_mode == "等额年金（各年相同）":
            cf_val = st.number_input("年均净现金流 (万元)", value=300.0, step=10.0)
            cf_series = cf_val
        else:
            cf_str = st.text_area("各年净现金流，逗号分隔（万元）", "300,300,300,300,300")
            cf_series = [float(x.strip()) for x in cf_str.split(",") if x.strip() != ""]
    else:
        # 高级分项构建器
        st.subheader("💵 现金流分项构建器")
        st.caption("输入各项收益与支出，自动计算每年净现金流 = 总收益 - 总支出")

        # 收益项
        st.markdown("**收益项**")
        num_rev = st.number_input("收益项数量", min_value=1, value=2, step=1)
        rev_items = []
        for i in range(num_rev):
            cols = st.columns(3)
            with cols[0]:
                name = st.text_input(f"收益{i+1}名称", f"电力销售收入" if i==0 else f"碳收益", key=f"rev_name_{i}")
            with cols[1]:
                amount = st.number_input(f"年金额 (万元)", value=500.0, step=10.0, key=f"rev_amt_{i}")
            with cols[2]:
                growth = st.number_input(f"年增长率 (%)", value=0.0, step=0.1, key=f"rev_growth_{i}") / 100
            rev_items.append({'name': name, 'amount': amount, 'growth': growth})

        # 支出项
        st.markdown("**支出项**")
        num_cost = st.number_input("支出项数量", min_value=1, value=3, step=1)
        cost_items = []
        for i in range(num_cost):
            cols = st.columns(3)
            with cols[0]:
                name = st.text_input(f"支出{i+1}名称", f"运维费" if i==0 else (f"外购电费" if i==1 else f"其他"), key=f"cost_name_{i}")
            with cols[1]:
                amount = st.number_input(f"年金额 (万元)", value=200.0, step=10.0, key=f"cost_amt_{i}")
            with cols[2]:
                growth = st.number_input(f"年增长率 (%)", value=0.0, step=0.1, key=f"cost_growth_{i}") / 100
            cost_items.append({'name': name, 'amount': amount, 'growth': growth})

        # 生成现金流序列
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
        st.info("根据分项汇总，年均净现金流为 {:.2f} 万元".format(np.mean(cf_series)))

    # ---------- 融资结构 ----------
    if use_finance:
        st.subheader("🏦 融资结构")
        loan_ratio = st.slider("贷款比例 (%)", 0, 100, 70) / 100
        loan_rate = st.number_input("贷款年利率 (%)", value=4.2, step=0.1) / 100
        loan_years = st.number_input("贷款年限", min_value=1, value=min(15, n_base))
        repay_method = st.selectbox("还款方式", ["等额本息", "等额本金"])
        loan_amount = I * loan_ratio
        if loan_amount > 0:
            annual_payments, interest_list = loan_schedule(loan_amount, loan_rate, loan_years, repay_method)
            full_interests = np.zeros(n_base)
            full_interests[:loan_years] = interest_list
            if np.isscalar(cf_series):
                cf_series = np.full(n_base, cf_series)
            cf_series = np.asarray(cf_series) - np.array(full_interests)
            st.caption("已从净现金流中扣除各年利息支出。")

    # ---------- 大修/替换成本 ----------
    if use_replacement:
        st.subheader("🔧 大修/替换成本时间线")
        st.caption("定义特定年份的一次性支出（万元），例如储能第10年更换。")
        replace_count = st.number_input("替换事件数量", min_value=0, value=1, step=1)
        replacements = []
        for i in range(replace_count):
            c1, c2 = st.columns(2)
            with c1:
                year = st.number_input(f"事件{i+1}年份", min_value=1, max_value=n_base, value=10, key=f"rep_year_{i}")
            with c2:
                cost = st.number_input(f"事件{i+1}金额 (万元)", value=500.0, step=10.0, key=f"rep_cost_{i}")
            replacements.append((year, cost))
        if replacements and np.isscalar(cf_series):
            cf_series = np.full(n_base, cf_series)
        for yr, cst in replacements:
            cf_series[yr-1] -= cst

    # ---------- 碳排放与碳收益 ----------
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
        carbon_revenue = annual_green_gen * emission_factor / 1000 * carbon_price + annual_green_gen * green_cert_price / 10000
        st.caption(f"估算年碳收益约 {carbon_revenue:.2f} 万元")
        if not use_advanced_cf:
            if np.isscalar(cf_series):
                cf_series = np.full(n_base, cf_series)
            cf_series = np.asarray(cf_series) + carbon_revenue
            st.info("碳收益已自动并入各年净现金流。")

    # 保存参数
    st.session_state.params = {
        'I': I, 'r_base': r_base, 'n_base': n_base, 'Q': Q, 'C_op': C_op,
        'cf_series': cf_series,
        'use_advanced_cf': use_advanced_cf, 'use_carbon': use_carbon, 'use_finance': use_finance,
        'use_replacement': use_replacement, 'use_lcoe': use_lcoe,
        'rev_items': rev_items if use_advanced_cf else None,
        'cost_items': cost_items if use_advanced_cf else None,
        'carbon_params': (emission_factor, carbon_price, green_cert_price, annual_green_gen) if use_carbon else None
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
    p = st.session_state.params
    I, r_base, n_base, Q, C_op = p['I'], p['r_base'], p['n_base'], p['Q'], p['C_op']
    cf = p['cf_series']

    npv_val = npv(I, cf, r_base, n_base)
    irr_val = irr(I, cf, n_base)
    lcoh_val = lcoh(I, r_base, n_base, C_op, Q) if "LCOH" in targets_to_show else None
    lcoe_val = None
    if use_lcoe:
        lcoe_val = lcoe(I, r_base, n_base, C_op, Q)

    # 显示指标卡片
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

    # 输入数据明细
    with st.expander("📋 查看输入数据明细"):
        data = [
            ["初始投资 I", f"{I} 万元"],
            ["基准折现率 r", f"{r_base*100:.2f}%"],
            ["项目寿命期 n", f"{n_base} 年"],
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

# ---------- 多方案对比 ----------
if use_multi_scenario and input_mode == "手动输入":
    st.header("📊 多方案对比分析")
    num_cases = st.number_input("对比方案数量", min_value=2, value=2, step=1)
    case_params = []
    for i in range(num_cases):
        with st.expander(f"方案 {i+1} 参数"):
            cI = st.number_input(f"初始投资 (万元)", value=I, key=f"case_I_{i}")
            cr = st.number_input(f"折现率", value=r_base, key=f"case_r_{i}", format="%.3f")
            cn = st.number_input(f"项目寿命 (年)", value=n_base, key=f"case_n_{i}")
            c_cf_str = st.text_input(f"净现金流序列 (逗号分隔或单个数值)", 
                                     value="300" if np.isscalar(cf) else ",".join(map(str, np.asarray(cf)[:cn])),
                                     key=f"case_cf_{i}")
            try:
                c_cf_list = [float(x.strip()) for x in c_cf_str.split(",")]
                if len(c_cf_list) == 1:
                    c_cf = c_cf_list[0]
                else:
                    c_cf = np.array(c_cf_list[:cn])
            except:
                st.error("现金流格式错误")
                c_cf = 300
            case_params.append({'name': f'方案{i+1}', 'I': cI, 'r': cr, 'n': cn, 'cf': c_cf})
    
    if st.button("对比分析"):
        results = []
        for case in case_params:
            npv_val = npv(case['I'], case['cf'], case['r'], case['n'])
            irr_val = irr(case['I'], case['cf'], case['n'])
            results.append({
                "方案": case['name'],
                "NPV (万元)": npv_val,
                "IRR (%)": irr_val*100 if not np.isnan(irr_val) else np.nan
            })
        st.dataframe(pd.DataFrame(results))
        if len(case_params) >= 2:
            delta_I = case_params[1]['I'] - case_params[0]['I']
            delta_cf = np.array(case_params[1]['cf']) - np.array(case_params[0]['cf'])
            delta_n = max(case_params[0]['n'], case_params[1]['n'])
            delta_npv = npv(delta_I, delta_cf, case_params[0]['r'], delta_n)
            st.metric("增量NPV（方案2-方案1）", f"{delta_npv:.2f} 万元")

# ---------- 盈亏平衡分析 ----------
if use_breakeven and use_multi_scenario and input_mode == "手动输入":
    st.header("⚖️ 盈亏平衡分析（双方案）")
    if len(case_params) >= 2:
        target_var = st.selectbox("选择要求解的变量", ["初始投资", "折现率", "年净现金流"])
        var_index = 0
        if target_var == "折现率":
            var_index = 1
        elif target_var == "年净现金流":
            var_index = 2
        def diff_npv(x):
            p1 = case_params[0].copy()
            p2 = case_params[1].copy()
            if var_index == 0: p1['I'] = x
            elif var_index == 1: p1['r'] = x
            else: p1['cf'] = np.full(p1['n'], x)
            return npv(p1['I'], p1['cf'], p1['r'], p1['n']) - npv(p2['I'], p2['cf'], p2['r'], p2['n'])
        try:
            low = 0.1; high = 1000
            for _ in range(30):
                mid = (low+high)/2
                if diff_npv(mid) * diff_npv(low) <= 0:
                    high = mid
                else:
                    low = mid
            breakeven_val = mid
            st.success(f"盈亏平衡点：{target_var} = {breakeven_val:.2f}")
        except:
            st.warning("未能收敛到盈亏平衡点，请检查参数范围。")

# ---------- 多场景矩阵分析 ----------
if use_matrix and input_mode == "手动输入":
    st.header("🧮 多场景矩阵分析")
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        price_range = st.slider("电价范围 (元/kWh)", 0.4, 0.8, (0.5, 0.7), 0.02)
    with col_m2:
        hour_range = st.slider("利用小时数范围", 800, 2000, (1000, 1500), 50)
    with col_m3:
        carbon_range = st.slider("碳价范围 (元/tCO₂)", 0, 200, (30, 100), 10)
    steps = 5
    prices = np.linspace(price_range[0], price_range[1], steps)
    hours = np.linspace(hour_range[0], hour_range[1], steps)
    carbons = np.linspace(carbon_range[0], carbon_range[1], steps)
    grid = np.zeros((len(prices), len(hours)))
    for i, pr in enumerate(prices):
        for j, hr in enumerate(hours):
            annual_rev = hr * pr * 10
            cf_adj = annual_rev - C_op
            grid[i,j] = npv(I, cf_adj, r_base, n_base)
    df_matrix = pd.DataFrame(grid, index=[f"{p:.2f}" for p in prices], columns=[f"{h:.0f}h" for h in hours])
    st.subheader("NPV 矩阵 (万元)")
    st.dataframe(df_matrix.style.format("{:.2f}"))
    fig, ax = plt.subplots()
    c = ax.contourf(hours, prices, grid, cmap='RdYlGn')
    ax.set_xlabel("利用小时数 (h)")
    ax.set_ylabel("电价 (元/kWh)")
    fig.colorbar(c, ax=ax, label="NPV (万元)")
    st.pyplot(fig)

# ---------- 敏感性分析 ----------
st.header("📈 敏感性分析")

# ---------- 参数中英文映射 ----------
param_names_cn = {
    "初始投资 I (万元)": "初始投资 I",
    "年净现金流 (万元)": "年净现金流",
    "折现率 r": "折现率 r",
    "项目寿命 n (年)": "项目寿命 n",
    "年运营成本 C_op (万元)": "年运营成本",
    "年制氢量 Q (kg)": "年制氢量",
    "年发电量 (万kWh)": "年发电量"
}
param_names_en = {
    "初始投资 I (万元)": "Initial Investment",
    "年净现金流 (万元)": "Annual Net CF",
    "折现率 r": "Discount Rate",
    "项目寿命 n (年)": "Project Life",
    "年运营成本 C_op (万元)": "Annual OPEX",
    "年制氢量 Q (kg)": "Annual H2 Output",
    "年发电量 (万kWh)": "Annual Power Gen"
}

tab1, tab2, tab3 = st.tabs(["单因素分析", "双因素分析", "全局敏感性 (Sobol)"])

def get_base_params():
    p = st.session_state.params
    cf_base = np.mean(p['cf_series']) if not np.isscalar(p['cf_series']) else p['cf_series']
    return {'I': p['I'], 'r': p['r_base'], 'n': p['n_base'], 'C_op': p['C_op'], 'Q': p['Q'], 'cf': cf_base}

def update_param(pname, new_val, base):
    I, r, n, C_op, Q, cf = base['I'], base['r'], base['n'], base['C_op'], base['Q'], base['cf']
    if "初始投资" in pname: I = new_val
    elif "净现金流" in pname: cf = new_val
    elif "折现率" in pname: r = new_val
    elif "寿命" in pname: n = int(max(1, round(new_val)))
    elif "运营成本" in pname: C_op = new_val
    elif "制氢量" in pname or "发电量" in pname: Q = new_val
    return I, r, n, C_op, Q, cf

def get_relevant(target):
    if target in ["NPV", "IRR"]:
        return ["初始投资 I (万元)", "年净现金流 (万元)", "折现率 r", "项目寿命 n (年)"]
    elif target == "LCOH":
        return ["初始投资 I (万元)", "折现率 r", "项目寿命 n (年)", "年运营成本 C_op (万元)", "年制氢量 Q (kg)"]
    else:  # LCOE
        return ["初始投资 I (万元)", "折现率 r", "项目寿命 n (年)", "年运营成本 C_op (万元)", "年发电量 (万kWh)"]

# ---------- 单因素 ----------
with tab1:
    st.subheader("单因素敏感性分析（龙卷风图）")
    if input_mode != "手动输入":
        st.info("仅支持手动输入模式")
    else:
        target = st.selectbox("分析目标", targets_to_show + (["LCOE"] if use_lcoe else []), key="single_target")
        base = get_base_params()
        param_dict = {
            "初始投资 I (万元)": base['I'],
            "年净现金流 (万元)": base['cf'],
            "折现率 r": base['r'],
            "项目寿命 n (年)": base['n'],
            "年运营成本 C_op (万元)": base['C_op'],
            "年制氢量 Q (kg)": base['Q'],
            "年发电量 (万kWh)": base['Q']
        }
        relevant = get_relevant(target)
        selected = st.multiselect("选择参数", relevant, default=relevant[:3], key="single_multiselect")
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
            for pname in selected:
                c1, c2 = st.columns(2)
                with c1: down = st.number_input(f"{pname} 减少量", value=0.0, key=f"abs_d_{pname}")
                with c2: up = st.number_input(f"{pname} 增加量", value=0.0, key=f"abs_u_{pname}")
                abs_dict[pname] = (down, up)

        if st.button("运行单因素分析", key="run_single"):
            change_type = 'absolute' if use_abs else 'relative'
            results = []
            for pname in selected:
                base_val = param_dict[pname]
                if change_type == 'relative':
                    low_ch, high_ch = levels
                else:
                    down, up = abs_dict.get(pname, (0.0, 0.0))
                    low_ch, high_ch = -down, up
                for tag, ch in [("Low", low_ch), ("High", high_ch)]:
                    new_val = base_val * (1 + ch/100) if change_type == 'relative' else base_val + ch
                    I_t, r_t, n_t, C_t, Q_t, cf_t = update_param(pname, new_val, base)
                    if target == "NPV": val = npv(I_t, cf_t, r_t, n_t)
                    elif target == "IRR":
                        val = irr(I_t, cf_t, n_t)
                        val = val*100 if not np.isnan(val) else np.nan
                    elif target == "LCOH": val = lcoh(I_t, r_t, n_t, C_t, Q_t)
                    else: val = lcoe(I_t, r_t, n_t, C_t, Q_t)
                    display_name = param_names_cn.get(pname, pname)
                    results.append({"参数": display_name, "方向": tag, "变动": f"{ch:+.1f}%", target: val})
            df = pd.DataFrame(results)
            pivot = df.pivot(index="参数", columns="方向", values=target).reset_index()
            pivot_clean = pivot.dropna(subset=["Low", "High"])
            if len(pivot_clean) < len(pivot):
                st.warning("部分参数计算结果为NaN，已自动从龙卷风图中剔除。")
            pivot_clean["变化范围"] = pivot_clean["High"] - pivot_clean["Low"]
            st.subheader("📋 敏感性数据表")
            st.dataframe(pivot_clean.style.format(subset=["Low", "High", "变化范围"], formatter="{:.4f}"))

            if len(pivot_clean) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                params_en = [param_names_en.get(p, p) for p in pivot_clean["参数"]]
                ranges = pivot_clean["变化范围"]
                colors = ['#d62728' if x > 0 else '#2ca02c' for x in ranges]
                ax.barh(params_en, ranges, color=colors, edgecolor='white')
                ax.set_xlabel(f"{target} Change Range", fontsize=12)
                ax.set_title(f"Single-factor Sensitivity for {target}", fontsize=14)
                ax.axvline(0, color='black', linewidth=0.8)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.error("没有可用的数据来绘图。")

# ---------- 双因素 ----------
with tab2:
    st.subheader("双因素敏感性分析（热力图）")
    if input_mode != "手动输入":
        st.info("仅支持手动输入模式")
    else:
        target2 = st.selectbox("分析目标", targets_to_show + (["LCOE"] if use_lcoe else []), key="dual_target")
        base2 = get_base_params()
        param_dict2 = {
            "初始投资 I (万元)": base2['I'], "年净现金流 (万元)": base2['cf'],
            "折现率 r": base2['r'], "项目寿命 n (年)": base2['n'],
            "年运营成本 C_op (万元)": base2['C_op'],
            "年制氢量 Q (kg)": base2['Q'], "年发电量 (万kWh)": base2['Q']
        }
        relevant2 = get_relevant(target2)
        col_a, col_b = st.columns(2)
        with col_a:
            param_x = st.selectbox("X轴参数", relevant2, index=0,
                                   format_func=lambda x: param_names_cn.get(x, x), key="dual_param_x")
            x_range = st.slider("X变动范围 (%)", -50, 50, (-20, 20), 5)
        with col_b:
            param_y = st.selectbox("Y轴参数", relevant2, index=min(1, len(relevant2)-1),
                                   format_func=lambda x: param_names_cn.get(x, x), key="dual_param_y")
            y_range = st.slider("Y变动范围 (%)", -50, 50, (-20, 20), 5)

        if st.button("生成热力图及表格", key="run_dual"):
            xs = np.linspace(x_range[0]/100, x_range[1]/100, 10)
            ys = np.linspace(y_range[0]/100, y_range[1]/100, 10)
            grid = np.zeros((len(ys), len(xs)))
            has_nan = False
            for i, dy in enumerate(ys):
                for j, dx in enumerate(xs):
                    I_t, r_t, n_t, C_t, Q_t, cf_t = base2['I'], base2['r'], base2['n'], base2['C_op'], base2['Q'], base2['cf']
                    for pname, chg in [(param_x, dx), (param_y, dy)]:
                        new_val = param_dict2[pname] * (1 + chg)
                        I_t, r_t, n_t, C_t, Q_t, cf_t = update_param(pname, new_val,
                                                                      {'I':I_t,'r':r_t,'n':n_t,'C_op':C_t,'Q':Q_t,'cf':cf_t})
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
            df_grid.index.name = f"{param_names_cn.get(param_y, param_y)} 变化"
            df_grid.columns.name = f"{param_names_cn.get(param_x, param_x)} 变化"
            st.subheader("📋 网格数据表")
            st.dataframe(df_grid.style.format("{:.4f}"))

            fig, ax = plt.subplots()
            c = ax.contourf(xs*100, ys*100, grid, cmap='RdYlGn')
            ax.set_xlabel(f"{param_names_en.get(param_x, param_x)} change (%)")
            ax.set_ylabel(f"{param_names_en.get(param_y, param_y)} change (%)")
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
        base3 = get_base_params()
        relevant3 = get_relevant(target3)
        selected_sobol = st.multiselect("选择参数", relevant3, default=relevant3[:3], key="sobol_multiselect")
        sample_N = st.number_input("基础样本数 N", value=256, min_value=64, step=64, help="总运行次数 = N*(2D+2)")
        if st.button("运行 Sobol 分析", key="run_sobol"):
            if len(selected_sobol) < 2:
                st.warning("至少选择两个参数")
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
                    if "折现率" in pname: low, high = max(0.01, low), min(0.5, high)
                    if "寿命" in pname: low, high = max(1, int(low)), max(2, int(high))
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
                        "参数": [param_names_cn.get(p, p) for p in selected_sobol],
                        "一阶指数 S1": Si['S1'],
                        "总效应指数 ST": Si['ST']
                    })
                    st.subheader("📋 Sobol 敏感指数表")
                    st.dataframe(df_si.style.format(subset=["一阶指数 S1", "总效应指数 ST"], formatter="{:.4f}"))

                    fig, ax = plt.subplots()
                    x = np.arange(len(selected_sobol))
                    w = 0.35
                    ax.bar(x - w/2, Si['S1'], w, label='S1', color='#1f77b4')
                    ax.bar(x + w/2, Si['ST'], w, label='ST', color='#ff7f0e')
                    ax.set_xticks(x)
                    ax.set_xticklabels([param_names_en.get(p, p) for p in selected_sobol], rotation=30)
                    ax.set_ylabel("Sensitivity Index")
                    ax.set_title(f"Sobol Sensitivity for {target3}")
                    ax.legend()
                    st.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.caption("项目经济性分析平台 v2.0")
