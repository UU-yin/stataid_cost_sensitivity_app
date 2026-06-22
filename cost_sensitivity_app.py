import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os, requests, re
from copy import deepcopy

# ---------- 中文字体 ----------
@st.cache_resource
def register_chinese_font():
    target_fonts = ['SimHei','Microsoft YaHei','WenQuanYi Zen Hei','Noto Sans CJK SC','Arial Unicode MS']
    available = [f.name for f in fm.fontManager.ttflist]
    for font in target_fonts:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            return
    font_path = "/tmp/SimHei.ttf"
    if not os.path.exists(font_path):
        try:
            r = requests.get("https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf", timeout=15)
            if r.status_code==200:
                with open(font_path,'wb') as f: f.write(r.content)
        except: pass
    if os.path.exists(font_path):
        try:
            fm.fontManager.addfont(font_path)
            plt.rcParams['font.sans-serif'] = ['SimHei'] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            fm._load_fontmanager(try_read_cache=False)
        except: pass
register_chinese_font()

# ---------- 财务函数 ----------
try:
    import numpy_financial as npf
    IRR_FUNC = npf.irr
except:
    from scipy.optimize import newton
    def IRR_FUNC(cfs):
        cfs = np.asarray(cfs, dtype=float)
        if np.all(cfs>=0) or np.all(cfs<=0): return np.nan
        try: return newton(lambda r: np.npv(r, cfs), 0.1, maxiter=100)
        except: return np.nan

try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    SALIB_AVAILABLE = True
except: SALIB_AVAILABLE = False

def crf(r,n):
    return (r*(1+r)**n)/((1+r)**n-1) if r!=0 else 1/n

def npv(I,cf,r,n):
    t = np.arange(1,n+1)
    if np.isscalar(cf): cfs = np.full(n,cf,dtype=float)
    else:
        cfs = np.asarray(cf,dtype=float).ravel()
        if len(cfs)<n: cfs = np.pad(cfs,(0,n-len(cfs)))
        else: cfs = cfs[:n]
    return np.sum(cfs/(1+r)**t) - I

def irr(I,cf,n):
    if np.isscalar(cf): cfs = np.full(n,cf,dtype=float)
    else:
        cfs = np.asarray(cf,dtype=float).ravel()
        if len(cfs)<n: cfs = np.pad(cfs,(0,n-len(cfs)))
        else: cfs = cfs[:n]
    full_cf = np.insert(cfs,0,-I)
    full_cf = np.nan_to_num(full_cf, nan=0.0, posinf=1e9, neginf=-1e9)
    if np.all(full_cf>=0) or np.all(full_cf<=0): return np.nan
    try: return IRR_FUNC(full_cf)
    except: return np.nan

def lcoh(I,r,n,C_op,Q):
    if Q==0: return np.nan
    return (I*crf(r,n)+C_op)/Q

def lcoe(I,r,n,C_op,Q_gen):
    if Q_gen==0: return np.nan
    return (I*crf(r,n)+C_op)/Q_gen

def loan_schedule(principal, annual_rate, years, method='等额本息'):
    if method=='等额本息':
        if annual_rate==0:
            annual_payment = principal/years
            interests = [0]*years
        else:
            annual_payment = principal*(annual_rate*(1+annual_rate)**years)/((1+annual_rate)**years-1)
            interests=[]; balance=principal
            for _ in range(years):
                interest = balance*annual_rate
                principal_paid = annual_payment - interest
                balance -= principal_paid
                interests.append(interest)
        return [annual_payment]*years, interests
    else:
        annual_principal = principal/years
        payments=[]; interests=[]; balance=principal
        for _ in range(years):
            interest = balance*annual_rate
            payments.append(annual_principal+interest)
            interests.append(interest)
            balance -= annual_principal
        return payments, interests

# ---------- 表达式求值 ----------
def safe_eval(expr, var_dict):
    expr = str(expr).replace(' ','')
    if not re.match(r'^[0-9a-zA-Z_\+\-\*/\(\)\.]+$', expr):
        return np.nan
    local_vars = {k: v for k,v in var_dict.items() if not k.startswith('__')}
    try:
        return eval(expr, {"__builtins__": None}, local_vars)
    except:
        return np.nan

def parse_item_amount(amount_str, var_dict):
    if isinstance(amount_str, (int,float)):
        return float(amount_str)
    if not isinstance(amount_str, str):
        return np.nan
    s = amount_str.strip()
    if s=='':
        return 0.0
    try:
        return float(s)
    except:
        pass
    return safe_eval(s, var_dict)

# ---------- 项目现金流计算 ----------
def compute_full_project(params):
    I = params['I']
    r = params['r_base']
    n = params['n_base']
    Q = params['Q']
    C_op = params['C_op']
    include_dep = params.get('include_depreciation', False)
    custom_dep = params.get('custom_depreciation', None)

    var_dict = params.get('custom_vars', {})
    if params.get('use_advanced_cf'):
        rev_items = params.get('rev_items', [])
        cost_items = params.get('cost_items', [])
        total_rev = np.zeros(n)
        total_cost = np.zeros(n)
        for item in rev_items:
            amt = parse_item_amount(item['amount'], var_dict)
            if np.isnan(amt): amt = 0.0
            growth = item.get('growth',0.0)
            for t in range(n):
                total_rev[t] += amt * (1+growth)**t
        for item in cost_items:
            amt = parse_item_amount(item['amount'], var_dict)
            if np.isnan(amt): amt = 0.0
            growth = item.get('growth',0.0)
            for t in range(n):
                total_cost[t] += amt * (1+growth)**t
        cf_series = total_rev - total_cost
    else:
        cf_val = params.get('cf_series', 300.0)
        if np.isscalar(cf_val):
            cf_series = np.full(n, cf_val, dtype=float)
        else:
            cf_series = np.asarray(cf_val, dtype=float).ravel()
            if len(cf_series)<n:
                cf_series = np.pad(cf_series, (0, n-len(cf_series)))
            else:
                cf_series = cf_series[:n]

    if include_dep:
        annual_dep = custom_dep if custom_dep is not None else (I/n if n>0 else 0)
        cf_series = cf_series - annual_dep

    if params.get('use_finance'):
        loan_ratio = params.get('loan_ratio',0.0)
        loan_rate = params.get('loan_rate',0.0)
        loan_years = int(params.get('loan_years',0))
        loan_amount = I * loan_ratio
        if loan_amount>0 and loan_years>0:
            payments, _ = loan_schedule(loan_amount, loan_rate, loan_years,
                                        method=params.get('repay_method','等额本息'))
            for t in range(min(loan_years, n)):
                cf_series[t] -= payments[t]

    if params.get('use_replacement') and params.get('replacements'):
        for yr, cost in params['replacements']:
            if 0<yr<=n:
                cf_series[yr-1] -= cost

    if params.get('use_carbon') and params.get('carbon_params'):
        ef, cp, gcp, agg = params['carbon_params']
        carbon_rev = (agg * ef / 1000 * cp) + (agg * gcp / 1000)
        cf_series = cf_series + carbon_rev

    cf_series = np.asarray(cf_series, dtype=float).ravel()
    cf_series = np.nan_to_num(cf_series, nan=0.0, posinf=1e9, neginf=-1e9)
    if len(cf_series)<n:
        cf_series = np.pad(cf_series, (0,n-len(cf_series)))
    return I, r, n, Q, C_op, cf_series[:n]

# ---------- 静态回收期 ----------
def payback_period(I, cf_series):
    cumulative = 0.0
    for t,cf in enumerate(cf_series, start=1):
        cumulative += cf
        if cumulative >= I:
            prev = cumulative - cf
            return (t-1) + (I-prev)/cf
    return float('inf')

# ---------- 逆向求解函数 ----------
def solve_param_for_target(target_type, target_value, param_key, base_params, all_specs, key_to_updater):
    base_val = None
    for spec in all_specs:
        if spec[0] == param_key:
            base_val = spec[2]
            break
    if base_val is None:
        return None, False, "参数未找到"

    if param_key == 'r_base':
        low, high = 0.005, 0.5
    elif param_key in ['n_base', 'loan_years']:
        low, high = 2, 50
    elif param_key == 'loan_ratio':
        low, high = 0.0, 1.0
    else:
        if base_val >= 0:
            low = base_val * 0.1
            high = base_val * 5.0
        else:
            low = base_val * 5.0
            high = base_val * 0.1
        if low == 0: low = 1e-6
        if high == 0: high = -1e-6
        if low > high: low, high = high, low

    def calc_target(p):
        I_t, r_t, n_t, Q_t, C_t, cf_t = compute_full_project(p)
        if target_type == 'irr':
            val = irr(I_t, cf_t, n_t)
            if val is None or np.isnan(val):
                cf_full = np.insert(cf_t, 0, -I_t)
                if np.all(cf_full >= 0): return 9.99
                elif np.all(cf_full <= 0): return -0.01
                else: return np.nan
            return val
        else:
            return npv(I_t, cf_t, p['r_base'], n_t)

    def f(x):
        p = deepcopy(base_params)
        updater = key_to_updater.get(param_key)
        if updater is None: return np.nan
        if param_key in ['n_base', 'loan_years']:
            updater(p, max(1, int(round(x))))
        else:
            updater(p, x)
        return calc_target(p)

    base_target_val = calc_target(base_params)
    if base_target_val is None or np.isnan(base_target_val):
        return None, False, "基准参数无法计算有效IRR"

    f_low = f(low)
    f_high = f(high)
    expand_count = 0
    max_expand = 50
    while expand_count < max_expand:
        if not np.isnan(f_low) and not np.isnan(f_high):
            if min(f_low, f_high) <= target_value <= max(f_low, f_high):
                break
        if np.isnan(f_low):
            low *= 0.5; f_low = f(low)
        elif np.isnan(f_high):
            high *= 1.5; f_high = f(high)
        else:
            if target_value < min(f_low, f_high):
                if f_low <= f_high: low *= 0.5; f_low = f(low)
                else: high *= 0.5; f_high = f(high)
            else:
                if f_low >= f_high: low *= 0.5; f_low = f(low)
                else: high *= 1.5; f_high = f(high)
        expand_count += 1

    if np.isnan(f_low) or np.isnan(f_high):
        return None, False, "无法计算有效IRR区间"
    if abs(f_high - base_target_val) < 1e-6 and abs(f_low - base_target_val) < 1e-6:
        return None, False, "该参数几乎不影响IRR"
    if (f_low - target_value) * (f_high - target_value) > 0:
        return None, False, f"目标IRR {target_value*100:.2f}% 不可达（范围 [{f_low*100:.2f}%, {f_high*100:.2f}%]）"

    for _ in range(100):
        mid = (low + high) / 2
        f_mid = f(mid)
        if np.isnan(f_mid):
            if abs(f_low - target_value) < abs(f_high - target_value): high = mid
            else: low = mid
            continue
        if abs(f_mid - target_value) < 1e-6:
            return mid, True, "求解成功"
        if (f_low - target_value) * (f_mid - target_value) <= 0:
            high = mid; f_high = f_mid
        else:
            low = mid; f_low = f_mid
    return (low + high) / 2, True, "近似解"

# ---------- 构建参数列表 (重构，无 lambda，彻底避免括号错误) ----------
def _upd_I(p, v): p.update({'I': v})
def _upd_r(p, v): p.update({'r_base': v})
def _upd_n(p, v): p.update({'n_base': int(max(1, round(v)))})
def _upd_C_op(p, v): p.update({'C_op': v})
def _upd_Q(p, v): p.update({'Q': v})
def _upd_loan_ratio(p, v): p.update({'loan_ratio': v})
def _upd_loan_rate(p, v): p.update({'loan_rate': v})
def _upd_loan_years(p, v): p.update({'loan_years': int(max(1, round(v)))})
def _make_upd_rev_amount(idx):
    def upd(p, v): p['rev_items'][idx].update({'amount': v})
    return upd
def _make_upd_rev_growth(idx):
    def upd(p, v): p['rev_items'][idx].update({'growth': v})
    return upd
def _make_upd_cost_amount(idx):
    def upd(p, v): p['cost_items'][idx].update({'amount': v})
    return upd
def _make_upd_cost_growth(idx):
    def upd(p, v): p['cost_items'][idx].update({'growth': v})
    return upd
def _make_upd_var(k):
    def upd(p, v): p['custom_vars'].update({k: v})
    return upd
def _make_upd_replace(idx):
    def upd(p, v): p['replacements'][idx] = (p['replacements'][idx][0], v)
    return upd
def _make_upd_carbon(idx):
    def upd(p, v): p['carbon_params'][idx] = v
    return upd

def build_all_param_specs(base_params, unit_label):
    specs = []
    specs.append(('I', f'初始投资 ({unit_label})', base_params['I'], _upd_I))
    specs.append(('r_base', '折现率 r', base_params['r_base'], _upd_r))
    specs.append(('n_base', '项目寿命 (年)', base_params['n_base'], _upd_n))
    specs.append(('C_op', f'年运营成本 ({unit_label})', base_params['C_op'], _upd_C_op))
    if base_params.get('use_advanced_cf'):
        if base_params.get('rev_items'):
            for i, item in enumerate(base_params['rev_items']):
                name = item['name']
                specs.append((f'rev_{i}_amount', f'收益-{name} 金额', item['amount'], _make_upd_rev_amount(i)))
                specs.append((f'rev_{i}_growth', f'收益-{name} 增长率', item['growth'], _make_upd_rev_growth(i)))
        if base_params.get('cost_items'):
            for i, item in enumerate(base_params['cost_items']):
                name = item['name']
                specs.append((f'cost_{i}_amount', f'支出-{name} 金额', item['amount'], _make_upd_cost_amount(i)))
                specs.append((f'cost_{i}_growth', f'支出-{name} 增长率', item['growth'], _make_upd_cost_growth(i)))
    for var_name, var_val in base_params.get('custom_vars', {}).items():
        specs.append((f'var_{var_name}', f'变量-{var_name}', var_val, _make_upd_var(var_name)))
    if base_params.get('use_finance'):
        specs.append(('loan_ratio', '贷款比例', base_params['loan_ratio'], _upd_loan_ratio))
        specs.append(('loan_rate', '贷款年利率', base_params['loan_rate'], _upd_loan_rate))
        specs.append(('loan_years', '贷款年限', base_params['loan_years'], _upd_loan_years))
    if base_params.get('use_replacement') and base_params.get('replacements'):
        for i, (yr, cost) in enumerate(base_params['replacements']):
            specs.append((f'replace_{i}_cost', f'替换{i+1}金额', cost, _make_upd_replace(i)))
    if base_params.get('use_carbon') and base_params.get('carbon_params'):
        cp_list = base_params['carbon_params']
        specs.append(('emission_factor', '排放因子', cp_list[0], _make_upd_carbon(0)))
        specs.append(('carbon_price', '碳价', cp_list[1], _make_upd_carbon(1)))
        specs.append(('green_cert_price', '绿证价格', cp_list[2], _make_upd_carbon(2)))
        specs.append(('annual_green_gen', '自发绿电', cp_list[3], _make_upd_carbon(3)))
    return specs

# ---------- 页面设置 ----------
st.set_page_config(page_title="项目经济性分析平台", layout="wide")
st.title("项目成本计算与敏感性分析平台")

# ---------- 侧边栏 ----------
st.sidebar.header("📌 分析方法")
analysis_scope = st.sidebar.selectbox("指标数量", ["单个","两个","三个"], index=1, key='scope_sel')
all_targets = ["NPV","IRR","LCOH"]
if analysis_scope == "单个":
    selected_targets = [st.sidebar.selectbox("指标", all_targets, index=0, key='target_single')]
elif analysis_scope == "两个":
    selected_targets = st.sidebar.multiselect("两个指标", all_targets, default=["NPV","IRR"], max_selections=2, key='target_two')
else:
    selected_targets = st.sidebar.multiselect("指标", all_targets, default=all_targets, key='target_three')
st.session_state['selected_targets'] = selected_targets if selected_targets else all_targets

unit_choice = st.sidebar.selectbox("💲 单位", ["万元","亿元"], index=0, key='unit_sel')
UNIT_SCALE = 10000.0 if unit_choice=="亿元" else 1.0
unit_label = unit_choice

include_depreciation = st.sidebar.checkbox("折旧计入现金流", value=False, key='dep_check')
st.sidebar.header("⚙️ 高级功能")
use_advanced_cf = st.sidebar.checkbox("现金流分项构建器", value=False, key='advcf_check')
use_carbon = st.sidebar.checkbox("碳排放与碳收益", value=False, key='carbon_check')
use_finance = st.sidebar.checkbox("融资结构", value=False, key='finance_check')
use_replacement = st.sidebar.checkbox("大修/替换成本", value=False, key='replace_check')
use_lcoe = st.sidebar.checkbox("计算LCOE", value=False, key='lcoe_check')

st.sidebar.header("🎯 逆向工具")
use_irr_backsolve = st.sidebar.checkbox("单参数逆向求解", value=False, key='backsolve_check')

# ---------- 输入方式 ----------
st.header("📥 数据输入")
input_mode = st.radio("输入方式", ["手动输入","上传文件"], horizontal=True, key='input_mode_radio')

if 'params' not in st.session_state:
    st.session_state.params = {}
if 'custom_vars' not in st.session_state:
    st.session_state.custom_vars = {}

if input_mode == "手动输入":
    has_lcoh = "LCOH" in st.session_state.selected_targets
    col1,col2,col3 = st.columns(3)
    with col1:
        I_raw = st.number_input(f"初始投资 I ({unit_label})", value=0.0, step=1.0, key='inv_I')
        I = I_raw * UNIT_SCALE
    with col2:
        r_base = st.number_input("基准折现率 r", value=0.08, step=0.01, format="%.3f", key='disc_r')
    with col3:
        n_base = st.number_input("项目寿命 n (年)", value=20, min_value=1, key='life_n')
    custom_depreciation = None
    if include_depreciation:
        auto_dep = I / n_base if n_base>0 else 0
        dep_raw = st.number_input(f"年折旧额 ({unit_label}/年)", value=auto_dep/UNIT_SCALE, key='dep_amt')
        custom_depreciation = dep_raw * UNIT_SCALE

    if has_lcoh or use_lcoe:
        c4,c5 = st.columns(2)
        with c4:
            Q = st.number_input("年制氢量 (kg)" if has_lcoh else "年发电量 (万kWh)", value=0.0, step=1.0, key='Q_val')
        with c5:
            C_op_raw = st.number_input(f"年运营成本 ({unit_label}/年)", value=0.0, key='op_cost')
            C_op = C_op_raw * UNIT_SCALE
    else:
        Q = 1.0; C_op = 0.0

    with st.expander("🔢 自定义变量（用于公式）"):
        st.caption("定义中间变量，如 Q_green, P_green。金额列可使用表达式（如 Q_green * P_green）")
        if 'custom_vars_df' not in st.session_state:
            st.session_state.custom_vars_df = pd.DataFrame(columns=['变量名','数值'])
        edited_vars = st.data_editor(
            st.session_state.custom_vars_df,
            num_rows="dynamic",
            column_config={
                "变量名": st.column_config.TextColumn(required=True),
                "数值": st.column_config.NumberColumn(format="%.4f")
            },
            key='var_edit'
        )
        custom_vars = {}
        for _, row in edited_vars.iterrows():
            if row['变量名'] and row['变量名'].strip():
                custom_vars[row['变量名'].strip()] = row['数值'] * UNIT_SCALE
        st.session_state.custom_vars = custom_vars

    rev_items = []
    cost_items = []
    if not use_advanced_cf:
        st.subheader("净现金流设置")
        cf_mode = st.radio("类型", ["等额年金","逐年输入"], horizontal=True, key='cf_mode')
        if cf_mode == "等额年金":
            cf_val_raw = st.number_input(f"年均净现金流 ({unit_label})", value=0.0, key='annuity_cf')
            cf_series = cf_val_raw * UNIT_SCALE
        else:
            cf_str = st.text_area(f"逗号分隔 ({unit_label})", "0,0,0", key='yearly_cf')
            try:
                cf_series = [float(x.strip())*UNIT_SCALE for x in cf_str.split(',') if x.strip()]
            except:
                st.error("格式错误"); st.stop()
    else:
        st.subheader("💵 现金流分项构建")
        tab_rev, tab_cost = st.tabs(["收益项","支出项"])
        with tab_rev:
            st.caption("金额列可填数字或表达式（如 Q_green * P_green）")
            if 'rev_df' not in st.session_state:
                st.session_state.rev_df = pd.DataFrame(columns=['项目名称','金额','年增长率(%)'])
            rev_edited = st.data_editor(
                st.session_state.rev_df,
                num_rows="dynamic",
                column_config={
                    "项目名称": st.column_config.TextColumn(),
                    "金额": st.column_config.TextColumn(),
                    "年增长率(%)": st.column_config.NumberColumn(format="%.2f")
                },
                key='rev_editor'
            )
            rev_items = []
            for _, row in rev_edited.iterrows():
                if row['项目名称']:
                    amount = row['金额']
                    growth = row['年增长率(%)']/100.0 if row['年增长率(%)'] else 0.0
                    try:
                        num_amt = float(amount)
                        amount = num_amt * UNIT_SCALE
                    except:
                        pass
                    rev_items.append({'name': row['项目名称'], 'amount': amount, 'growth': growth})
        with tab_cost:
            st.caption("金额列支持表达式")
            if 'cost_df' not in st.session_state:
                st.session_state.cost_df = pd.DataFrame(columns=['项目名称','金额','年增长率(%)'])
            cost_edited = st.data_editor(
                st.session_state.cost_df,
                num_rows="dynamic",
                column_config={
                    "项目名称": st.column_config.TextColumn(),
                    "金额": st.column_config.TextColumn(),
                    "年增长率(%)": st.column_config.NumberColumn(format="%.2f")
                },
                key='cost_editor'
            )
            cost_items = []
            for _, row in cost_edited.iterrows():
                if row['项目名称']:
                    amount = row['金额']
                    growth = row['年增长率(%)']/100.0 if row['年增长率(%)'] else 0.0
                    try:
                        num_amt = float(amount)
                        amount = num_amt * UNIT_SCALE
                    except:
                        pass
                    cost_items.append({'name': row['项目名称'], 'amount': amount, 'growth': growth})

        cf_series = None

    loan_ratio=0.0; loan_rate=0.0; loan_years=0; repay_method='等额本息'
    if use_finance:
        st.subheader("🏦 融资结构")
        loan_ratio = st.slider("贷款比例 (%)",0,100,0,key='loan_ratio_slider')/100
        loan_rate = st.number_input("贷款年利率 (%)",value=4.2,step=0.1,key='loan_rate_input')/100
        loan_years = st.number_input("贷款年限",min_value=1,value=15,key='loan_years_input')
        repay_method = st.selectbox("还款方式",["等额本息","等额本金"],key='repay_method')

    replacements=[]
    if use_replacement:
        st.subheader("🔧 大修/替换成本")
        rcnt = st.number_input("事件数量",min_value=0,value=0,step=1,key='rep_count')
        for i in range(rcnt):
            c1,c2=st.columns(2)
            with c1: year=st.number_input(f"事件{i+1}年份",min_value=1,value=10,key=f'ry_{i}')
            with c2: cost_raw=st.number_input(f"金额({unit_label})",value=0.0,key=f'rc_{i}')
            replacements.append((year, cost_raw*UNIT_SCALE))

    carbon_params=None
    if use_carbon:
        st.subheader("🌱 碳排放与碳收益")
        c1,c2,c3=st.columns(3)
        with c1: ef=st.number_input("电网排放因子 (kgCO₂/kWh)",value=0.58,key='ef_carbon')
        with c2: cp=st.number_input("碳价 (元/tCO₂)",value=50.0,key='cp_carbon')
        with c3: gcp=st.number_input("绿证价格 (元/个)",value=7.76,key='gcp_carbon')
        agg=st.number_input("年自发绿电量 (万kWh)",value=0.0,key='agg_carbon')
        carbon_params=[ef,cp,gcp,agg]

    st.session_state.params = {
        'I':I,'r_base':r_base,'n_base':n_base,'Q':Q,'C_op':C_op,
        'cf_series': cf_series if not use_advanced_cf else None,
        'use_advanced_cf': use_advanced_cf,
        'rev_items': rev_items,
        'cost_items': cost_items,
        'custom_vars': st.session_state.custom_vars,
        'use_finance':use_finance,'loan_ratio':loan_ratio,'loan_rate':loan_rate,
        'loan_years':loan_years,'repay_method':repay_method,
        'use_replacement':use_replacement,'replacements':replacements,
        'use_carbon':use_carbon,'carbon_params':carbon_params,
        'use_lcoe':use_lcoe,'unit_scale':UNIT_SCALE,
        'include_depreciation':include_depreciation,'custom_depreciation':custom_depreciation
    }

# ---------- 基准计算结果 ----------
st.header("📊 基准计算结果")
targets_to_show = st.session_state.get('selected_targets', ["NPV","IRR","LCOH"])

if input_mode == "手动输入" and st.session_state.params:
    base_params = deepcopy(st.session_state.params)
    I,r,n,Q,C_op,cf = compute_full_project(base_params)
    npv_val = npv(I,cf,r,n)
    irr_val = irr(I,cf,n)
    lcoh_val = lcoh(I,r,n,C_op,Q) if "LCOH" in targets_to_show else None
    lcoe_val = lcoe(I,r,n,C_op,Q) if use_lcoe else None
    scale = UNIT_SCALE
    cols = st.columns(len(targets_to_show)+(1 if use_lcoe else 0))
    idx=0
    for t in targets_to_show:
        if t=="NPV": cols[idx].metric(f"NPV ({unit_label})", f"{npv_val/scale:.2f}")
        elif t=="IRR": cols[idx].metric("IRR (%)", f"{irr_val*100:.2f}" if not np.isnan(irr_val) else "无解")
        elif t=="LCOH": cols[idx].metric("LCOH (元/kg)", f"{lcoh_val:.4f}")
        idx+=1
    if use_lcoe: cols[idx].metric("LCOE (元/kWh)", f"{lcoe_val:.4f}")

    st.subheader("⏱ 静态投资回收期")
    pay_cash = payback_period(I,cf)
    include_dep = base_params.get('include_depreciation',False)
    if include_dep:
        pay_account = pay_cash
    else:
        annual_dep = I/n if n>0 else 0
        net_profits = cf - annual_dep
        pay_account = payback_period(I, net_profits)
    c1,c2 = st.columns(2)
    c1.metric("现金流回收期", f"{pay_cash:.2f}年" if pay_cash!=float('inf') else "无法回收")
    c2.metric("会计回收期", f"{pay_account:.2f}年" if pay_account!=float('inf') else "无法回收")

    # ---------- 逆向求解 ----------
    if use_irr_backsolve:
        st.header("🎯 单参数逆向求解与盈亏分析")
        all_specs = build_all_param_specs(base_params, unit_label)
        all_display_names = [s[1] for s in all_specs]
        display_to_key = {s[1]: s[0] for s in all_specs}
        key_to_base = {s[0]: s[2] for s in all_specs}
        key_to_updater = {s[0]: s[3] for s in all_specs}

        col1, col2 = st.columns(2)
        with col1:
            target_irr = st.number_input("目标 IRR (%)", value=9.0, step=0.1, key='backsolve_target_irr')
        with col2:
            param_display = st.selectbox("选择要反算的参数", all_display_names, key='backsolve_param_sel')
        if st.button("开始逆向求解", key='backsolve_button'):
            param_key = display_to_key[param_display]
            base_val = key_to_base[param_key]
            solved_val, success, msg = solve_param_for_target('irr', target_irr/100.0, param_key,
                                                              deepcopy(st.session_state.params),
                                                              all_specs, key_to_updater)
            if success and solved_val is not None:
                if param_key in ['n_base', 'loan_years']:
                    disp_val = int(round(solved_val))
                    base_disp = int(round(base_val))
                else:
                    disp_val = solved_val / UNIT_SCALE
                    base_disp = base_val / UNIT_SCALE
                be_val, be_success, be_msg = solve_param_for_target('irr', 0.0, param_key,
                                                                    deepcopy(st.session_state.params),
                                                                    all_specs, key_to_updater)
                be_disp = None
                if be_success and be_val is not None:
                    if param_key in ['n_base', 'loan_years']:
                        be_disp = int(round(be_val))
                    else:
                        be_disp = be_val / UNIT_SCALE
                change_to_target = (disp_val - base_disp) / base_disp * 100 if base_disp != 0 else 0
                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("当前值", f"{base_disp:.4f}" if not isinstance(base_disp,int) else f"{base_disp}")
                col_r2.metric(f"目标值 (IRR={target_irr}%)", f"{disp_val:.4f}" if not isinstance(disp_val,int) else f"{disp_val}",
                              delta=f"{change_to_target:+.1f}%")
                if be_disp is not None:
                    change_to_be = (be_disp - base_disp) / base_disp * 100 if base_disp != 0 else 0
                    col_r3.metric("盈亏平衡值 (IRR=0)", f"{be_disp:.4f}" if not isinstance(be_disp,int) else f"{be_disp}",
                                  delta=f"{change_to_be:+.1f}%", delta_color="off")
                else:
                    col_r3.metric("盈亏平衡值", "无解")
                st.success(f"✅ {msg}：要使 IRR = {target_irr}%，参数 **{param_display}** 应为 **{disp_val}**")
                p_verify = deepcopy(st.session_state.params)
                updater = key_to_updater.get(param_key)
                if updater:
                    if param_key in ['n_base','loan_years']:
                        updater(p_verify, int(round(solved_val)))
                    else:
                        updater(p_verify, solved_val)
                I_v, r_v, n_v, Q_v, C_v, cf_v = compute_full_project(p_verify)
                irr_v = irr(I_v, cf_v, n_v)
                npv_v = npv(I_v, cf_v, r_v, n_v)
                st.info(f"验证：IRR = {irr_v*100:.4f}%，NPV = {npv_v/UNIT_SCALE:.2f} {unit_label}")
            else:
                st.error(msg)

st.sidebar.markdown("---")
st.sidebar.caption("氢能项目经济性分析平台 v5.3")