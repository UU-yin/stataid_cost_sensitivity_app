import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os, requests, re
from copy import deepcopy
from io import BytesIO

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
    return IRR_FUNC(np.insert(cfs,0,-I))

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

# ---------- 表达式求值引擎 ----------
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

# ---------- 现金流计算 ----------
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

# ---------- 页面设置 ----------
st.set_page_config(page_title="项目经济性分析平台", layout="wide")
st.title("项目成本计算与敏感性分析平台")

# ---------- 侧边栏 ----------
st.sidebar.header("📌 分析方法")
analysis_scope = st.sidebar.selectbox("指标数量", ["单个","两个","三个"], index=1, key='scope')
all_targets = ["NPV","IRR","LCOH"]
if analysis_scope == "单个":
    selected_targets = [st.sidebar.selectbox("指标", all_targets, index=0, key='single_target')]
elif analysis_scope == "两个":
    selected_targets = st.sidebar.multiselect("两个指标", all_targets, default=["NPV","IRR"], max_selections=2, key='two_targets')
else:
    selected_targets = st.sidebar.multiselect("指标", all_targets, default=all_targets, key='three_targets')
st.session_state['selected_targets'] = selected_targets if selected_targets else all_targets

unit_choice = st.sidebar.selectbox("💲 单位", ["万元","亿元"], index=0, key='unit')
UNIT_SCALE = 10000.0 if unit_choice=="亿元" else 1.0
unit_label = unit_choice

include_depreciation = st.sidebar.checkbox("折旧计入现金流", value=False, key='dep_check')
st.sidebar.header("⚙️ 高级功能")
use_advanced_cf = st.sidebar.checkbox("现金流分项构建器", value=False, key='advcf')
use_carbon = st.sidebar.checkbox("碳排放与碳收益", value=False, key='carbon')
use_finance = st.sidebar.checkbox("融资结构", value=False, key='finance')
use_replacement = st.sidebar.checkbox("大修/替换成本", value=False, key='replace')
use_lcoe = st.sidebar.checkbox("计算LCOE", value=False, key='lcoe')
use_multi_scenario = st.sidebar.checkbox("多方案对比", value=False, key='multi')
use_breakeven = st.sidebar.checkbox("盈亏平衡分析", value=False, key='breakeven')
use_matrix = st.sidebar.checkbox("多场景矩阵", value=False, key='matrix')
st.sidebar.header("🎯 逆向工具")
use_irr_backsolve = st.sidebar.checkbox("单参数逆向求解", value=False, key='backsolve')
use_irr_contour = st.sidebar.checkbox("双参数边界图", value=False, key='contour')

# ---------- 输入方式 ----------
st.header("📥 数据输入")
input_mode = st.radio("输入方式", ["手动输入","上传文件"], horizontal=True, key='input_mode')
if 'params' not in st.session_state:
    st.session_state.params = {}
if 'custom_vars' not in st.session_state:
    st.session_state.custom_vars = {}

# ---------- 手动输入 ----------
if input_mode == "手动输入":
    has_lcoh = "LCOH" in st.session_state.selected_targets
    col1,col2,col3 = st.columns(3)
    with col1:
        I_raw = st.number_input(f"初始投资 I ({unit_label})", value=0.0, step=1.0, key='I')
        I = I_raw * UNIT_SCALE
    with col2:
        r_base = st.number_input("基准折现率 r", value=0.08, step=0.01, format="%.3f", key='r')
    with col3:
        n_base = st.number_input("项目寿命 n (年)", value=20, min_value=1, key='n')
    custom_depreciation = None
    if include_depreciation:
        auto_dep = I / n_base if n_base>0 else 0
        dep_raw = st.number_input(f"年折旧额 ({unit_label}/年)", value=auto_dep/UNIT_SCALE, key='dep')
        custom_depreciation = dep_raw * UNIT_SCALE

    if has_lcoh or use_lcoe:
        c4,c5 = st.columns(2)
        with c4:
            Q = st.number_input("年制氢量 (kg)" if has_lcoh else "年发电量 (万kWh)", value=0.0, step=1.0, key='Q')
        with c5:
            C_op_raw = st.number_input(f"年运营成本 ({unit_label}/年)", value=0.0, key='C_op')
            C_op = C_op_raw * UNIT_SCALE
    else:
        Q = 1.0; C_op = 0.0

    # 自定义变量
    with st.expander("🔢 自定义变量（用于公式）"):
        st.caption("定义中间变量，例如：Q_green, P_green。金额列可使用表达式（如 Q_green * P_green）")
        if 'custom_vars_df' not in st.session_state:
            st.session_state.custom_vars_df = pd.DataFrame(columns=['变量名','数值'])
        edited_vars = st.data_editor(
            st.session_state.custom_vars_df,
            num_rows="dynamic",
            column_config={
                "变量名": st.column_config.TextColumn(required=True),
                "数值": st.column_config.NumberColumn(format="%.4f")
            },
            key='var_editor'
        )
        custom_vars = {}
        for _, row in edited_vars.iterrows():
            if row['变量名'] and row['变量名'].strip():
                custom_vars[row['变量名'].strip()] = row['数值']
        st.session_state.custom_vars = custom_vars

    rev_items = []
    cost_items = []
    if not use_advanced_cf:
        st.subheader("净现金流设置")
        cf_mode = st.radio("类型", ["等额年金","逐年输入"], horizontal=True, key='cf_mode')
        if cf_mode == "等额年金":
            cf_val_raw = st.number_input(f"年均净现金流 ({unit_label})", value=0.0, key='annuity')
            cf_series = cf_val_raw * UNIT_SCALE
        else:
            cf_str = st.text_area(f"逗号分隔 ({unit_label})", "0,0,0", key='yearly')
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
                    "金额": st.column_config.TextColumn(),  # 可输入表达式
                    "年增长率(%)": st.column_config.NumberColumn(format="%.2f")
                },
                key='rev_editor'
            )
            rev_items = []
            for _, row in rev_edited.iterrows():
                if row['项目名称']:
                    amount = row['金额']
                    growth = row['年增长率(%)']/100.0 if row['年增长率(%)'] else 0.0
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
                    cost_items.append({'name': row['项目名称'], 'amount': amount, 'growth': growth})

        cf_series = None  # 用于后续存储

    # 融资
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
            with c1: year=st.number_input(f"事件{i+1}年份",min_value=1,value=10,key=f'ry{i}')
            with c2: cost_raw=st.number_input(f"金额({unit_label})",value=0.0,key=f'rc{i}')
            replacements.append((year, cost_raw*UNIT_SCALE))

    carbon_params=None
    if use_carbon:
        st.subheader("🌱 碳排放与碳收益")
        c1,c2,c3=st.columns(3)
        with c1: ef=st.number_input("电网排放因子 (kgCO₂/kWh)",value=0.58,key='ef')
        with c2: cp=st.number_input("碳价 (元/tCO₂)",value=50.0,key='cp')
        with c3: gcp=st.number_input("绿证价格 (元/个)",value=7.76,key='gcp')
        agg=st.number_input("年自发绿电量 (万kWh)",value=0.0,key='agg')
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

# ---------- 文件上传 ----------
elif input_mode == "上传文件":
    uploaded = st.file_uploader("上传 CSV 或 Excel", type=['csv','xlsx'])
    st.markdown("文件必须包含列：`类型`（收入/支出）、`项目`、`金额`（支持表达式）、`年增长率(%)`（可选）")
    if uploaded:
        try:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            required = {'类型','项目','金额'}
            if required.issubset(df.columns):
                revs = []; costs = []
                for _, row in df.iterrows():
                    typ = str(row['类型']).strip()
                    name = str(row['项目']).strip()
                    amt = row['金额'] if not pd.isna(row['金额']) else '0'
                    growth = row.get('年增长率(%)',0)
                    if pd.isna(growth): growth=0.0
                    else: growth = float(growth)/100.0
                    if typ in ['收入','收益','revenue','r']:
                        revs.append({'name':name, 'amount':amt, 'growth':growth})
                    elif typ in ['支出','成本','expense','c','cost']:
                        costs.append({'name':name, 'amount':amt, 'growth':growth})
                st.session_state.uploaded_rev = revs
                st.session_state.uploaded_cost = costs
                st.success("文件解析成功！请在下方输入其他参数。")
            else:
                st.error("缺少必要列：类型、项目、金额")
        except Exception as e:
            st.error(f"解析失败: {e}")

# ---------- 基准计算结果 ----------
st.header("📊 基准计算结果")
targets_to_show = st.session_state.get('selected_targets', ["NPV","IRR","LCOH"])

if input_mode == "手动输入" or (input_mode=="上传文件" and 'uploaded_rev' in st.session_state):
    if input_mode == "上传文件":
        # 上传模式还需手动输入基本参数，这里简单提示
        st.warning("上传文件后，请手动输入基本参数（投资、折现率、寿命等）以及融资、替换、碳收益信息（尚未集成）。建议切换至手动输入并利用表格。")
    else:
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

        # 回收期
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

# ---------- 敏感性分析参数构建 ----------
def build_all_param_specs(base_params):
    specs = []
    # 基本参数
    specs.append(('I', f'初始投资 ({unit_label})', base_params['I'], lambda p,v: p.update({'I': v})))
    specs.append(('r_base', '折现率 r', base_params['r_base'], lambda p,v: p.update({'r_base': v})))
    specs.append(('n_base', '项目寿命 (年)', base_params['n_base'], lambda p,v: p.update({'n_base': int(max(1, round(v)))})))
    specs.append(('C_op', f'年运营成本 ({unit_label})', base_params['C_op'], lambda p,v: p.update({'C_op': v})))
    
    # 现金流分项（收益/支出）
    if base_params.get('use_advanced_cf') and base_params.get('rev_items'):
        for i, item in enumerate(base_params['rev_items']):
            name = item['name']
            specs.append((
                f'rev_{i}_amount', 
                f'收益-{name} 金额', 
                item['amount'],
                lambda p, v, idx=i: p['rev_items'][idx].update({'amount': v})
            ))
            specs.append((
                f'rev_{i}_growth', 
                f'收益-{name} 增长率', 
                item['growth'],
                lambda p, v, idx=i: p['rev_items'][idx].update({'growth': v})
            ))
    if base_params.get('use_advanced_cf') and base_params.get('cost_items'):
        for i, item in enumerate(base_params['cost_items']):
            name = item['name']
            specs.append((
                f'cost_{i}_amount', 
                f'支出-{name} 金额', 
                item['amount'],
                lambda p, v, idx=i: p['cost_items'][idx].update({'amount': v})
            ))
            specs.append((
                f'cost_{i}_growth', 
                f'支出-{name} 增长率', 
                item['growth'],
                lambda p, v, idx=i: p['cost_items'][idx].update({'growth': v})
            ))
    
    # 自定义变量
    for var_name, var_val in base_params.get('custom_vars', {}).items():
        specs.append((
            f'var_{var_name}', 
            f'变量-{var_name}', 
            var_val,
            lambda p, v, k=var_name: p['custom_vars'].update({k: v})
        ))
    
    # 融资参数
    if base_params.get('use_finance'):
        specs.append(('loan_ratio', '贷款比例', base_params['loan_ratio'], lambda p,v: p.update({'loan_ratio': v})))
        specs.append(('loan_rate', '贷款年利率', base_params['loan_rate'], lambda p,v: p.update({'loan_rate': v})))
        specs.append(('loan_years', '贷款年限', base_params['loan_years'], lambda p,v: p.update({'loan_years': int(max(1, round(v)))})))
    
    # 替换成本
    if base_params.get('use_replacement') and base_params.get('replacements'):
        for i, (yr, cost) in enumerate(base_params['replacements']):
            specs.append((
                f'replace_{i}_cost', 
                f'替换{i+1}金额', 
                cost,
                lambda p, v, idx=i: p['replacements'].__setitem__(idx, (p['replacements'][idx][0], v))
            ))
    
    # 碳排放参数
    if base_params.get('use_carbon') and base_params.get('carbon_params'):
        cp = base_params['carbon_params']
        specs.append(('emission_factor', '排放因子', cp[0], lambda p,v: p['carbon_params'].__setitem__(0, v)))
        specs.append(('carbon_price', '碳价', cp[1], lambda p,v: p['carbon_params'].__setitem__(1, v)))
        specs.append(('green_cert_price', '绿证价格', cp[2], lambda p,v: p['carbon_params'].__setitem__(2, v)))
        specs.append(('annual_green_gen', '自发绿电', cp[3], lambda p,v: p['carbon_params'].__setitem__(3, v)))
    
    return specs

if input_mode == "手动输入":
    base_params_original = deepcopy(st.session_state.params)
    all_specs = build_all_param_specs(base_params_original)
    all_display_names = [s[1] for s in all_specs]
    display_to_key = {s[1]: s[0] for s in all_specs}
    key_to_base = {s[0]: s[2] for s in all_specs}
    key_to_updater = {s[0]: s[3] for s in all_specs}
else:
    all_display_names = []
    display_to_key = {}
    key_to_base = {}
    key_to_updater = {}

# ---------- 敏感性分析选项卡 ----------
tab1, tab2, tab3 = st.tabs(["单因素分析", "双因素分析", "全局敏感性 (Sobol)"])

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
                    unit_str = f" ({unit_label})" if target == "NPV" else ""
                    st.subheader(f"📋 敏感性数据表 (单位: {target}{unit_str})")
                    st.dataframe(pivot_clean.style.format(subset=["Low", "High", "变化范围"], formatter="{:.4f}"))
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ranges = pivot_clean["变化范围"]
                    colors = ['#d62728' if x > 0 else '#2ca02c' for x in ranges]
                    ax.barh(pivot_clean["参数"], ranges, color=colors)
                    ax.set_xlabel(f"{target} 变化范围{unit_str}", fontsize=12)
                    ax.set_title(f"单因素敏感性分析: {target}", fontsize=14)
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
                                           index=min(1, len(all_display_names)-1) if len(all_display_names)>1 else 0,
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
                st.warning("部分IRR无解，已替换为0")
                grid = np.nan_to_num(grid, nan=0.0)

            df_grid = pd.DataFrame(grid, index=[f"{y*100:+.0f}%" for y in ys], columns=[f"{x*100:+.0f}%" for x in xs])
            df_grid.index.name = f"{param_y_display} 变化"
            df_grid.columns.name = f"{param_x_display} 变化"
            st.subheader("📋 网格数据表")
            st.dataframe(df_grid.style.format("{:.4f}"))

            fig, ax = plt.subplots()
            c = ax.contourf(xs*100, ys*100, grid, cmap='RdYlGn')
            ax.set_xlabel(f"{param_x_display} 变化 (%)", fontsize=12)
            ax.set_ylabel(f"{param_y_display} 变化 (%)", fontsize=12)
            ax.set_title(f"双因素敏感性分析: {target2}", fontsize=14)
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
        sample_N = st.number_input("基础样本数 N", value=256, min_value=64, step=64)
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
                    elif key in ['n_base', 'loan_years']:
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
                        key_to_updater[key](p, val)
                    I_t, r_t, n_t, Q_t, C_t, cf_t = compute_full_project(p)
                    if target3 == "NPV": Y[i] = npv(I_t, cf_t, r_t, n_t)
                    elif target3 == "IRR":
                        ir = irr(I_t, cf_t, n_t)
                        Y[i] = ir*100 if not np.isnan(ir) else 0.0
                    elif target3 == "LCOH": Y[i] = lcoh(I_t, r_t, n_t, C_t, Q_t)
                    else: Y[i] = lcoe(I_t, r_t, n_t, C_t, Q_t)

                if np.any(np.isnan(Y)):
                    st.error("计算结果包含NaN，请检查参数范围")
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
                    ax.set_ylabel("敏感度指数")
                    ax.set_title(f"Sobol 全局敏感性: {target3}")
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)

# ---------- 逆向求解 ----------
if (use_irr_backsolve or use_irr_contour) and input_mode == "手动输入":
    st.header("🎯 逆向与盈亏分析")
    if all_display_names:
        if use_irr_backsolve:
            st.subheader("单参数逆向求解与盈亏分析")
            col1, col2 = st.columns(2)
            with col1:
                target_irr = st.number_input("目标 IRR (%)", value=9.0, step=0.1, key="backsolve_target_irr")
            with col2:
                param_display = st.selectbox("选择要反算的参数", all_display_names, key="backsolve_param_select")
            if st.button("开始逆向求解", key="run_backsolve_button"):
                param_key = display_to_key[param_display]
                base_val = key_to_base[param_key]
                # 调用之前的 solve_param_for_target 函数（需要实现）
                # 这里简略表示，实际应包含该函数
                st.info("逆向求解功能需集成完整的 solve_param_for_target 函数，可参考之前提供的代码。")
        if use_irr_contour:
            st.subheader("双参数盈亏边界等值线图")
            st.info("等值线图功能待集成。")
    else:
        st.warning("请先在主界面输入基本参数")

st.sidebar.markdown("---")
st.sidebar.caption("氢能项目经济性分析平台 v4.2")