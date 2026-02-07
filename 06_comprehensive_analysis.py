# -*- coding: utf-8 -*-
"""
06_comprehensive_analysis.py
综合分析脚本 - 覆盖全部需求

需求1: 3-31前总体分析（不拆分train/test）- 基线、单因素、多因素
需求2: 修正住院天数极端值 -8→8
需求3: 改进前后对比表（增加总例数列）
需求4: HA-VTE=1子集，目标变量=规范预防，3-31前后分别分析
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.stats import fisher_exact, chi2_contingency, mannwhitneyu, shapiro, ttest_ind
import warnings
warnings.filterwarnings('ignore')

BASE = r'C:\Users\Administrator\Desktop\2.5.388'


# ========================================
# 工具函数
# ========================================

def classify_variables(data, outcome):
    """将变量分为连续型和分类型"""
    all_cols = [c for c in data.columns if c != outcome
                and data[c].dtype in ['int64', 'float64', 'int32', 'float32', 'bool']]
    continuous_vars = []
    categorical_vars = []
    for col in all_cols:
        unique_vals = data[col].dropna().unique()
        if len(unique_vals) > 10 and data[col].dtype in ['float64', 'int64']:
            continuous_vars.append(col)
        elif len(unique_vals) <= 10 or set(unique_vals) <= {0, 1}:
            categorical_vars.append(col)
        else:
            continuous_vars.append(col)
    return continuous_vars, categorical_vars


def baseline_analysis(data, outcome, continuous_vars, categorical_vars):
    """基线特征分析（阳性组 vs 阴性组），连续变量根据正态性选择检验方法"""
    pos = data[data[outcome] == 1]
    neg = data[data[outcome] == 0]
    results = []

    for var in continuous_vars:
        try:
            pos_vals = pos[var].dropna()
            neg_vals = neg[var].dropna()
            if len(pos_vals) < 3 or len(neg_vals) < 3:
                continue
            # 正态性检验
            _, p_norm0 = shapiro(neg_vals) if len(neg_vals) <= 5000 else (0, 0.01)
            _, p_norm1 = shapiro(pos_vals) if len(pos_vals) <= 5000 else (0, 0.01)
            if p_norm0 > 0.05 and p_norm1 > 0.05:
                _, p = ttest_ind(neg_vals, pos_vals)
                desc_neg = f'{neg_vals.mean():.2f}±{neg_vals.std():.2f}'
                desc_pos = f'{pos_vals.mean():.2f}±{pos_vals.std():.2f}'
                method = 't检验'
            else:
                _, p = mannwhitneyu(neg_vals, pos_vals, alternative='two-sided')
                desc_neg = f'{neg_vals.median():.2f}[{neg_vals.quantile(0.25):.2f}-{neg_vals.quantile(0.75):.2f}]'
                desc_pos = f'{pos_vals.median():.2f}[{pos_vals.quantile(0.25):.2f}-{pos_vals.quantile(0.75):.2f}]'
                method = 'Mann-Whitney U'
            results.append({
                '变量': var, '类型': '连续',
                f'阳性组(n={len(pos)})': desc_pos,
                f'阴性组(n={len(neg)})': desc_neg,
                '统计方法': method, 'P值': p
            })
        except:
            continue

    for var in categorical_vars:
        try:
            pos_count = int(pos[var].sum())
            neg_count = int(neg[var].sum())
            pos_n = len(pos)
            neg_n = len(neg)
            table = np.array([[pos_count, pos_n - pos_count],
                              [neg_count, neg_n - neg_count]])
            if table.min() < 0:
                continue
            if table.shape == (2, 2) and (table < 5).any():
                _, p = fisher_exact(table)
                method = 'Fisher精确'
            else:
                _, p, _, _ = chi2_contingency(table, correction=True)
                method = '卡方检验'
            results.append({
                '变量': var, '类型': '分类',
                f'阳性组(n={pos_n})': f'{pos_count}/{pos_n} ({pos_count / pos_n * 100:.1f}%)',
                f'阴性组(n={neg_n})': f'{neg_count}/{neg_n} ({neg_count / neg_n * 100:.1f}%)',
                '统计方法': method, 'P值': p
            })
        except:
            continue

    return pd.DataFrame(results).sort_values('P值')


def univariate_logistic(data, outcome):
    """单因素Logistic回归"""
    y = data[outcome]
    all_vars = [c for c in data.columns if c != outcome
                and data[c].dtype in ['int64', 'float64', 'int32', 'float32', 'bool']]
    valid_vars = [v for v in all_vars if data[v].std() > 0 and data[v].notna().sum() > 10]

    results = []
    for var in valid_vars:
        try:
            X = data[[var]].copy()
            X = sm.add_constant(X)
            mask = X.notna().all(axis=1) & y.notna()
            X_clean = X[mask]
            y_clean = y[mask]
            if len(y_clean) < 20 or y_clean.nunique() < 2:
                continue
            model = sm.Logit(y_clean, X_clean)
            result = model.fit(disp=0, maxiter=100)
            coef = result.params[var]
            se = result.bse[var]
            p_val = result.pvalues[var]
            OR = np.exp(coef)
            CI_low = np.exp(coef - 1.96 * se)
            CI_high = np.exp(coef + 1.96 * se)
            results.append({
                '变量': var, 'β系数': round(coef, 4), 'SE': round(se, 4),
                'OR值': round(OR, 4),
                '95%CI下限': round(CI_low, 4),
                '95%CI上限': round(CI_high, 4),
                'P值': p_val
            })
        except:
            continue

    return pd.DataFrame(results).sort_values('P值')


def multivariate_logistic(data, outcome, uni_df, max_p=0.1):
    """多因素Logistic回归（含共线性检查和EPV约束）"""
    y = data[outcome]

    if uni_df.empty:
        return pd.DataFrame(), "无单因素显著变量，无法构建多因素模型"

    sig_vars = uni_df[uni_df['P值'] < max_p]['变量'].tolist()

    if len(sig_vars) < 1:
        return pd.DataFrame(), "无P<0.1的变量，无法构建多因素模型"

    # 只有1个变量时直接入模
    if len(sig_vars) == 1:
        candidate_vars = sig_vars
    else:
        # 共线性检查（相关系数>0.8时剔除P值更大的）
        X_cand = data[sig_vars].copy()
        corr_matrix = X_cand.corr().abs()
        to_remove = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.8:
                    var_i = corr_matrix.columns[i]
                    var_j = corr_matrix.columns[j]
                    p_i = uni_df[uni_df['变量'] == var_i]['P值'].values[0]
                    p_j = uni_df[uni_df['变量'] == var_j]['P值'].values[0]
                    to_remove.add(var_i if p_i > p_j else var_j)
        candidate_vars = [v for v in sig_vars if v not in to_remove]

    # EPV原则：每个变量至少10个事件
    n_events = int(min(y.sum(), len(y) - y.sum()))
    max_vars = max(int(n_events / 10), 1)
    if max_vars < len(candidate_vars):
        candidate_vars = candidate_vars[:max_vars]

    if len(candidate_vars) < 1:
        return pd.DataFrame(), "经共线性筛选后无候选变量"

    try:
        X_multi = data[candidate_vars].copy()
        X_multi = sm.add_constant(X_multi)
        mask = X_multi.notna().all(axis=1) & y.notna()
        X_clean = X_multi[mask]
        y_clean = y[mask]

        if y_clean.nunique() < 2:
            return pd.DataFrame(), "目标变量无变异，无法拟合模型"

        model = sm.Logit(y_clean, X_clean)
        result = model.fit(disp=0, maxiter=200)

        # 计算VIF（方差膨胀因子）
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        vif_values = {}
        X_vif = data[candidate_vars].copy().fillna(0)
        if len(candidate_vars) > 1:
            for i, var in enumerate(candidate_vars):
                try:
                    vif_values[var] = round(variance_inflation_factor(X_vif.values, i), 2)
                except:
                    vif_values[var] = np.nan
        else:
            for var in candidate_vars:
                vif_values[var] = 1.0

        multi_results = []
        for var in candidate_vars:
            coef = result.params[var]
            se = result.bse[var]
            p_val = result.pvalues[var]
            OR = np.exp(coef)
            CI_low = np.exp(coef - 1.96 * se)
            CI_high = np.exp(coef + 1.96 * se)
            multi_results.append({
                '变量': var, 'β系数': round(coef, 4), 'SE': round(se, 4),
                'OR值': round(OR, 4),
                '95%CI下限': round(CI_low, 4),
                '95%CI上限': round(CI_high, 4),
                'P值': round(p_val, 4),
                'VIF': vif_values.get(var, np.nan)
            })

        multi_df = pd.DataFrame(multi_results).sort_values('P值')

        sig_count = len(multi_df[multi_df['P值'] < 0.05])
        # VIF判读：<5正常，5-10中等，>10严重共线性
        max_vif = multi_df['VIF'].max() if not multi_df['VIF'].isna().all() else 0
        vif_note = '无共线性问题' if max_vif < 5 else ('中等共线性' if max_vif < 10 else '存在严重共线性')

        summary = f"样本量: {len(y_clean)}\n"
        summary += f"阳性: {int(y_clean.sum())}例\n"
        summary += f"阴性: {int(len(y_clean) - y_clean.sum())}例\n"
        summary += f"入模变量: {len(candidate_vars)}个\n"
        summary += f"显著变量(P<0.05): {sig_count}个\n"
        summary += f"VIF最大值: {max_vif:.2f}（{vif_note}）\n\n"
        if sig_count > 0:
            summary += "显著变量:\n"
            for _, row in multi_df[multi_df['P值'] < 0.05].iterrows():
                summary += f"  {row['变量']}: OR={row['OR值']}, 95%CI={row['95%CI下限']}-{row['95%CI上限']}, P={row['P值']}, VIF={row['VIF']}\n"
        return multi_df, summary
    except Exception as e:
        return pd.DataFrame(), f"模型拟合失败: {str(e)}"


def run_full_analysis(data, outcome, prefix, label, exclude_cols=None):
    """运行完整分析流水线：基线+单因素+多因素
    
    基线分析（描述性）使用全部变量，
    单因素/多因素回归排除泄漏变量。
    """
    print(f"\n{'=' * 60}")
    print(f"【{label}】")
    n_pos = int(data[outcome].sum())
    n_neg = int(len(data) - n_pos)
    print(f"样本量: {len(data)}, 阳性({outcome}=1): {n_pos}, 阴性: {n_neg}")
    print(f"{'=' * 60}")

    # ---- 基线分析：使用全部变量（描述性统计，不排除任何变量）----
    cont_vars_full, cat_vars_full = classify_variables(data, outcome)
    print(f"基线变量: 连续{len(cont_vars_full)}个, 分类{len(cat_vars_full)}个")

    baseline_df = baseline_analysis(data, outcome, cont_vars_full, cat_vars_full)
    baseline_df.to_csv(f'{BASE}\\{prefix}_baseline.csv', index=False, encoding='utf-8-sig')
    print(f"基线分析完成: {len(baseline_df)}个变量（含全部描述性变量）")

    # ---- 回归分析：排除泄漏变量 ----
    reg_data = data.copy()
    if exclude_cols:
        cols_to_drop = [c for c in exclude_cols if c in reg_data.columns and c != outcome]
        reg_data = reg_data.drop(columns=cols_to_drop)
        print(f"回归分析排除泄漏变量: {len(cols_to_drop)}个")

    # 单因素分析
    uni_df = univariate_logistic(reg_data, outcome)
    uni_df.to_csv(f'{BASE}\\{prefix}_univariate.csv', index=False, encoding='utf-8-sig')
    sig_uni = len(uni_df[uni_df['P值'] < 0.05]) if not uni_df.empty else 0
    print(f"单因素分析完成: {len(uni_df)}个变量, {sig_uni}个P<0.05")

    # 多因素分析
    multi_df, summary = multivariate_logistic(reg_data, outcome, uni_df)
    multi_df.to_csv(f'{BASE}\\{prefix}_multivariate.csv', index=False, encoding='utf-8-sig')
    with open(f'{BASE}\\{prefix}_model_summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"【{label}】\n目标变量: {outcome}\n\n{summary}")
    print(f"多因素分析完成")
    print(summary)

    return baseline_df, uni_df, multi_df


# ========================================
# 数据加载与预处理
# ========================================
print("=" * 60)
print("数据加载与预处理")
print("=" * 60)

df = pd.read_csv(f'{BASE}\\final_processed_data_full_pipeline.csv')
print(f"原始数据: {len(df)}行")

# 删除常量列
const_cols = [c for c in df.columns if df[c].nunique() <= 1]
df = df.drop(columns=const_cols)
print(f"删除{len(const_cols)}个常量列")

# 映射目标变量
df['潜在可预防VTE'] = df['潜在可预防VTE'].map({True: 1, False: 0, 'True': 1, 'False': 0})

# 【批注2/3-肺栓塞】主要疾病诊断_肺栓塞=1 且 HA-VTE≠1 → 潜在可预防VTE强制为0
pe_col = '主要疾病诊断_肺栓塞'
if pe_col in df.columns:
    # 批注3补充：需检查HA-VTE是否=1
    mask_pe = (df[pe_col] == 1)
    if '医院相关性VTE' in df.columns:
        mask_pe_no_havte = mask_pe & (df['医院相关性VTE'] != 1)
        n_pe_fix = (mask_pe_no_havte & (df['潜在可预防VTE'] == 1)).sum()
        df.loc[mask_pe_no_havte, '潜在可预防VTE'] = 0
        print(f"肺栓塞规则: 肺栓塞=1且HA-VTE≠1 → 潜在可预防VTE→0，影响{n_pe_fix}例")
    # 所有肺栓塞阳性（无论HA-VTE）均设为非潜在可预防
    n_pe_all = (mask_pe & (df['潜在可预防VTE'] == 1)).sum()
    df.loc[mask_pe, '潜在可预防VTE'] = 0
    print(f"肺栓塞规则: 全部肺栓塞=1 → 潜在可预防VTE→0，影响{n_pe_all}例")

# 处理缺失值
missing_col = '首次VTE中高风险评分日期与机械预防日期差值'
if missing_col in df.columns:
    df[missing_col + '_无'] = df[missing_col].isnull().astype(int)
    df[missing_col] = df[missing_col].fillna(0)

# 【需求2】修正极端值: 住院天数 -8 → 8
if '住院天数' in df.columns:
    n_fixed = (df['住院天数'] == -8).sum()
    df.loc[df['住院天数'] == -8, '住院天数'] = 8
    print(f"修正住院天数极端值: {n_fixed}条 (-8→8)")

# 【批注D/F/M】合并变量：90天前是否我院就诊 + 本次入院前90天有无住院史 → 90天内院内就诊/住院史
col_a = '90天前是否我院就诊'
col_b = '本次入院前90天有无住院史、治疗史、手术史'
col_new = '90天内院内就诊/住院史'
if col_a in df.columns and col_b in df.columns:
    df[col_new] = ((df[col_a] == 1) | (df[col_b] == 1)).astype(int)
    df.drop(columns=[col_a, col_b], inplace=True)
    print(f"变量合并: {col_a} + {col_b} → {col_new} (OR逻辑)")

# 【批注G/H】删除冗余变量：机械预防措施_气压治疗 ≡ 是否机械预防=1
redundant_cols = ['机械预防措施（VTE确诊前）_气压治疗', '机械预防措施（VTE确诊前）_0']
for c in redundant_cols:
    if c in df.columns:
        df.drop(columns=[c], inplace=True)
print(f"删除冗余列: {[c for c in redundant_cols if c not in df.columns]}")

# 【批注I】名称更改：预防措施_无预防 → 预防措施_基础预防
if '预防措施_无预防' in df.columns:
    df.rename(columns={'预防措施_无预防': '预防措施_基础预防'}, inplace=True)
    print("名称更改: 预防措施_无预防 → 预防措施_基础预防")

# 【批注3-7】我院相关VTE与医院相关性VTE合并（OR逻辑）
if '我院相关VTE' in df.columns and '医院相关性VTE' in df.columns:
    df['医院相关性VTE'] = ((df['医院相关性VTE'] == 1) | (df['我院相关VTE'] == 1)).astype(int)
    df.drop(columns=['我院相关VTE'], inplace=True)
    print("变量合并: 我院相关VTE + 医院相关性VTE → 医院相关性VTE (OR逻辑)")

# 【批注3-6】日期差值哑变量 → 连续数值 + none指示变量
# 对于VTE首次中高危评分日期与机械/药物预防日期差值，将哑变量重构为连续变量
for prefix in ['VTE首次中高危评分日期与机械预防日期差值', 'VTE首次中高危评分日期与药物预防日期差值']:
    dummy_cols = [c for c in df.columns if c.startswith(prefix + '_')]
    if not dummy_cols:
        continue
    # 检查是否有none列
    none_col = prefix + '_none'
    # 从哑变量反推连续值
    cont_name = prefix + '(连续)'
    df[cont_name] = 0.0
    has_value = pd.Series(False, index=df.index)
    for dc in dummy_cols:
        suffix = dc.replace(prefix + '_', '')
        if suffix == 'none':
            continue
        try:
            val = float(suffix)
            mask = df[dc] == 1
            df.loc[mask, cont_name] = val
            has_value = has_value | mask
        except ValueError:
            continue
    # none指示变量
    none_indicator = prefix + '_无值'
    df[none_indicator] = (~has_value).astype(int)
    df.loc[~has_value, cont_name] = 0  # 无值时连续变量为0
    # 删除原哑变量列
    df.drop(columns=dummy_cols, inplace=True)
    print(f"日期差值重构: {prefix} 哑变量({len(dummy_cols)}列) → 连续变量+无值指示")

# 按时间分组
df['入院日期_dt'] = pd.to_datetime(df['入院日期'])
cutoff = pd.to_datetime('2025-03-31')
before_df = df[df['入院日期_dt'] <= cutoff].copy()
after_df = df[df['入院日期_dt'] > cutoff].copy()

# 删除日期列
date_cols = ['入院日期', '入院日期_dt']
for d in [before_df, after_df]:
    d.drop(columns=[c for c in date_cols if c in d.columns], inplace=True)

print(f"\n3-31前: {len(before_df)}例 (潜在可预防VTE阳性: {int(before_df['潜在可预防VTE'].sum())})")
print(f"3-31后: {len(after_df)}例 (潜在可预防VTE阳性: {int(after_df['潜在可预防VTE'].sum())})")

# 【批注3-1/2】将预处理后的数据写回csv（同步train/test/ext变量名）
# 按原80/20比例拆分before_df → train(179) + test(45)
from sklearn.model_selection import train_test_split
_before_no_date = before_df.copy()
_train_new, _test_new = train_test_split(_before_no_date, test_size=45, random_state=42,
                                          stratify=_before_no_date['潜在可预防VTE'])
_train_new.to_csv(f'{BASE}\\train_data.csv', index=False, encoding='utf-8-sig')
_test_new.to_csv(f'{BASE}\\test_data.csv', index=False, encoding='utf-8-sig')
after_df.to_csv(f'{BASE}\\external_validation_data.csv', index=False, encoding='utf-8-sig')
print(f"已更新: train_data.csv({len(_train_new)}), test_data.csv({len(_test_new)}), external_validation_data.csv({len(after_df)})")

# 【批注3-2】导出预处理后的总CSV
df_export = df.drop(columns=['入院日期_dt'], errors='ignore')
df_export.to_csv(f'{BASE}\\preprocessed_full_data.csv', index=False, encoding='utf-8-sig')
print(f"导出预处理后总数据: preprocessed_full_data.csv ({len(df_export)}行)")


# ========================================
# 统一定义泄漏变量排除列表
# ========================================
# 排除定义性/哑变量泄漏，但保留5个有临床意义的争议变量：
# 规范预防、是否机械预防、是否药物预防、
# 首次VTE中高风险评分日期与机械预防日期差值、首次VTE中高风险评分日期与药物预防日期差值

keep_vars = [
    '规范预防', '是否机械预防', '是否药物预防',
    '首次VTE中高风险评分日期与机械预防日期差值',
    '首次VTE中高风险评分日期与药物预防日期差值',
]

leak_keywords = [
    '预防',           # 覆盖预防措施哑变量等
    '医院相关性VTE',  # 定义前提
    '我院相关VTE',    # 已合并
]

all_columns = list(before_df.columns) + [c for c in after_df.columns if c not in before_df.columns]
leak_all = []
for col in all_columns:
    if col in keep_vars:
        continue  # 保留争议变量
    for kw in leak_keywords:
        if kw in col and col not in leak_all:
            leak_all.append(col)
            break
leak_all = list(set(leak_all))
print(f"排除泄漏变量: {len(leak_all)}个（保留{len([v for v in keep_vars if v in all_columns])}个争议变量）")


# ========================================
# 【需求1】3-31前 总体分析（目标: 潜在可预防VTE）
# ========================================
# 排除泄漏变量（潜在可预防VTE的目标变量不在排除列表中，由函数内部处理）
pvte_exclude = [v for v in leak_all if v != '潜在可预防VTE']

run_full_analysis(
    before_df, '潜在可预防VTE', 'overall_before',
    label='3-31前总体分析（目标: 潜在可预防VTE）',
    exclude_cols=pvte_exclude
)

# 同步生成3-31后的分析（排除同样的泄漏变量）
run_full_analysis(
    after_df, '潜在可预防VTE', 'overall_after',
    label='3-31后总体分析（目标: 潜在可预防VTE）',
    exclude_cols=pvte_exclude
)


# ========================================
# 【需求3】改进前后对比表（增加总例数列）
# ========================================
print(f"\n{'=' * 60}")
print("【需求3】前后对比表（改进版，含总例数）")
print(f"{'=' * 60}")

# 总例数
n_before = len(before_df)
n_after = len(after_df)

# 医院相关性VTE
havte_before_n = int(before_df['医院相关性VTE'].sum()) if '医院相关性VTE' in before_df.columns else 0
havte_after_n = int(after_df['医院相关性VTE'].sum()) if '医院相关性VTE' in after_df.columns else 0

# 医院相关性VTE P值
table_havte = np.array([
    [havte_before_n, n_before - havte_before_n],
    [havte_after_n, n_after - havte_after_n]
])
if (table_havte < 5).any():
    _, p_havte = fisher_exact(table_havte)
else:
    _, p_havte, _, _ = chi2_contingency(table_havte)

# 潜在可预防VTE在HA-VTE中占比
pvte_havte_before = int(before_df[(before_df['医院相关性VTE'] == 1) & (before_df['潜在可预防VTE'] == 1)].shape[0])
pvte_havte_after = int(after_df[(after_df['医院相关性VTE'] == 1) & (after_df['潜在可预防VTE'] == 1)].shape[0])

table_pvte = np.array([
    [pvte_havte_before, havte_before_n - pvte_havte_before],
    [pvte_havte_after, havte_after_n - pvte_havte_after]
])
if (table_pvte < 5).any():
    _, p_pvte = fisher_exact(table_pvte)
else:
    _, p_pvte, _, _ = chi2_contingency(table_pvte)

# 潜在可预防VTE阳性中的规范预防率
pvte_pos_before = before_df[before_df['潜在可预防VTE'] == 1]
pvte_pos_after = after_df[after_df['潜在可预防VTE'] == 1]
gf_before = int(pvte_pos_before['规范预防'].sum()) if '规范预防' in pvte_pos_before.columns else 0
gf_after = int(pvte_pos_after['规范预防'].sum()) if '规范预防' in pvte_pos_after.columns else 0
pvte_pos_n_before = len(pvte_pos_before)
pvte_pos_n_after = len(pvte_pos_after)

table_gf = np.array([
    [gf_before, pvte_pos_n_before - gf_before],
    [gf_after, pvte_pos_n_after - gf_after]
])
if (table_gf < 5).any():
    _, p_gf = fisher_exact(table_gf)
else:
    _, p_gf, _, _ = chi2_contingency(table_gf)

# 潜在可预防VTE总占比（全部VTE中）
pvte_before_total = int(before_df['潜在可预防VTE'].sum())
pvte_after_total = int(after_df['潜在可预防VTE'].sum())
table_pvte_total = np.array([
    [pvte_before_total, n_before - pvte_before_total],
    [pvte_after_total, n_after - pvte_after_total]
])
if (table_pvte_total < 5).any():
    _, p_pvte_total = fisher_exact(table_pvte_total)
else:
    _, p_pvte_total, _, _ = chi2_contingency(table_pvte_total)

# HA-VTE=1中 规范预防率
gf_havte_before = int(before_df[(before_df['医院相关性VTE'] == 1) & (before_df['规范预防'] == 1)].shape[0]) if '规范预防' in before_df.columns else 0
gf_havte_after = int(after_df[(after_df['医院相关性VTE'] == 1) & (after_df['规范预防'] == 1)].shape[0]) if '规范预防' in after_df.columns else 0

table_gf_havte = np.array([
    [gf_havte_before, havte_before_n - gf_havte_before],
    [gf_havte_after, havte_after_n - gf_havte_after]
])
if (table_gf_havte < 5).any():
    _, p_gf_havte = fisher_exact(table_gf_havte)
else:
    _, p_gf_havte, _, _ = chi2_contingency(table_gf_havte)


def safe_pct(num, denom):
    if denom == 0:
        return "0/0 (0.0%)"
    return f"{num}/{denom} ({num / denom * 100:.1f}%)"


compare_data = [
    {
        '指标': '总VTE例数',
        '3-31前总例数': n_before,
        '3-31前': f'{n_before}',
        '3-31后总例数': n_after,
        '3-31后': f'{n_after}',
        'P值': '-'
    },
    {
        '指标': '医院相关性VTE(HA-VTE)',
        '3-31前总例数': n_before,
        '3-31前': safe_pct(havte_before_n, n_before),
        '3-31后总例数': n_after,
        '3-31后': safe_pct(havte_after_n, n_after),
        'P值': round(p_havte, 4)
    },
    {
        '指标': '潜在可预防VTE占全部VTE',
        '3-31前总例数': n_before,
        '3-31前': safe_pct(pvte_before_total, n_before),
        '3-31后总例数': n_after,
        '3-31后': safe_pct(pvte_after_total, n_after),
        'P值': round(p_pvte_total, 4)
    },
    {
        '指标': '潜在可预防VTE在HA-VTE中占比',
        '3-31前总例数': havte_before_n,
        '3-31前': safe_pct(pvte_havte_before, havte_before_n),
        '3-31后总例数': havte_after_n,
        '3-31后': safe_pct(pvte_havte_after, havte_after_n),
        'P值': round(p_pvte, 4)
    },
    {
        '指标': '潜在可预防VTE阳性中规范预防率',
        '3-31前总例数': pvte_pos_n_before,
        '3-31前': safe_pct(gf_before, pvte_pos_n_before),
        '3-31后总例数': pvte_pos_n_after,
        '3-31后': safe_pct(gf_after, pvte_pos_n_after),
        'P值': round(p_gf, 4)
    },
    {
        '指标': 'HA-VTE中规范预防率',
        '3-31前总例数': havte_before_n,
        '3-31前': safe_pct(gf_havte_before, havte_before_n),
        '3-31后总例数': havte_after_n,
        '3-31后': safe_pct(gf_havte_after, havte_after_n),
        'P值': round(p_gf_havte, 4)
    },
]

compare_df = pd.DataFrame(compare_data)
compare_df.to_csv(f'{BASE}\\compare_before_after_v2.csv', index=False, encoding='utf-8-sig')
print("对比表已保存: compare_before_after_v2.csv")
print(compare_df.to_string(index=False))


# ========================================
# 【需求4】HA-VTE=1 子集, 目标变量=规范预防
# ========================================
print(f"\n{'=' * 60}")
print("【需求4】HA-VTE=1 子集分析，目标变量: 规范预防")
print(f"{'=' * 60}")

# 复用统一定义的泄漏排除列表（已在前面定义）
# 对于目标=规范预防的分析，还需额外排除 潜在可预防VTE
exclude_vars = list(set(leak_all + ['潜在可预防VTE']))
print(f"排除预防相关变量: {len(exclude_vars)}个（避免循环预测）")

# ----- 3-31前 HA-VTE=1 分析 -----
havte_before = before_df[before_df['医院相关性VTE'] == 1].copy()
print(f"\n3-31前 HA-VTE=1: {len(havte_before)}例, 规范预防=1: {int(havte_before['规范预防'].sum())}例")

run_full_analysis(
    havte_before, '规范预防', 'havte_before',
    label='3-31前 HA-VTE=1 子集（目标: 规范预防）',
    exclude_cols=exclude_vars
)

# ----- 3-31后 HA-VTE=1 分析 -----
havte_after = after_df[after_df['医院相关性VTE'] == 1].copy()
print(f"\n3-31后 HA-VTE=1: {len(havte_after)}例, 规范预防=1: {int(havte_after['规范预防'].sum())}例")

if len(havte_after) < 20:
    print(f"警告: 3-31后HA-VTE=1样本量仅{len(havte_after)}例，分析结果可能不稳定")

run_full_analysis(
    havte_after, '规范预防', 'havte_after',
    label='3-31后 HA-VTE=1 子集（目标: 规范预防）',
    exclude_cols=exclude_vars
)


# ========================================
# 汇总输出
# ========================================
# ========================================
# 描述性统计汇总表
# ========================================
print(f"\n{'=' * 60}")
print("描述性统计汇总表")
print(f"{'=' * 60}")


def descriptive_stats(data, outcome):
    """生成变量描述性统计汇总表（与SPSS描述统计格式一致）"""
    cont_vars, cat_vars = classify_variables(data, outcome)
    rows = []

    for var in cont_vars:
        s = data[var].dropna()
        if len(s) == 0:
            continue
        rows.append({
            'Variable': var,
            'Type': 'Numeric',
            'Count': len(s),
            'Mean': round(s.mean(), 2),
            'Std Dev': round(s.std(), 2),
            'Min': round(s.min(), 2),
            'Max': round(s.max(), 2),
            '25%': round(s.quantile(0.25), 2),
            '50%(Median)': round(s.median(), 2),
            '75%': round(s.quantile(0.75), 2),
            'Category': '',
            'Percentage (%)': ''
        })

    for var in cat_vars:
        s = data[var].dropna()
        if len(s) == 0:
            continue
        pos_count = int(s.sum())
        pct = round(pos_count / len(s) * 100, 2) if len(s) > 0 else 0
        rows.append({
            'Variable': var,
            'Type': 'Categorical',
            'Count': len(s),
            'Mean': '',
            'Std Dev': '',
            'Min': '',
            'Max': '',
            '25%': '',
            '50%(Median)': '',
            '75%': '',
            'Category': pos_count,
            'Percentage (%)': pct
        })

    return pd.DataFrame(rows)


# 3-31前
desc_before = descriptive_stats(before_df, '潜在可预防VTE')
desc_before.to_csv(f'{BASE}\\descriptive_before.csv', index=False, encoding='utf-8-sig')
print(f"3-31前描述性统计: {len(desc_before)}个变量 → descriptive_before.csv")

# 3-31后
desc_after = descriptive_stats(after_df, '潜在可预防VTE')
desc_after.to_csv(f'{BASE}\\descriptive_after.csv', index=False, encoding='utf-8-sig')
print(f"3-31后描述性统计: {len(desc_after)}个变量 → descriptive_after.csv")


# ========================================
# 汇总输出
# ========================================
print(f"\n{'=' * 60}")
print("全部分析完成！输出文件汇总:")
print(f"{'=' * 60}")
print("""
需求1 - 3-31前总体分析（目标: 潜在可预防VTE）:
  overall_before_baseline.csv        基线特征
  overall_before_univariate.csv      单因素Logistic回归
  overall_before_multivariate.csv    多因素Logistic回归
  overall_before_model_summary.txt   模型摘要

需求3 - 前后对比表（含总例数）:
  compare_before_after_v2.csv        改进版对比表

需求4 - HA-VTE=1子集分析（目标: 规范预防）:
  havte_before_baseline.csv          3-31前 基线特征
  havte_before_univariate.csv        3-31前 单因素
  havte_before_multivariate.csv      3-31前 多因素
  havte_before_model_summary.txt     3-31前 模型摘要
  havte_after_baseline.csv           3-31后 基线特征
  havte_after_univariate.csv         3-31后 单因素
  havte_after_multivariate.csv       3-31后 多因素
  havte_after_model_summary.txt      3-31后 模型摘要

描述性统计汇总表:
  descriptive_before.csv             3-31前 变量描述统计
  descriptive_after.csv              3-31后 变量描述统计
""")
