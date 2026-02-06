# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

train_df = pd.read_csv(r'C:\Users\Administrator\Desktop\2.5.388\train_data.csv')
var_info = pd.read_csv(r'C:\Users\Administrator\Desktop\2.5.388\variable_info.csv')

outcome = '潜在可预防VTE'
group0 = train_df[train_df[outcome] == 0]
group1 = train_df[train_df[outcome] == 1]

continuous_vars = var_info[var_info['type'] == 'continuous']['variable'].tolist()
categorical_vars = var_info[var_info['type'] == 'categorical']['variable'].tolist()

key_continuous = [v for v in continuous_vars if v in train_df.columns][:20]
key_categorical = []
for v in categorical_vars:
    if v in train_df.columns:
        pos_rate = train_df[v].mean()
        if 0.05 <= pos_rate <= 0.95:
            key_categorical.append(v)
key_categorical = key_categorical[:50]

results = []

for var in key_continuous:
    try:
        g0 = group0[var].dropna()
        g1 = group1[var].dropna()
        if len(g0) < 3 or len(g1) < 3:
            continue
        _, p_normal0 = stats.shapiro(g0) if len(g0) <= 5000 else (0, 0.01)
        _, p_normal1 = stats.shapiro(g1) if len(g1) <= 5000 else (0, 0.01)
        if p_normal0 > 0.05 and p_normal1 > 0.05:
            stat, p_val = stats.ttest_ind(g0, g1)
            desc0 = f'{g0.mean():.2f}±{g0.std():.2f}'
            desc1 = f'{g1.mean():.2f}±{g1.std():.2f}'
            test_method = 't-test'
        else:
            stat, p_val = stats.mannwhitneyu(g0, g1, alternative='two-sided')
            desc0 = f'{g0.median():.2f}[{g0.quantile(0.25):.2f}-{g0.quantile(0.75):.2f}]'
            desc1 = f'{g1.median():.2f}[{g1.quantile(0.25):.2f}-{g1.quantile(0.75):.2f}]'
            test_method = 'Mann-Whitney'
        sig = '*' if p_val < 0.05 else ''
        results.append({'变量': var, '类型': '连续', '阴性组(n={})'.format(len(group0)): desc0,
            '阳性组(n={})'.format(len(group1)): desc1, '统计方法': test_method, 'P值': f'{p_val:.4f}{sig}'})
    except:
        continue

for var in key_categorical:
    try:
        ct = pd.crosstab(train_df[var], train_df[outcome])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2, p_val, dof, expected = stats.chi2_contingency(ct)
        if (expected < 5).sum() / expected.size > 0.2:
            if ct.shape == (2, 2):
                _, p_val = stats.fisher_exact(ct)
                test_method = 'Fisher'
            else:
                test_method = 'Chi-square'
        else:
            test_method = 'Chi-square'
        n0 = group0[var].sum()
        n1 = group1[var].sum()
        pct0 = n0 / len(group0) * 100
        pct1 = n1 / len(group1) * 100
        desc0 = f'{int(n0)}({pct0:.1f}%)'
        desc1 = f'{int(n1)}({pct1:.1f}%)'
        sig = '*' if p_val < 0.05 else ''
        results.append({'变量': var, '类型': '分类', '阴性组(n={})'.format(len(group0)): desc0,
            '阳性组(n={})'.format(len(group1)): desc1, '统计方法': test_method, 'P值': f'{p_val:.4f}{sig}'})
    except:
        continue

results_df = pd.DataFrame(results)
results_df.to_csv(r'C:\Users\Administrator\Desktop\2.5.388\python_baseline_results.csv', index=False, encoding='utf-8-sig')
