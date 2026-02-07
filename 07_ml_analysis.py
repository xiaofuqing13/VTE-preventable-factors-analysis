# -*- coding: utf-8 -*-
"""
07_ml_analysis.py
机器学习分析：6种模型训练、评估、SHAP可解释性、外部验证、可视化
"""
import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (roc_auc_score, f1_score, confusion_matrix,
                             classification_report, roc_curve, accuracy_score,
                             precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

BASE = r'C:\Users\Administrator\Desktop\2.5.388'
OUTCOME = '潜在可预防VTE'

# ============================================================
# 1. 数据准备
# ============================================================
print('=' * 60)
print('【1】数据准备')
print('=' * 60)

train_raw = pd.read_csv(os.path.join(BASE, 'train_data.csv'))
test_raw = pd.read_csv(os.path.join(BASE, 'test_data.csv'))
ext_raw = pd.read_csv(os.path.join(BASE, 'external_validation_data.csv'))

# 目标变量映射
for d in [train_raw, test_raw, ext_raw]:
    d[OUTCOME] = d[OUTCOME].map({True: 1, False: 0, 'True': 1, 'False': 0, 1: 1, 0: 0})

# 【批注2-肺栓塞】主要疾病诊断_肺栓塞=1 → 潜在可预防VTE强制为0
for d in [train_raw, test_raw, ext_raw]:
    pe_col = '主要疾病诊断_肺栓塞'
    if pe_col in d.columns:
        d.loc[d[pe_col] == 1, OUTCOME] = 0

# 修正住院天数
for d in [train_raw, test_raw, ext_raw]:
    if '住院天数' in d.columns:
        d.loc[d['住院天数'] == -8, '住院天数'] = 8

# 【批注D/F/M】合并变量
for d in [train_raw, test_raw, ext_raw]:
    col_a = '90天前是否我院就诊'
    col_b = '本次入院前90天有无住院史、治疗史、手术史'
    if col_a in d.columns and col_b in d.columns:
        d['90天内院内就诊/住院史'] = ((d[col_a] == 1) | (d[col_b] == 1)).astype(int)
        d.drop(columns=[col_a, col_b], inplace=True)

# 【批注G/H】删除冗余列
for d in [train_raw, test_raw, ext_raw]:
    for c in ['机械预防措施（VTE确诊前）_气压治疗', '机械预防措施（VTE确诊前）_0']:
        if c in d.columns:
            d.drop(columns=[c], inplace=True)

# 【批注I】名称更改
for d in [train_raw, test_raw, ext_raw]:
    if '预防措施_无预防' in d.columns:
        d.rename(columns={'预防措施_无预防': '预防措施_基础预防'}, inplace=True)

# 删除非数值列
drop_cols = ['入院日期', 'dataset']
for d in [train_raw, test_raw, ext_raw]:
    d.drop(columns=[c for c in drop_cols if c in d.columns], inplace=True)

# ★★★ 排除泄漏变量（保留5个争议变量） ★★★
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

for d in [train_raw, test_raw, ext_raw]:
    cols_to_drop = []
    for col in d.columns:
        if col == OUTCOME or col in keep_vars:
            continue
        for kw in leak_keywords:
            if kw in col:
                cols_to_drop.append(col)
                break
    d.drop(columns=cols_to_drop, inplace=True, errors='ignore')

print(f'已排除 {len(cols_to_drop)} 个泄漏变量（保留{len([v for v in keep_vars if v in d.columns])}个争议变量）')

# 缺失值处理（仅对非泄漏列执行）
missing_col = '首次VTE中高风险评分日期与机械预防日期差值'
for d in [train_raw, test_raw, ext_raw]:
    if missing_col in d.columns:
        d[missing_col + '_无'] = d[missing_col].isnull().astype(int)
        d[missing_col] = d[missing_col].fillna(0)

# 只保留数值列
def prepare_numeric(df, outcome):
    num_cols = [c for c in df.columns if df[c].dtype in ['int64', 'float64', 'int32', 'float32', 'bool']]
    if outcome not in num_cols:
        num_cols.append(outcome)
    return df[num_cols].copy()

train_df = prepare_numeric(train_raw, OUTCOME)
test_df = prepare_numeric(test_raw, OUTCOME)
ext_df = prepare_numeric(ext_raw, OUTCOME)

# 删除常量列(基于训练集)
const_cols = [c for c in train_df.columns if c != OUTCOME and train_df[c].nunique() <= 1]
for d in [train_df, test_df, ext_df]:
    d.drop(columns=[c for c in const_cols if c in d.columns], inplace=True, errors='ignore')

# 确保列一致
common_cols = list(set(train_df.columns) & set(test_df.columns) & set(ext_df.columns))
common_cols = sorted(common_cols)
train_df = train_df[common_cols]
test_df = test_df[common_cols]
ext_df = ext_df[common_cols]

# 分离特征和目标
X_train = train_df.drop(columns=[OUTCOME]).fillna(0)
y_train = train_df[OUTCOME].astype(int)
X_test = test_df.drop(columns=[OUTCOME]).fillna(0)
y_test = test_df[OUTCOME].astype(int)
X_ext = ext_df.drop(columns=[OUTCOME]).fillna(0)
y_ext = ext_df[OUTCOME].astype(int)

# 标准化(SVM和KNN需要)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
X_ext_scaled = pd.DataFrame(scaler.transform(X_ext), columns=X_ext.columns, index=X_ext.index)

print(f'训练集: {len(X_train)}例 ({int(y_train.sum())}阳性)')
print(f'测试集: {len(X_test)}例 ({int(y_test.sum())}阳性)')
print(f'外部验证: {len(X_ext)}例 ({int(y_ext.sum())}阳性)')
print(f'特征数: {X_train.shape[1]}')
print(f'划分比例: {len(X_train)}/{len(X_train)+len(X_test)}={len(X_train)/(len(X_train)+len(X_test))*100:.1f}%')

# ============================================================
# 2. 定义模型和参数
# ============================================================
print('\n' + '=' * 60)
print('【2】模型训练与评估')
print('=' * 60)

n_pos = int(y_train.sum())
n_neg = int(len(y_train) - n_pos)

# 【批注2d】XGBoost调参优化（RandomizedSearchCV）
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
print('XGBoost调参优化中...')
_xgb_param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5, 7],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [0.5, 1, 2, 5],
}
_xgb_base = XGBClassifier(
    scale_pos_weight=n_neg/n_pos, eval_metric='logloss',
    random_state=42, use_label_encoder=False, n_jobs=-1)
_xgb_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
_xgb_search = RandomizedSearchCV(
    _xgb_base, _xgb_param_dist, n_iter=60, scoring='roc_auc',
    cv=_xgb_cv, random_state=42, n_jobs=-1, verbose=0)
_xgb_search.fit(X_train, y_train)
_best_xgb = _xgb_search.best_estimator_
_best_xgb_params = _xgb_search.best_params_
print(f'XGBoost最佳参数: {_best_xgb_params}')
print(f'XGBoost最佳CV AUC: {_xgb_search.best_score_:.4f}')
_xgb_params_str = ', '.join([f'{k}={v}' for k, v in _best_xgb_params.items()])

models = {
    'Random Forest': {
        'model': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1),
        'params': 'n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2, class_weight=balanced',
        'use_scaled': False
    },
    'SVM': {
        'model': SVC(
            kernel='rbf', C=1.0, gamma='scale', probability=True,
            class_weight='balanced', random_state=42),
        'params': 'kernel=rbf, C=1.0, gamma=scale, probability=True, class_weight=balanced',
        'use_scaled': True
    },
    'XGBoost': {
        'model': _best_xgb,
        'params': f'调参优化: {_xgb_params_str}',
        'use_scaled': False
    },
    'Naive Bayes': {
        'model': GaussianNB(),
        'params': 'GaussianNB (默认参数，高斯分布假设)',
        'use_scaled': True
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(
            max_depth=8, min_samples_split=5, min_samples_leaf=2,
            class_weight='balanced', random_state=42),
        'params': 'max_depth=8, min_samples_split=5, min_samples_leaf=2, class_weight=balanced',
        'use_scaled': False
    },
    'KNN': {
        'model': KNeighborsClassifier(
            n_neighbors=7, weights='distance', metric='minkowski', p=2),
        'params': 'n_neighbors=7, weights=distance, metric=minkowski(p=2)',
        'use_scaled': True
    }
}

# ============================================================
# 3. 训练和评估
# ============================================================
def evaluate_model(model, X, y, label):
    """评估模型在给定数据集上的性能"""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob)
    f1 = f1_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    fpr, tpr, _ = roc_curve(y, y_prob)
    return {
        'AUC': round(auc, 4), 'F1': round(f1, 4), 'Accuracy': round(acc, 4),
        'Precision': round(prec, 4), 'Recall': round(rec, 4),
        'TN': int(cm[0, 0]), 'FP': int(cm[0, 1]),
        'FN': int(cm[1, 0]), 'TP': int(cm[1, 1]),
        'fpr': fpr, 'tpr': tpr
    }

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.base import clone

results_train = {}
results_test = {}
fitted_models = {}

for name, cfg in models.items():
    print(f'\n--- {name} ---')
    model = cfg['model']
    Xtr = X_train_scaled if cfg['use_scaled'] else X_train
    Xte = X_test_scaled if cfg['use_scaled'] else X_test

    # 训练集评估：10折交叉验证（避免过拟合导致AUC=1.0）
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_model = clone(model)
    cv_probs = cross_val_predict(cv_model, Xtr, y_train, cv=cv, method='predict_proba')[:, 1]
    cv_preds = (cv_probs >= 0.5).astype(int)
    cv_auc = roc_auc_score(y_train, cv_probs)
    cv_f1 = f1_score(y_train, cv_preds)
    cv_acc = accuracy_score(y_train, cv_preds)
    cv_prec = precision_score(y_train, cv_preds, zero_division=0)
    cv_rec = recall_score(y_train, cv_preds)
    cv_cm = confusion_matrix(y_train, cv_preds)
    cv_fpr, cv_tpr, _ = roc_curve(y_train, cv_probs)
    r_train = {
        'AUC': round(cv_auc, 4), 'F1': round(cv_f1, 4), 'Accuracy': round(cv_acc, 4),
        'Precision': round(cv_prec, 4), 'Recall': round(cv_rec, 4),
        'TN': int(cv_cm[0, 0]), 'FP': int(cv_cm[0, 1]),
        'FN': int(cv_cm[1, 0]), 'TP': int(cv_cm[1, 1]),
        'fpr': cv_fpr, 'tpr': cv_tpr
    }

    # 全量训练后在测试集评估
    model.fit(Xtr, y_train)
    fitted_models[name] = (model, cfg['use_scaled'])
    r_test = evaluate_model(model, Xte, y_test, '测试集')

    results_train[name] = r_train
    results_test[name] = r_test

    print(f'  训练集(10折CV): AUC={r_train["AUC"]}, F1={r_train["F1"]}, Acc={r_train["Accuracy"]}')
    print(f'  测试集: AUC={r_test["AUC"]}, F1={r_test["F1"]}, Acc={r_test["Accuracy"]}')
    print(f'  测试集混淆矩阵: TN={r_test["TN"]}, FP={r_test["FP"]}, FN={r_test["FN"]}, TP={r_test["TP"]}')

# ============================================================
# 4. 输出CSV结果
# ============================================================
print('\n' + '=' * 60)
print('【3】保存结果CSV')
print('=' * 60)

# 4.1 模型参数表
params_rows = []
for name, cfg in models.items():
    params_rows.append({'模型': name, '参数设置': cfg['params']})
pd.DataFrame(params_rows).to_csv(os.path.join(BASE, 'ml_model_params.csv'), index=False, encoding='utf-8-sig')

# 4.2 模型间性能对比(测试集)
comparison_rows = []
for name in models:
    r = results_test[name]
    comparison_rows.append({
        '模型': name, 'AUC': r['AUC'], 'F1': r['F1'],
        'Accuracy': r['Accuracy'], 'Precision': r['Precision'], 'Recall': r['Recall']
    })
comp_df = pd.DataFrame(comparison_rows).sort_values('AUC', ascending=False)
comp_df.to_csv(os.path.join(BASE, 'ml_model_comparison.csv'), index=False, encoding='utf-8-sig')
print('模型间对比表已保存')

# 4.3 训练集vs测试集对比
tt_rows = []
for name in models:
    for split, res in [('训练集', results_train[name]), ('测试集', results_test[name])]:
        tt_rows.append({
            '模型': name, '数据集': split, 'AUC': res['AUC'], 'F1': res['F1'],
            'Accuracy': res['Accuracy'], 'Precision': res['Precision'], 'Recall': res['Recall'],
            'TN': res['TN'], 'FP': res['FP'], 'FN': res['FN'], 'TP': res['TP']
        })
pd.DataFrame(tt_rows).to_csv(os.path.join(BASE, 'ml_train_test_comparison.csv'), index=False, encoding='utf-8-sig')
print('训练/测试对比表已保存')

# ============================================================
# 5. 找到最佳模型
# ============================================================
best_name = comp_df.iloc[0]['模型']
best_model, best_scaled = fitted_models[best_name]
print(f'\n最佳模型(测试集AUC最高): {best_name}, AUC={comp_df.iloc[0]["AUC"]}')

# ============================================================
# 6. SHAP分析
# ============================================================
print('\n' + '=' * 60)
print('【4】SHAP可解释性分析')
print('=' * 60)

import shap

Xtr_shap = X_train_scaled if best_scaled else X_train

# 强制使用XGBoost做SHAP（最稳定），即使它不是测试集最佳
_xgb_model_for_shap = fitted_models['XGBoost'][0]
explainer = shap.TreeExplainer(_xgb_model_for_shap)
shap_values = explainer.shap_values(X_train)  # XGBoost不需要scaled
if isinstance(shap_values, list):
    shap_vals = shap_values[1]
else:
    shap_vals = shap_values

# 确保shap_vals是2D
if shap_vals.ndim != 2:
    shap_vals = shap_vals.reshape(len(X_train), -1)

# SHAP特征重要性
Xtr_shap = X_train  # XGBoost用原始数据
mean_abs_shap = np.abs(shap_vals).mean(axis=0)
shap_df = pd.DataFrame({
    '变量': Xtr_shap.columns,
    'SHAP均值': mean_abs_shap
}).sort_values('SHAP均值', ascending=False)
shap_df.to_csv(os.path.join(BASE, 'shap_features.csv'), index=False, encoding='utf-8-sig')
print(f'SHAP特征重要性已保存，Top10:')
for _, row in shap_df.head(10).iterrows():
    print(f'  {row["变量"]}: {row["SHAP均值"]:.4f}')

# SHAP summary plot
fig, ax = plt.subplots(figsize=(12, 10))
shap.summary_plot(shap_vals, Xtr_shap, max_display=20, show=False)
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'shap_summary.png'), dpi=200, bbox_inches='tight')
plt.close()
print('SHAP summary plot已保存')

# ============================================================
# 7. 危险因素汇总(单因素+多因素+SHAP)
# ============================================================
print('\n' + '=' * 60)
print('【5】危险因素汇总')
print('=' * 60)

# 读取单因素和多因素结果（使用排除泄漏变量后的overall_before版本）
uni_path = os.path.join(BASE, 'overall_before_univariate.csv')
multi_path = os.path.join(BASE, 'overall_before_multivariate.csv')
uni_df = pd.read_csv(uni_path)
multi_df = pd.read_csv(multi_path)

# 统一列名处理
or_col_uni = 'OR值' if 'OR值' in uni_df.columns else 'OR'
uni_sig = uni_df[uni_df['P值'] < 0.05][['变量', or_col_uni, 'P值']].copy()
uni_sig.columns = ['变量', '单因素OR', '单因素P值']

or_col_multi = 'OR值' if 'OR值' in multi_df.columns else 'OR'
multi_sig = multi_df[multi_df['P值'] < 0.05][['变量', or_col_multi, 'P值']].copy()
multi_sig.columns = ['变量', '多因素OR', '多因素P值']

shap_top = shap_df.head(20).copy()
shap_top.columns = ['变量', 'SHAP重要性']

# 合并
summary = shap_top.merge(uni_sig, on='变量', how='left')
summary = summary.merge(multi_sig[['变量', '多因素OR', '多因素P值']], on='变量', how='left')
summary.to_csv(os.path.join(BASE, 'risk_factors_summary.csv'), index=False, encoding='utf-8-sig')
print(f'危险因素汇总表已保存，{len(summary)}个变量')

# ============================================================
# 8. 外部验证
# ============================================================
print('\n' + '=' * 60)
print('【6】外部验证 (3-31后88例)')
print('=' * 60)

Xext = X_ext_scaled if best_scaled else X_ext
r_ext = evaluate_model(best_model, Xext, y_ext, '外部验证')
print(f'  {best_name} 外部验证: AUC={r_ext["AUC"]}, F1={r_ext["F1"]}, Acc={r_ext["Accuracy"]}')
print(f'  混淆矩阵: TN={r_ext["TN"]}, FP={r_ext["FP"]}, FN={r_ext["FN"]}, TP={r_ext["TP"]}')

# 所有模型外部验证
ext_rows = []
for name, (model, use_scaled) in fitted_models.items():
    Xe = X_ext_scaled if use_scaled else X_ext
    r = evaluate_model(model, Xe, y_ext, name)
    ext_rows.append({
        '模型': name, 'AUC': r['AUC'], 'F1': r['F1'],
        'Accuracy': r['Accuracy'], 'Precision': r['Precision'], 'Recall': r['Recall'],
        'TN': r['TN'], 'FP': r['FP'], 'FN': r['FN'], 'TP': r['TP']
    })
ext_df_result = pd.DataFrame(ext_rows).sort_values('AUC', ascending=False)
ext_df_result.to_csv(os.path.join(BASE, 'external_validation_results.csv'), index=False, encoding='utf-8-sig')
print('外部验证结果已保存')

# ============================================================
# 9. 可视化
# ============================================================
print('\n' + '=' * 60)
print('【7】可视化')
print('=' * 60)

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

# 9.1 测试集ROC对比
fig, ax = plt.subplots(figsize=(8, 7))
for i, name in enumerate(models):
    r = results_test[name]
    ax.plot(r['fpr'], r['tpr'], color=colors[i], lw=2,
            label=f'{name} (AUC={r["AUC"]:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('假阳性率 (1-特异性)', fontsize=12)
ax.set_ylabel('真阳性率 (敏感性)', fontsize=12)
ax.set_title('测试集ROC曲线对比', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'roc_comparison.png'), dpi=200)
plt.close()
print('ROC对比图已保存')

# 9.2 训练集ROC对比
fig, ax = plt.subplots(figsize=(8, 7))
for i, name in enumerate(models):
    r = results_train[name]
    ax.plot(r['fpr'], r['tpr'], color=colors[i], lw=2,
            label=f'{name} (AUC={r["AUC"]:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('假阳性率 (1-特异性)', fontsize=12)
ax.set_ylabel('真阳性率 (敏感性)', fontsize=12)
ax.set_title('训练集ROC曲线对比（10折交叉验证）', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'roc_train.png'), dpi=200)
plt.close()

# 9.3 混淆矩阵热力图 - 每个模型单独生成（训练集+测试集并排）
def plot_cm_pair(cm_train, cm_test, model_name, save_path):
    """绘制训练集+测试集并排混淆矩阵"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    for ax, cm, title_suffix in [(ax1, cm_train, '(train) 混淆矩阵'), (ax2, cm_test, '(test) 混淆矩阵')]:
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        for ii in range(2):
            for jj in range(2):
                # 深色背景用白字，浅色用黑字
                val = cm[ii, jj]
                color = 'white' if val > cm.max() * 0.5 else 'black'
                ax.text(jj, ii, str(val), ha='center', va='center',
                        fontsize=18, fontweight='bold', color=color)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['0', '1'], fontsize=11)
        ax.set_yticklabels(['0', '1'], fontsize=11)
        ax.set_xlabel('Predicted label', fontsize=11)
        ax.set_ylabel('True label', fontsize=11)
        ax.set_title(f'{model_name} {title_suffix}', fontsize=12, fontweight='bold')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

for name in models:
    r_tr = results_train[name]
    r_te = results_test[name]
    cm_tr = np.array([[r_tr['TN'], r_tr['FP']], [r_tr['FN'], r_tr['TP']]])
    cm_te = np.array([[r_te['TN'], r_te['FP']], [r_te['FN'], r_te['TP']]])
    fname = name.replace(' ', '_').lower()
    save_p = os.path.join(BASE, f'cm_{fname}.png')
    plot_cm_pair(cm_tr, cm_te, name, save_p)

# 同时保留汇总图
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for idx, name in enumerate(models):
    ax = axes[idx // 3, idx % 3]
    r = results_test[name]
    cm = np.array([[r['TN'], r['FP']], [r['FN'], r['TP']]])
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    for ii in range(2):
        for jj in range(2):
            val = cm[ii, jj]
            color = 'white' if val > cm.max() * 0.5 else 'black'
            ax.text(jj, ii, str(val), ha='center', va='center', fontsize=16, fontweight='bold', color=color)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['0', '1'])
    ax.set_yticklabels(['0', '1'])
    ax.set_xlabel('Predicted label', fontsize=10)
    ax.set_ylabel('True label', fontsize=10)
    ax.set_title(f'{name}\nAUC={r["AUC"]:.3f}', fontsize=11)
plt.suptitle('测试集混淆矩阵汇总', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'confusion_matrices.png'), dpi=200, bbox_inches='tight')
plt.close()
print('混淆矩阵图已保存（单独+汇总）')

# 9.3b 每个模型单独ROC图（训练10折CV + 测试集）
for name in models:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    r_tr = results_train[name]
    r_te = results_test[name]
    ax.plot(r_tr['fpr'], r_tr['tpr'], color='#377eb8', lw=2,
            label=f'训练集10折CV (AUC={r_tr["AUC"]:.3f})')
    ax.plot(r_te['fpr'], r_te['tpr'], color='#e41a1c', lw=2,
            label=f'测试集 (AUC={r_te["AUC"]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('假阳性率 (1-特异性)', fontsize=11)
    ax.set_ylabel('真阳性率 (敏感性)', fontsize=11)
    ax.set_title(f'{name} ROC曲线', fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    fname = name.replace(' ', '_').lower()
    plt.savefig(os.path.join(BASE, f'roc_{fname}.png'), dpi=200)
    plt.close()
print('单模型ROC图已保存')

# 9.4 模型性能柱状对比图(训练集vs测试集)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
model_names = list(models.keys())
x = np.arange(len(model_names))
width = 0.35

for idx, metric in enumerate(['AUC', 'F1']):
    ax = axes[idx]
    train_vals = [results_train[n][metric] for n in model_names]
    test_vals = [results_test[n][metric] for n in model_names]
    bars1 = ax.bar(x - width/2, train_vals, width, label='训练集', color='#4daf4a', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_vals, width, label='测试集', color='#e41a1c', alpha=0.8)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'训练集 vs 测试集 {metric}', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=9)
    ax.legend()
    ax.set_ylim([0, 1.1])
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'train_test_comparison.png'), dpi=200)
plt.close()
print('训练/测试对比柱状图已保存')

# 9.5 外部验证ROC
fig, ax = plt.subplots(figsize=(8, 7))
for i, name in enumerate(models):
    model, use_scaled = fitted_models[name]
    Xe = X_ext_scaled if use_scaled else X_ext
    y_prob = model.predict_proba(Xe)[:, 1]
    fpr, tpr, _ = roc_curve(y_ext, y_prob)
    auc_val = roc_auc_score(y_ext, y_prob)
    ax.plot(fpr, tpr, color=colors[i], lw=2, label=f'{name} (AUC={auc_val:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('假阳性率 (1-特异性)', fontsize=12)
ax.set_ylabel('真阳性率 (敏感性)', fontsize=12)
ax.set_title('外部验证(3-31后)ROC曲线', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'external_roc.png'), dpi=200)
plt.close()
print('外部验证ROC图已保存')

# 9.6 多因素森林图
multi_full = pd.read_csv(os.path.join(BASE, 'overall_before_multivariate.csv'))
or_col = 'OR值' if 'OR值' in multi_full.columns else 'OR'

forest_data = multi_full[multi_full['P值'] < 0.1].copy()
if not forest_data.empty:
    fig, ax = plt.subplots(figsize=(10, max(4, len(forest_data) * 0.5 + 1)))
    y_pos = range(len(forest_data))
    ors = forest_data[or_col].values
    # 解析CI（兼容两种格式）
    if '95%CI下限' in forest_data.columns and '95%CI上限' in forest_data.columns:
        ci_low = forest_data['95%CI下限'].values
        ci_high = forest_data['95%CI上限'].values
    elif '95%CI' in forest_data.columns:
        ci_low, ci_high = [], []
        for ci in forest_data['95%CI']:
            parts = str(ci).split('-')
            try:
                ci_low.append(float(parts[0]))
                ci_high.append(float(parts[1]) if len(parts) > 1 else float(parts[0]))
            except:
                ci_low.append(ors[len(ci_low)])
                ci_high.append(ors[len(ci_high)])
        ci_low = np.array(ci_low)
        ci_high = np.array(ci_high)
    else:
        ci_low = ors * 0.5
        ci_high = ors * 2.0
    errors = [ors - ci_low, ci_high - ors]

    ax.errorbar(ors, y_pos, xerr=errors, fmt='o', color='#2166ac', markersize=8,
                capsize=4, elinewidth=2, markeredgewidth=2)
    ax.axvline(x=1, color='red', linestyle='--', lw=1.5, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(forest_data['变量'].values, fontsize=10)
    ax.set_xlabel('OR (95% CI)', fontsize=12)
    ax.set_title('多因素Logistic回归森林图 (3-31前224例)', fontsize=13)
    ax.set_xscale('log')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, 'forest_plot.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print('森林图已保存')

# ============================================================
# 10. 汇总输出
# ============================================================
print('\n' + '=' * 60)
print('【8】完成汇总')
print('=' * 60)

print(f"""
输出文件:
  ml_model_params.csv       - 模型参数设置
  ml_model_comparison.csv   - 模型间性能对比(测试集)
  ml_train_test_comparison.csv - 各模型训练/测试对比
  shap_features.csv         - SHAP特征重要性
  risk_factors_summary.csv  - 危险因素汇总(单因素+多因素+SHAP)
  external_validation_results.csv - 外部验证结果

  roc_comparison.png        - 测试集ROC对比
  roc_train.png            - 训练集ROC对比
  confusion_matrices.png   - 混淆矩阵热力图
  train_test_comparison.png - 训练/测试性能柱状图
  external_roc.png         - 外部验证ROC
  shap_summary.png         - SHAP summary plot
  forest_plot.png          - 多因素森林图

最佳模型: {best_name} (测试集AUC={comp_df.iloc[0]["AUC"]})
""")
