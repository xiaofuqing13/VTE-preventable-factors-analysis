# -*- coding: utf-8 -*-
"""
16_gen_all_figures.py
生成报告所需的全部图表（含每个模型单独的混淆矩阵/ROC + SHAP特征重要性/交互热力图）
"""
import os, warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
rcParams['axes.unicode_minus'] = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             accuracy_score, precision_score, recall_score, f1_score)
import xgboost as xgb

BASE = r'C:\Users\Administrator\Desktop\2.5.388'

# ========== 1. 加载数据 ==========
print('Loading data...')
train = pd.read_csv(os.path.join(BASE, 'train_data.csv'))
test = pd.read_csv(os.path.join(BASE, 'test_data.csv'))
ext = pd.read_csv(os.path.join(BASE, 'external_validation_data.csv'))

target = '潜在可预防VTE'
drop_cols = ['入院日期', '预防措施', 'dataset']
leak_keywords = ['预防', '医院相关性VTE', '我院相关VTE']

def prep(df):
    d = df.copy()
    for c in drop_cols:
        if c in d.columns:
            d = d.drop(columns=[c])
    y = d[target]
    X = d.drop(columns=[target])
    # 排除泄漏变量
    leak_cols = [c for c in X.columns if any(kw in c for kw in leak_keywords)]
    X = X.drop(columns=leak_cols, errors='ignore')
    # 只保留数值列
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(0)
    # 删除常数列
    X = X.loc[:, X.nunique() > 1]
    return X, y

X_train, y_train = prep(train)
X_test, y_test = prep(test)
X_ext, y_ext = prep(ext)

# 取公共列
common = list(set(X_train.columns) & set(X_test.columns) & set(X_ext.columns))
common.sort()
X_train = X_train[common]
X_test = X_test[common]
X_ext = X_ext[common]

print(f'Features: {len(common)}, Train: {len(y_train)}, Test: {len(y_test)}, Ext: {len(y_ext)}')

# 标准化（for SVM, KNN）
scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=common, index=X_train.index)
X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=common, index=X_test.index)

# ========== 2. 训练模型 ==========
print('Training models...')
n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
spw = n_neg / n_pos

models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5,
                                             min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1),
    'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, class_weight='balanced', random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.01,
                                  subsample=0.9, colsample_bytree=1.0, min_child_weight=3,
                                  reg_alpha=0.01, reg_lambda=0.5,
                                  scale_pos_weight=spw, use_label_encoder=False,
                                  eval_metric='logloss', random_state=42),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(max_depth=8, min_samples_split=5, min_samples_leaf=2,
                                             class_weight='balanced', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=7, weights='distance', metric='minkowski', p=2)
}

results = {}
for name, model in models.items():
    print(f'  Training {name}...')
    if name in ['SVM', 'KNN']:
        model.fit(X_train_sc, y_train)
        y_train_prob = model.predict_proba(X_train_sc)[:, 1]
        y_test_prob = model.predict_proba(X_test_sc)[:, 1]
        y_train_pred = model.predict(X_train_sc)
        y_test_pred = model.predict(X_test_sc)
    else:
        model.fit(X_train, y_train)
        y_train_prob = model.predict_proba(X_train)[:, 1]
        y_test_prob = model.predict_proba(X_test)[:, 1]
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

    results[name] = {
        'model': model,
        'y_train_prob': y_train_prob,
        'y_test_prob': y_test_prob,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
    }

# ========== 3. 每个模型单独的混淆矩阵 ==========
print('Generating per-model confusion matrices...')
for name, res in results.items():
    cm = confusion_matrix(y_test, res['y_test_pred'])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(f'{name} 测试集混淆矩阵', fontsize=12)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=16,
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    ax.set_xlabel('预测值', fontsize=10)
    ax.set_ylabel('真实值', fontsize=10)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['阴性(0)', '阳性(1)'])
    ax.set_yticklabels(['阴性(0)', '阳性(1)'])
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    fname = name.replace(' ', '_').lower()
    fig.savefig(os.path.join(BASE, f'cm_{fname}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

# ========== 4. 每个模型单独的ROC曲线（训练+测试） ==========
print('Generating per-model ROC curves...')
for name, res in results.items():
    fig, ax = plt.subplots(figsize=(5, 4.5))
    # 训练集
    fpr_tr, tpr_tr, _ = roc_curve(y_train, res['y_train_prob'])
    auc_tr = auc(fpr_tr, tpr_tr)
    ax.plot(fpr_tr, tpr_tr, 'b-', lw=2, label=f'训练集 AUC={auc_tr:.3f}')
    # 测试集
    fpr_te, tpr_te, _ = roc_curve(y_test, res['y_test_prob'])
    auc_te = auc(fpr_te, tpr_te)
    ax.plot(fpr_te, tpr_te, 'r-', lw=2, label=f'测试集 AUC={auc_te:.3f}')
    ax.plot([0,1], [0,1], 'k--', lw=1)
    ax.set_xlabel('1 - 特异度 (FPR)', fontsize=10)
    ax.set_ylabel('灵敏度 (TPR)', fontsize=10)
    ax.set_title(f'{name} ROC曲线', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.05])
    plt.tight_layout()
    fname = name.replace(' ', '_').lower()
    fig.savefig(os.path.join(BASE, f'roc_{fname}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

# ========== 5. SHAP分析（XGBoost） ==========
print('Generating SHAP figures...')
import shap
xgb_model = results['XGBoost']['model']
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# 图A: 特征重要性柱状图 (mean |SHAP|)
shap_abs_mean = np.abs(shap_values).mean(axis=0)
feat_imp = pd.Series(shap_abs_mean, index=common).sort_values(ascending=False)
top20 = feat_imp.head(20)

fig, ax = plt.subplots(figsize=(8, 6))
top20.iloc[::-1].plot(kind='barh', ax=ax, color='steelblue')
ax.set_xlabel('Mean |SHAP Value|', fontsize=11)
ax.set_title('XGBoost SHAP 特征重要性 Top20', fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(BASE, 'shap_importance.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# 图B: SHAP Summary Plot (蜂群图)
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values, X_test, max_display=20, show=False)
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'shap_summary_v2.png'), dpi=150, bbox_inches='tight')
plt.close('all')

# 图C: SHAP 交互热力图 (top10特征的交互)
print('Generating SHAP interaction heatmap...')
try:
    shap_interaction = explainer.shap_interaction_values(X_test)
    top10_idx = [list(common).index(f) for f in top20.index[:10] if f in common]
    top10_names = [common[i] for i in top10_idx]
    interact_mean = np.abs(shap_interaction).mean(axis=0)
    interact_sub = interact_mean[np.ix_(top10_idx, top10_idx)]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(interact_sub, cmap='YlOrRd')
    ax.set_xticks(range(len(top10_names)))
    ax.set_yticks(range(len(top10_names)))
    short_names = [n[:12] for n in top10_names]
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_title('SHAP 交互效应热力图 (Top10特征)', fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, label='Mean |SHAP interaction|')
    plt.tight_layout()
    fig.savefig(os.path.join(BASE, 'shap_interaction.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  SHAP interaction heatmap saved.')
except Exception as e:
    print(f'  SHAP interaction failed: {e}')
    # 生成替代图：SHAP dependence plot top3
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, feat in enumerate(top20.index[:3]):
        shap.dependence_plot(feat, shap_values, X_test, ax=axes[i], show=False)
    plt.tight_layout()
    fig.savefig(os.path.join(BASE, 'shap_interaction.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

# ========== 6. 3-31前后基线对比 ==========
print('Computing before/after baseline comparison...')
raw_before = pd.read_csv(os.path.join(BASE, 'train_data.csv'))
raw_test = pd.read_csv(os.path.join(BASE, 'test_data.csv'))
raw_before = pd.concat([raw_before, raw_test], ignore_index=True)
raw_after = pd.read_csv(os.path.join(BASE, 'external_validation_data.csv'))

# 关键基线变量对比
key_vars = ['年龄', '住院天数', '入院收缩压', 'VTE中高危评分',
            '医院相关性VTE', '90天前是否我院就诊', '是否机械预防',
            '出血风险评估（1/0）', '规范预防', '潜在可预防VTE']
comparison_rows = []
for v in key_vars:
    if v not in raw_before.columns or v not in raw_after.columns:
        continue
    b = raw_before[v].dropna()
    a = raw_after[v].dropna()
    nuniq_b = b.nunique()
    if nuniq_b <= 2:
        # 分类
        b_n = int(b.sum()); b_tot = len(b); b_pct = b_n/b_tot*100
        a_n = int(a.sum()); a_tot = len(a); a_pct = a_n/a_tot*100
        from scipy.stats import chi2_contingency
        table = np.array([[b_n, b_tot-b_n],[a_n, a_tot-a_n]])
        try:
            chi2, p, _, _ = chi2_contingency(table, correction=False)
            stat_str = f'{chi2:.3f}'
        except:
            stat_str = '-'; p = 1.0
        comparison_rows.append([v, f'{b_n}/{b_tot} ({b_pct:.1f}%)',
                                f'{a_n}/{a_tot} ({a_pct:.1f}%)', stat_str, f'{p:.4f}'])
    else:
        # 连续
        from scipy.stats import ttest_ind
        t_stat, p = ttest_ind(b, a, equal_var=False)
        comparison_rows.append([v, f'{b.mean():.2f}±{b.std():.2f}',
                                f'{a.mean():.2f}±{a.std():.2f}', f'{t_stat:.3f}', f'{p:.4f}'])

comp_df = pd.DataFrame(comparison_rows, columns=['变量', '3-31前(n=224)', '3-31后(n=88)', '统计量', 'P值'])
comp_df.to_csv(os.path.join(BASE, 'baseline_comparison_periods.csv'), index=False, encoding='utf-8-sig')
print(f'Baseline comparison saved: {len(comp_df)} variables')

# ========== 7. 保存每个模型的训练/测试性能CSV ==========
print('Saving per-model performance...')
perf_data = []
for name, res in results.items():
    for dataset, y_true, y_pred, y_prob in [
        ('训练集', y_train, res['y_train_pred'], res['y_train_prob']),
        ('测试集', y_test, res['y_test_pred'], res['y_test_prob'])
    ]:
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr_v, tpr_v, _ = roc_curve(y_true, y_prob)
        auc_v = auc(fpr_v, tpr_v)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        perf_data.append({
            '模型': name, '数据集': dataset,
            '准确率': accuracy_score(y_true, y_pred),
            '精确率': precision_score(y_true, y_pred, zero_division=0),
            '灵敏度': recall_score(y_true, y_pred, zero_division=0),
            '特异度': spec,
            'F1': f1_score(y_true, y_pred, zero_division=0),
            'AUC': auc_v,
            'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
        })
perf_df = pd.DataFrame(perf_data)
perf_df.to_csv(os.path.join(BASE, 'ml_perf_full.csv'), index=False, encoding='utf-8-sig')
print('Done! All figures generated.')
