# VTE（静脉血栓栓塞）潜在可预防因素分析

## 研究概述

本项目对某院 **312例VTE患者** 进行回顾性分析，探索**潜在可预防VTE**的独立影响因素，并构建机器学习风险预测模型。

以 **2025年3月31日** 为时间节点，将数据分为两个时期：

- **3-31及之前**：224例（阳性101例，45.1%） → 用于模型训练与测试
- **3-31之后**：88例（阳性63例，71.6%） → 用作独立外部验证集

其中训练集与测试集按 80/20 比例从3-31前数据中划分：训练集179例，测试集45例。

### 研究目标

1. 识别潜在可预防VTE的独立危险/保护因素（Logistic回归）
2. 构建机器学习风险预测模型并通过外部验证集验证
3. 对比3-31前后VTE预防管理指标变化趋势
4. 分析HA-VTE（医院相关性VTE）中规范预防的影响因素

### 关键概念

- **潜在可预防VTE**：目标变量（二分类），表示该VTE事件是否为可通过规范预防措施避免的
- **HA-VTE**：医院相关性VTE，由"医院相关性VTE"和"我院相关VTE"两列取OR合并确定
- **规范预防**：在HA-VTE子集分析中作为目标变量，研究哪些因素影响患者是否得到规范预防

---

## 分析流程

```
原始数据(312例)
   │
   ├── 01_data_preprocessing.py  ──→  数据清洗 → 训练/测试/外部验证集划分
   │
   ├── 06_comprehensive_analysis.py ──→ 4组Logistic回归 + 基线 + 前后对比
   │
   ├── 07_ml_analysis.py  ──→  6种ML模型(10折CV) + SHAP + 外部验证
   │
   └── 17_report_v3.py  ──→  自动生成Word报告(29表16图)
```

---

## 一、Logistic回归分析

共进行4组独立的Logistic回归分析，每组均包含：基线特征描述 → 单因素回归 → 多因素回归（含VIF共线性检验）。

### 1.1 总体分析（目标：潜在可预防VTE）

**3-31前 224例：** 多因素显著变量4个

| 变量 | OR值 | P值 | 含义 |
|------|------|-----|------|
| 90天内院内就诊/住院史 | 7.339 | <0.001 | 独立危险因素 |
| 出血风险评估 | 0.224 | <0.001 | 独立保护因素 |
| 是否机械预防 | 0.279 | 0.001 | 独立保护因素 |
| 入院科室_康复科 | 4.649 | 0.010 | 独立危险因素 |

**3-31后 88例：** 多因素显著变量2个（是否机械预防 OR=0.032，规范预防 OR=0.107）

### 1.2 HA-VTE子集分析（目标：规范预防）

| 分组 | 样本量 | 多因素显著变量 |
|------|--------|----------------|
| 3-31前 HA-VTE | 153例 | 白细胞计数、是否机械预防、VTE中高危评分（3个） |
| 3-31后 HA-VTE | 66例 | 是否步速缓慢（1个） |

> 注：HA-VTE的判定采用"医院相关性VTE"和"我院相关VTE"两列OR合并，3-31后为66例而非单列的32例。

### 1.3 前后对比

| 指标 | 3-31前 | 3-31后 | P值 |
|------|--------|--------|-----|
| HA-VTE占比 | 153/224 (68.3%) | 66/88 (75.0%) | 0.305 |
| 潜在可预防VTE占全部VTE | 101/224 (45.1%) | 63/88 (71.6%) | <0.001 |
| 潜在可预防VTE在HA-VTE中占比 | 101/153 (66.0%) | 42/66 (63.6%) | 0.854 |
| 规范预防率（潜在可预防VTE中） | 15/101 (14.9%) | 9/63 (14.3%) | 1.000 |

---

## 二、机器学习预测

### 2.1 模型配置

6种模型，训练集采用 **10折分层交叉验证** 评估，避免过拟合导致AUC虚高：

| 模型 | 关键参数 |
|------|----------|
| Random Forest | n_estimators=200, max_depth=10, class_weight=balanced |
| SVM | kernel=rbf, C=1.0, gamma=scale, class_weight=balanced |
| XGBoost | RandomizedSearchCV 60组调参（10折CV），参数数据驱动 |
| Naive Bayes | GaussianNB 默认参数 |
| Decision Tree | max_depth=8, min_samples_split=5, class_weight=balanced |
| KNN | n_neighbors=7, weights=distance, metric=minkowski |

### 2.2 性能对比

| 模型 | 训练集AUC (10折CV) | 测试集AUC | 外部验证AUC |
|------|:--:|:--:|:--:|
| Decision Tree | 0.666 | **0.779** | 0.616 |
| XGBoost | 0.791 | **0.776** | 0.577 |
| Random Forest | 0.757 | 0.644 | 0.678 |
| SVM | 0.669 | 0.596 | 0.697 |
| Naive Bayes | 0.527 | 0.535 | 0.513 |
| KNN | 0.530 | 0.500 | 0.643 |

测试集最佳：Decision Tree (AUC=0.779)；外部验证最佳：SVM (AUC=0.697)。

### 2.3 SHAP可解释性 Top10

基于XGBoost模型的SHAP分析：

| 排名 | 变量 | SHAP均值 |
|------|------|----------|
| 1 | 90天内院内就诊/住院史 | 0.796 |
| 2 | VTE诊断与入院日期>24h | 0.367 |
| 3 | 出血风险评估 | 0.339 |
| 4 | 活化凝血酶原时间APTT | 0.286 |
| 5 | 入院收缩压 | 0.197 |
| 6 | VTE中高危评分 | 0.193 |
| 7 | 年龄 | 0.171 |
| 8 | 入院科室_康复科 | 0.171 |
| 9 | 炎症 | 0.148 |
| 10 | 身高 | 0.125 |

SHAP排名与Logistic回归高度一致：「90天内院内就诊/住院史」和「出血风险评估」在两种方法中均为最重要因素。

---

## 三、数据泄漏处理

为防止预防相关变量（参与"潜在可预防VTE"目标变量定义）污染模型，已在**回归分析和ML建模前**排除 **76个** 泄漏变量：

| 类别 | 变量举例 | 数量 |
|------|----------|------|
| 定义前提 | 医院相关性VTE、我院相关VTE | 2 |
| 预防状态 | 规范预防、是否机械预防、是否药物预防 | 3 |
| 预防措施哑变量 | 预防措施_无预防、预防措施_机械预防等 | 4 |
| 预防日期差值（哑变量） | VTE首次中高危评分日期与机械/药物预防日期差值_* | 60+ |
| 其他 | 基础预防措施、首次VTE中高风险评分日期与预防日期差值（原始） | 5 |

> 完整列表见 `excluded_leak_variables.csv`

基线描述性统计表中**保留**这些变量用于展示，仅在回归建模和ML特征中排除。

---

## 四、项目结构

### 核心脚本（按执行顺序）

| # | 脚本 | 功能 | 输入 | 主要输出 |
|---|------|------|------|----------|
| 1 | `01_data_preprocessing.py` | 数据预处理、One-Hot编码、训练/测试集划分 | 原始Excel | `final_processed_data_full_pipeline.csv`, `train/test/external_validation_data.csv` |
| 2 | `06_comprehensive_analysis.py` | 4组统计分析（基线+单因素+多因素）、前后对比 | 预处理后CSV | `overall_*.csv`, `havte_*.csv`, `compare_before_after_v2.csv` |
| 3 | `07_ml_analysis.py` | 6种ML模型训练(10折CV)、SHAP分析、外部验证、可视化 | 训练/测试/外部集 | `ml_*.csv`, `shap_*.csv/png`, `roc_*.png`, `cm_*.png` |
| 4 | `17_report_v3.py` | 自动生成Word分析报告（所有数字从CSV动态读取） | 全部CSV+PNG | `VTE影响因素分析报告.docx`（29表16图） |

### 辅助脚本

| 脚本 | 用途 |
|------|------|
| `08_leak_comparison.py` | 泄漏变量排除前后的ML性能对比实验 |
| `16_gen_all_figures.py` | 额外图表生成（SHAP交互热力图等） |
| `04_R_analysis.R` | R语言验证统计分析一致性 |
| `08_r_validation.R` / `10_r_ml_full.R` | R语言ML交叉验证 |
| `05_spss_prepare.py` | 生成SPSS语法文件 |
| `02_python_baseline.py` / `03_python_logistic.py` | 早期版本（已被`06`替代） |

### 输出文件说明

**Logistic回归结果（4组 × 3类 = 12个CSV）：**
```
overall_before_baseline.csv       ← 3-31前224例 基线特征
overall_before_univariate.csv     ← 3-31前224例 单因素
overall_before_multivariate.csv   ← 3-31前224例 多因素

overall_after_baseline.csv        ← 3-31后88例 基线特征
overall_after_univariate.csv      ← 3-31后88例 单因素
overall_after_multivariate.csv    ← 3-31后88例 多因素

havte_before_baseline.csv         ← 3-31前HA-VTE 153例（目标：规范预防）
havte_before_univariate.csv
havte_before_multivariate.csv

havte_after_baseline.csv          ← 3-31后HA-VTE 66例（目标：规范预防）
havte_after_univariate.csv
havte_after_multivariate.csv
```

**机器学习结果：**
- `ml_model_comparison.csv` — 6个模型测试集性能对比
- `ml_train_test_comparison.csv` — 各模型训练集(10折CV)与测试集对比
- `ml_model_params.csv` — 模型参数设置
- `external_validation_results.csv` — 外部验证（3-31后88例）结果
- `shap_features.csv` — SHAP特征重要性排名
- `risk_factors_summary.csv` — 危险因素汇总（单因素OR + 多因素OR + SHAP）
- `excluded_leak_variables.csv` — 排除的76个泄漏变量清单

**可视化（16张图）：**
- `roc_comparison.png` — 测试集6模型ROC曲线对比
- `roc_train.png` — 训练集6模型ROC对比
- `roc_{model}.png` — 各模型单独ROC（训练+测试）
- `cm_{model}.png` — 各模型混淆矩阵（训练+测试并排）
- `confusion_matrices.png` — 6模型混淆矩阵汇总
- `train_test_comparison.png` — 训练/测试AUC+F1柱状对比
- `external_roc.png` — 外部验证ROC
- `shap_summary.png` — SHAP蜂群图
- `forest_plot.png` — 多因素Logistic回归森林图

**数据文件：**
- `final_processed_data_full_pipeline.csv` — 完整预处理数据（312例，唯一数据源）
- `train_data.csv` — 训练集（179例）
- `test_data.csv` — 测试集（45例）
- `external_validation_data.csv` — 外部验证集（88例，3-31后数据）

**报告：**
- `VTE影响因素分析报告.docx` — 完整分析报告（29个表格、16张图片）

---

## 五、统计方法

### 基线分析
- 分类变量：χ²检验（期望频数<5时改用Fisher精确检验）
- 连续变量：Shapiro-Wilk正态性检验 → 正态用独立样本t检验，非正态用Mann-Whitney U检验

### 单因素分析
- 逐变量二元Logistic回归，报告OR、95%CI、P值

### 多因素分析
- 共线性检查（Pearson相关系数 r>0.8 的变量对取P值较大者剔除）
- EPV原则（Events Per Variable）约束入模变量数
- 二元Logistic回归，报告VIF检验结果

### 机器学习
- 训练集评估：10折分层交叉验证（每个样本恰好被验证1次）
- XGBoost调参：RandomizedSearchCV 60组参数组合，10折CV
- 可解释性：SHAP TreeExplainer（基于XGBoost）
- 外部验证：3-31后88例独立数据，模型训练过程中未参与

---

## 六、数据预处理说明

1. **极端值修正**：住院天数 -8天 → 8天（录入错误）
2. **变量合并**：「90天前是否我院就诊」和「本次入院前90天有无住院史」合并为「90天内院内就诊/住院史」（OR逻辑）
3. **HA-VTE合并**：「医院相关性VTE」和「我院相关VTE」取OR合并
4. **肺栓塞规则**：主要疾病诊断=肺栓塞的患者，潜在可预防VTE强制为0
5. **编码处理**：二分类变量0/1编码，多分类变量One-Hot编码
6. **常量列删除**：仅1个唯一值的列移除
7. **泄漏变量排除**：76个预防相关变量在建模前排除（详见第三节）
8. **标准化**：SVM和KNN模型使用StandardScaler

---

## 七、环境依赖

```
Python >= 3.10
```

核心依赖：
```
pandas >= 2.0
numpy >= 1.24
scipy >= 1.10
statsmodels >= 0.14
scikit-learn >= 1.3
xgboost >= 2.0
shap >= 0.43
matplotlib >= 3.7
python-docx >= 0.8
```

R语言（可选，用于交叉验证）：
```
R >= 4.0
tidyverse, caret, pROC, randomForest, xgboost, e1071, class, naivebayes
```

---

## 八、快速复现

```bash
# 1. 数据预处理（需要原始Excel文件）
python 01_data_preprocessing.py

# 2. 统计分析（Logistic回归 + 前后对比）
python 06_comprehensive_analysis.py

# 3. 机器学习（训练 + SHAP + 外部验证 + 可视化）
python 07_ml_analysis.py

# 4. 生成报告
python 17_report_v3.py
```

报告中所有数字均从CSV动态读取，无硬编码。修改数据后重新运行即可自动更新。
