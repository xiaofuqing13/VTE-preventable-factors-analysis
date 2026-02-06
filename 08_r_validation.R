# -*- coding: utf-8 -*-
# 08_r_validation.R
# R语言验证RandomForest和XGBoost结果一致性

library(randomForest)
library(xgboost)
library(pROC)

BASE <- "C:/Users/Administrator/Desktop/2.5.388"

# 1. 读取数据
cat("========== 数据读取 ==========\n")
train <- read.csv(file.path(BASE, "train_data.csv"), check.names=FALSE)
test <- read.csv(file.path(BASE, "test_data.csv"), check.names=FALSE)
ext <- read.csv(file.path(BASE, "external_validation_data.csv"), check.names=FALSE)

OUTCOME <- "潜在可预防VTE"

# 目标变量转换
for(nm in c("train","test","ext")) {
  d <- get(nm)
  d[[OUTCOME]] <- as.integer(as.logical(d[[OUTCOME]]))
  # 修正住院天数
  if("住院天数" %in% names(d)) {
    d$住院天数[d$住院天数 == -8] <- 8
  }
  assign(nm, d)
}

# 删除非数值列
drop_cols <- c("入院日期", "预防措施", "dataset")
for(nm in c("train","test","ext")) {
  d <- get(nm)
  d <- d[, !(names(d) %in% drop_cols)]
  assign(nm, d)
}

# 只保留数值列
keep_numeric <- function(df) {
  num_cols <- sapply(df, is.numeric)
  df[, num_cols]
}

train <- keep_numeric(train)
test <- keep_numeric(test)
ext <- keep_numeric(ext)

# 删除常量列
const_cols <- names(train)[sapply(train, function(x) length(unique(x[!is.na(x)])) <= 1)]
const_cols <- setdiff(const_cols, OUTCOME)
for(nm in c("train","test","ext")) {
  d <- get(nm)
  d <- d[, !(names(d) %in% const_cols)]
  assign(nm, d)
}

# 取公共列
common <- Reduce(intersect, list(names(train), names(test), names(ext)))
common <- sort(common)
train <- train[, common]
test <- test[, common]
ext <- ext[, common]

# 填充NA
train[is.na(train)] <- 0
test[is.na(test)] <- 0
ext[is.na(ext)] <- 0

# 分离特征和目标
y_train <- train[[OUTCOME]]
y_test <- test[[OUTCOME]]
y_ext <- ext[[OUTCOME]]
X_train <- train[, names(train) != OUTCOME]
X_test <- test[, names(test) != OUTCOME]
X_ext <- ext[, names(ext) != OUTCOME]

cat(sprintf("训练集: %d例 (%d阳性)\n", nrow(X_train), sum(y_train)))
cat(sprintf("测试集: %d例 (%d阳性)\n", nrow(X_test), sum(y_test)))
cat(sprintf("外部验证: %d例 (%d阳性)\n", nrow(X_ext), sum(y_ext)))
cat(sprintf("特征数: %d\n", ncol(X_train)))

# 2. Random Forest
cat("\n========== Random Forest ==========\n")
set.seed(42)
rf_model <- randomForest(
  x = X_train, y = as.factor(y_train),
  ntree = 200, mtry = floor(sqrt(ncol(X_train))),
  nodesize = 2, classwt = c("0"=1, "1"=sum(y_train==0)/sum(y_train==1))
)

rf_pred_train <- predict(rf_model, X_train, type="prob")[,2]
rf_pred_test <- predict(rf_model, X_test, type="prob")[,2]
rf_pred_ext <- predict(rf_model, X_ext, type="prob")[,2]

rf_auc_train <- auc(roc(y_train, rf_pred_train, quiet=TRUE))
rf_auc_test <- auc(roc(y_test, rf_pred_test, quiet=TRUE))
rf_auc_ext <- auc(roc(y_ext, rf_pred_ext, quiet=TRUE))

cat(sprintf("RF 训练集 AUC: %.4f\n", rf_auc_train))
cat(sprintf("RF 测试集 AUC: %.4f\n", rf_auc_test))
cat(sprintf("RF 外部验证 AUC: %.4f\n", rf_auc_ext))

# 3. XGBoost
cat("\n========== XGBoost ==========\n")
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
dtest <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)
dext <- xgb.DMatrix(data = as.matrix(X_ext), label = y_ext)

n_neg <- sum(y_train == 0)
n_pos <- sum(y_train == 1)

params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  max_depth = 5,
  eta = 0.1,
  scale_pos_weight = n_neg / n_pos
)

set.seed(42)
xgb_model <- xgb.train(
  params = params, data = dtrain,
  nrounds = 200, verbose = 0
)

xgb_pred_train <- predict(xgb_model, dtrain)
xgb_pred_test <- predict(xgb_model, dtest)
xgb_pred_ext <- predict(xgb_model, dext)

xgb_auc_train <- auc(roc(y_train, xgb_pred_train, quiet=TRUE))
xgb_auc_test <- auc(roc(y_test, xgb_pred_test, quiet=TRUE))
xgb_auc_ext <- auc(roc(y_ext, xgb_pred_ext, quiet=TRUE))

cat(sprintf("XGBoost 训练集 AUC: %.4f\n", xgb_auc_train))
cat(sprintf("XGBoost 测试集 AUC: %.4f\n", xgb_auc_test))
cat(sprintf("XGBoost 外部验证 AUC: %.4f\n", xgb_auc_ext))

# 4. 对比Python结果
cat("\n========== Python vs R 对比 ==========\n")
py_results <- read.csv(file.path(BASE, "ml_train_test_comparison.csv"), check.names=FALSE)

py_rf_train <- py_results[py_results$模型=="Random Forest" & py_results$数据集=="训练集", "AUC"]
py_rf_test <- py_results[py_results$模型=="Random Forest" & py_results$数据集=="测试集", "AUC"]
py_xgb_train <- py_results[py_results$模型=="XGBoost" & py_results$数据集=="训练集", "AUC"]
py_xgb_test <- py_results[py_results$模型=="XGBoost" & py_results$数据集=="测试集", "AUC"]

py_ext <- read.csv(file.path(BASE, "external_validation_results.csv"), check.names=FALSE)
py_rf_ext <- py_ext[py_ext$模型=="Random Forest", "AUC"]
py_xgb_ext <- py_ext[py_ext$模型=="XGBoost", "AUC"]

comparison <- data.frame(
  模型 = c("RF","RF","RF","XGBoost","XGBoost","XGBoost"),
  数据集 = rep(c("训练集","测试集","外部验证"), 2),
  Python_AUC = c(py_rf_train, py_rf_test, py_rf_ext, py_xgb_train, py_xgb_test, py_xgb_ext),
  R_AUC = c(rf_auc_train, rf_auc_test, rf_auc_ext, xgb_auc_train, xgb_auc_test, xgb_auc_ext)
)
comparison$差异 <- abs(comparison$Python_AUC - comparison$R_AUC)
comparison$一致性 <- ifelse(comparison$差异 < 0.05, "一致", "有差异")

print(comparison)
write.csv(comparison, file.path(BASE, "r_validation_comparison.csv"), row.names=FALSE, fileEncoding="UTF-8")

cat("\n验证结论：\n")
if(all(comparison$差异 < 0.05)) {
  cat("Python与R结果一致性良好（AUC差异均<0.05）\n")
} else {
  cat("部分结果存在差异，请查看详细对比表\n")
}
cat("验证结果已保存至 r_validation_comparison.csv\n")
