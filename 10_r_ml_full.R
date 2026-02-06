# -*- coding: utf-8 -*-
# 10_r_ml_full.R
# R语言完整6种机器学习模型分析
# 模型：Random Forest, SVM, XGBoost, Naive Bayes, Decision Tree, KNN

library(randomForest)
library(e1071)       # SVM + Naive Bayes
library(xgboost)
library(rpart)       # Decision Tree
library(class)       # KNN
library(pROC)

BASE <- "C:/Users/Administrator/Desktop/2.5.388"

# ============================================================
# 1. 数据准备
# ============================================================
cat("=" ,rep("=",58), "\n")
cat("【1】数据准备\n")
cat("=" ,rep("=",58), "\n")

train <- read.csv(file.path(BASE, "train_data.csv"), check.names=FALSE)
test <- read.csv(file.path(BASE, "test_data.csv"), check.names=FALSE)
ext <- read.csv(file.path(BASE, "external_validation_data.csv"), check.names=FALSE)

OUTCOME <- "潜在可预防VTE"

for(nm in c("train","test","ext")) {
  d <- get(nm)
  d[[OUTCOME]] <- as.integer(as.logical(d[[OUTCOME]]))
  if("住院天数" %in% names(d)) d$住院天数[d$住院天数 == -8] <- 8
  assign(nm, d)
}

drop_cols <- c("入院日期", "预防措施", "dataset")
for(nm in c("train","test","ext")) {
  d <- get(nm)
  d <- d[, !(names(d) %in% drop_cols)]
  assign(nm, d)
}

keep_numeric <- function(df) df[, sapply(df, is.numeric)]
train <- keep_numeric(train)
test <- keep_numeric(test)
ext <- keep_numeric(ext)

const_cols <- names(train)[sapply(train, function(x) length(unique(x[!is.na(x)])) <= 1)]
const_cols <- setdiff(const_cols, OUTCOME)
for(nm in c("train","test","ext")) {
  d <- get(nm)
  d <- d[, !(names(d) %in% const_cols)]
  assign(nm, d)
}

common <- Reduce(intersect, list(names(train), names(test), names(ext)))
common <- sort(common)
train <- train[, common]; test <- test[, common]; ext <- ext[, common]
train[is.na(train)] <- 0; test[is.na(test)] <- 0; ext[is.na(ext)] <- 0

y_train <- train[[OUTCOME]]; y_test <- test[[OUTCOME]]; y_ext <- ext[[OUTCOME]]
X_train <- train[, names(train) != OUTCOME]
X_test <- test[, names(test) != OUTCOME]
X_ext <- ext[, names(ext) != OUTCOME]

# 标准化(SVM, KNN, NB)
X_train_means <- colMeans(X_train); X_train_sds <- apply(X_train, 2, sd)
X_train_sds[X_train_sds == 0] <- 1
X_train_sc <- scale(X_train, center=X_train_means, scale=X_train_sds)
X_test_sc <- scale(X_test, center=X_train_means, scale=X_train_sds)
X_ext_sc <- scale(X_ext, center=X_train_means, scale=X_train_sds)

cat(sprintf("训练集: %d例 (%d阳性)\n", nrow(X_train), sum(y_train)))
cat(sprintf("测试集: %d例 (%d阳性)\n", nrow(X_test), sum(y_test)))
cat(sprintf("外部验证: %d例 (%d阳性)\n", nrow(X_ext), sum(y_ext)))
cat(sprintf("特征数: %d\n\n", ncol(X_train)))

# ============================================================
# 2. 评估函数
# ============================================================
evaluate <- function(y_true, y_pred, y_prob) {
  cm <- table(Actual=y_true, Predicted=y_pred)
  TP <- ifelse("1" %in% rownames(cm) & "1" %in% colnames(cm), cm["1","1"], 0)
  TN <- ifelse("0" %in% rownames(cm) & "0" %in% colnames(cm), cm["0","0"], 0)
  FP <- ifelse("0" %in% rownames(cm) & "1" %in% colnames(cm), cm["0","1"], 0)
  FN <- ifelse("1" %in% rownames(cm) & "0" %in% colnames(cm), cm["1","0"], 0)
  
  acc <- (TP+TN)/(TP+TN+FP+FN)
  prec <- ifelse(TP+FP > 0, TP/(TP+FP), 0)
  rec <- ifelse(TP+FN > 0, TP/(TP+FN), 0)
  f1 <- ifelse(prec+rec > 0, 2*prec*rec/(prec+rec), 0)
  auc_val <- as.numeric(auc(roc(y_true, y_prob, quiet=TRUE)))
  
  list(AUC=round(auc_val,4), F1=round(f1,4), Accuracy=round(acc,4),
       Precision=round(prec,4), Recall=round(rec,4),
       TN=TN, FP=FP, FN=FN, TP=TP)
}

# ============================================================
# 3. 训练6种模型
# ============================================================
cat("=" ,rep("=",58), "\n")
cat("【2】模型训练与评估\n")
cat("=" ,rep("=",58), "\n")

n_neg <- sum(y_train==0); n_pos <- sum(y_train==1)
results <- list()

# --- Random Forest ---
cat("\n--- Random Forest ---\n")
set.seed(42)
rf <- randomForest(x=X_train, y=as.factor(y_train), ntree=200,
                   mtry=floor(sqrt(ncol(X_train))), nodesize=2,
                   classwt=c("0"=1, "1"=n_neg/n_pos))
rf_prob_tr <- predict(rf, X_train, type="prob")[,2]
rf_prob_te <- predict(rf, X_test, type="prob")[,2]
rf_prob_ex <- predict(rf, X_ext, type="prob")[,2]
rf_pred_tr <- ifelse(rf_prob_tr > 0.5, 1, 0)
rf_pred_te <- ifelse(rf_prob_te > 0.5, 1, 0)
rf_pred_ex <- ifelse(rf_prob_ex > 0.5, 1, 0)
results[["Random Forest"]] <- list(
  train=evaluate(y_train, rf_pred_tr, rf_prob_tr),
  test=evaluate(y_test, rf_pred_te, rf_prob_te),
  ext=evaluate(y_ext, rf_pred_ex, rf_prob_ex),
  params="ntree=200, mtry=sqrt(p), nodesize=2, classwt=balanced"
)

# --- SVM ---
cat("--- SVM ---\n")
set.seed(42)
svm_model <- svm(x=X_train_sc, y=as.factor(y_train), kernel="radial",
                 cost=1.0, gamma=1/ncol(X_train_sc), probability=TRUE,
                 class.weights=c("0"=1, "1"=n_neg/n_pos))
svm_pred_tr <- predict(svm_model, X_train_sc, probability=TRUE)
svm_prob_tr <- attr(predict(svm_model, X_train_sc, probability=TRUE), "probabilities")[,"1"]
svm_pred_te <- predict(svm_model, X_test_sc, probability=TRUE)
svm_prob_te <- attr(predict(svm_model, X_test_sc, probability=TRUE), "probabilities")[,"1"]
svm_pred_ex <- predict(svm_model, X_ext_sc, probability=TRUE)
svm_prob_ex <- attr(predict(svm_model, X_ext_sc, probability=TRUE), "probabilities")[,"1"]
results[["SVM"]] <- list(
  train=evaluate(y_train, as.integer(as.character(svm_pred_tr)), svm_prob_tr),
  test=evaluate(y_test, as.integer(as.character(svm_pred_te)), svm_prob_te),
  ext=evaluate(y_ext, as.integer(as.character(svm_pred_ex)), svm_prob_ex),
  params="kernel=radial, cost=1.0, gamma=1/p, class.weights=balanced"
)

# --- XGBoost ---
cat("--- XGBoost ---\n")
dtrain <- xgb.DMatrix(data=as.matrix(X_train), label=y_train)
dtest <- xgb.DMatrix(data=as.matrix(X_test), label=y_test)
dext <- xgb.DMatrix(data=as.matrix(X_ext), label=y_ext)
set.seed(42)
xgb <- xgb.train(params=list(objective="binary:logistic", eval_metric="logloss",
                              max_depth=5, eta=0.1, scale_pos_weight=n_neg/n_pos),
                  data=dtrain, nrounds=200, verbose=0)
xgb_prob_tr <- predict(xgb, dtrain)
xgb_prob_te <- predict(xgb, dtest)
xgb_prob_ex <- predict(xgb, dext)
xgb_pred_tr <- ifelse(xgb_prob_tr > 0.5, 1, 0)
xgb_pred_te <- ifelse(xgb_prob_te > 0.5, 1, 0)
xgb_pred_ex <- ifelse(xgb_prob_ex > 0.5, 1, 0)
results[["XGBoost"]] <- list(
  train=evaluate(y_train, xgb_pred_tr, xgb_prob_tr),
  test=evaluate(y_test, xgb_pred_te, xgb_prob_te),
  ext=evaluate(y_ext, xgb_pred_ex, xgb_prob_ex),
  params=sprintf("max_depth=5, eta=0.1, scale_pos_weight=%.2f, nrounds=200", n_neg/n_pos)
)

# --- Naive Bayes ---
cat("--- Naive Bayes ---\n")
nb_model <- naiveBayes(x=as.data.frame(X_train_sc), y=as.factor(y_train))
nb_prob_tr <- predict(nb_model, as.data.frame(X_train_sc), type="raw")[,2]
nb_prob_te <- predict(nb_model, as.data.frame(X_test_sc), type="raw")[,2]
nb_prob_ex <- predict(nb_model, as.data.frame(X_ext_sc), type="raw")[,2]
nb_pred_tr <- ifelse(nb_prob_tr > 0.5, 1, 0)
nb_pred_te <- ifelse(nb_prob_te > 0.5, 1, 0)
nb_pred_ex <- ifelse(nb_prob_ex > 0.5, 1, 0)
results[["Naive Bayes"]] <- list(
  train=evaluate(y_train, nb_pred_tr, nb_prob_tr),
  test=evaluate(y_test, nb_pred_te, nb_prob_te),
  ext=evaluate(y_ext, nb_pred_ex, nb_prob_ex),
  params="GaussianNB (e1071::naiveBayes, 默认参数)"
)

# --- Decision Tree ---
cat("--- Decision Tree ---\n")
set.seed(42)
dt_model <- rpart(y ~ ., data=data.frame(y=as.factor(y_train), X_train),
                  method="class", control=rpart.control(maxdepth=8, minsplit=5, minbucket=2),
                  parms=list(prior=c(0.5, 0.5)))
dt_prob_tr <- predict(dt_model, data.frame(X_train), type="prob")[,2]
dt_prob_te <- predict(dt_model, data.frame(X_test), type="prob")[,2]
dt_prob_ex <- predict(dt_model, data.frame(X_ext), type="prob")[,2]
dt_pred_tr <- ifelse(dt_prob_tr > 0.5, 1, 0)
dt_pred_te <- ifelse(dt_prob_te > 0.5, 1, 0)
dt_pred_ex <- ifelse(dt_prob_ex > 0.5, 1, 0)
results[["Decision Tree"]] <- list(
  train=evaluate(y_train, dt_pred_tr, dt_prob_tr),
  test=evaluate(y_test, dt_pred_te, dt_prob_te),
  ext=evaluate(y_ext, dt_pred_ex, dt_prob_ex),
  params="maxdepth=8, minsplit=5, minbucket=2, prior=balanced (rpart)"
)

# --- KNN ---
cat("--- KNN ---\n")
set.seed(42)
knn_pred_tr <- knn(train=X_train_sc, test=X_train_sc, cl=y_train, k=7, prob=TRUE)
knn_prob_tr <- ifelse(knn_pred_tr == 1, attr(knn_pred_tr, "prob"), 1 - attr(knn_pred_tr, "prob"))
knn_pred_te <- knn(train=X_train_sc, test=X_test_sc, cl=y_train, k=7, prob=TRUE)
knn_prob_te <- ifelse(knn_pred_te == 1, attr(knn_pred_te, "prob"), 1 - attr(knn_pred_te, "prob"))
knn_pred_ex <- knn(train=X_train_sc, test=X_ext_sc, cl=y_train, k=7, prob=TRUE)
knn_prob_ex <- ifelse(knn_pred_ex == 1, attr(knn_pred_ex, "prob"), 1 - attr(knn_pred_ex, "prob"))
results[["KNN"]] <- list(
  train=evaluate(y_train, as.integer(as.character(knn_pred_tr)), knn_prob_tr),
  test=evaluate(y_test, as.integer(as.character(knn_pred_te)), knn_prob_te),
  ext=evaluate(y_ext, as.integer(as.character(knn_pred_ex)), knn_prob_ex),
  params="k=7, prob=TRUE (class::knn)"
)

# ============================================================
# 4. 输出结果
# ============================================================
cat("\n", "=" ,rep("=",58), "\n")
cat("【3】结果汇总\n")
cat("=" ,rep("=",58), "\n")

# 打印结果
for(name in names(results)) {
  r <- results[[name]]
  cat(sprintf("\n--- %s ---\n", name))
  cat(sprintf("  参数: %s\n", r$params))
  cat(sprintf("  训练集: AUC=%.4f, F1=%.4f, Acc=%.4f\n", r$train$AUC, r$train$F1, r$train$Accuracy))
  cat(sprintf("  测试集: AUC=%.4f, F1=%.4f, Acc=%.4f\n", r$test$AUC, r$test$F1, r$test$Accuracy))
  cat(sprintf("  外部验证: AUC=%.4f, F1=%.4f, Acc=%.4f\n", r$ext$AUC, r$ext$F1, r$ext$Accuracy))
  cat(sprintf("  测试集混淆矩阵: TN=%d, FP=%d, FN=%d, TP=%d\n", r$test$TN, r$test$FP, r$test$FN, r$test$TP))
}

# 保存CSV - 模型参数
params_df <- data.frame(
  模型=names(results),
  参数设置=sapply(results, function(r) r$params),
  stringsAsFactors=FALSE
)
write.csv(params_df, file.path(BASE, "r_ml_model_params.csv"), row.names=FALSE, fileEncoding="UTF-8")

# 保存CSV - 模型间对比(测试集)
comp_rows <- do.call(rbind, lapply(names(results), function(n) {
  r <- results[[n]]$test
  data.frame(模型=n, AUC=r$AUC, F1=r$F1, Accuracy=r$Accuracy,
             Precision=r$Precision, Recall=r$Recall, stringsAsFactors=FALSE)
}))
comp_rows <- comp_rows[order(-comp_rows$AUC), ]
write.csv(comp_rows, file.path(BASE, "r_ml_model_comparison.csv"), row.names=FALSE, fileEncoding="UTF-8")

# 保存CSV - 训练/测试对比
tt_rows <- do.call(rbind, lapply(names(results), function(n) {
  rbind(
    data.frame(模型=n, 数据集="训练集", AUC=results[[n]]$train$AUC, F1=results[[n]]$train$F1,
               Accuracy=results[[n]]$train$Accuracy, Precision=results[[n]]$train$Precision,
               Recall=results[[n]]$train$Recall, TN=results[[n]]$train$TN, FP=results[[n]]$train$FP,
               FN=results[[n]]$train$FN, TP=results[[n]]$train$TP, stringsAsFactors=FALSE),
    data.frame(模型=n, 数据集="测试集", AUC=results[[n]]$test$AUC, F1=results[[n]]$test$F1,
               Accuracy=results[[n]]$test$Accuracy, Precision=results[[n]]$test$Precision,
               Recall=results[[n]]$test$Recall, TN=results[[n]]$test$TN, FP=results[[n]]$test$FP,
               FN=results[[n]]$test$FN, TP=results[[n]]$test$TP, stringsAsFactors=FALSE)
  )
}))
write.csv(tt_rows, file.path(BASE, "r_ml_train_test_comparison.csv"), row.names=FALSE, fileEncoding="UTF-8")

# 保存CSV - 外部验证
ext_rows <- do.call(rbind, lapply(names(results), function(n) {
  r <- results[[n]]$ext
  data.frame(模型=n, AUC=r$AUC, F1=r$F1, Accuracy=r$Accuracy,
             Precision=r$Precision, Recall=r$Recall,
             TN=r$TN, FP=r$FP, FN=r$FN, TP=r$TP, stringsAsFactors=FALSE)
}))
ext_rows <- ext_rows[order(-ext_rows$AUC), ]
write.csv(ext_rows, file.path(BASE, "r_ml_external_validation.csv"), row.names=FALSE, fileEncoding="UTF-8")

# Python vs R 对比
cat("\n\n", "=" ,rep("=",58), "\n")
cat("【4】Python vs R 对比\n")
cat("=" ,rep("=",58), "\n")

py_comp <- read.csv(file.path(BASE, "ml_model_comparison.csv"), check.names=FALSE)
py_ext_comp <- read.csv(file.path(BASE, "external_validation_results.csv"), check.names=FALSE)

cross_rows <- list()
for(n in names(results)) {
  py_auc_test <- py_comp[py_comp$模型 == n, "AUC"]
  py_auc_ext <- py_ext_comp[py_ext_comp$模型 == n, "AUC"]
  r_auc_test <- results[[n]]$test$AUC
  r_auc_ext <- results[[n]]$ext$AUC
  
  if(length(py_auc_test) > 0 && length(py_auc_ext) > 0) {
    cross_rows[[length(cross_rows)+1]] <- data.frame(
      模型=n, 数据集="测试集",
      Python_AUC=py_auc_test, R_AUC=r_auc_test,
      差异=round(abs(py_auc_test - r_auc_test), 4),
      stringsAsFactors=FALSE
    )
    cross_rows[[length(cross_rows)+1]] <- data.frame(
      模型=n, 数据集="外部验证",
      Python_AUC=py_auc_ext, R_AUC=r_auc_ext,
      差异=round(abs(py_auc_ext - r_auc_ext), 4),
      stringsAsFactors=FALSE
    )
  }
}
cross_df <- do.call(rbind, cross_rows)
cross_df$一致性 <- ifelse(cross_df$差异 < 0.1, "一致", "有差异")
print(cross_df)
write.csv(cross_df, file.path(BASE, "r_vs_python_full_comparison.csv"), row.names=FALSE, fileEncoding="UTF-8")

cat("\n结论: ")
if(all(cross_df$差异 < 0.1)) {
  cat("Python与R所有6种模型AUC差异均<0.1，结果一致性良好\n")
} else {
  n_diff <- sum(cross_df$差异 >= 0.1)
  cat(sprintf("有%d项差异>=0.1，其余一致\n", n_diff))
}

cat("\n输出文件:\n")
cat("  r_ml_model_params.csv          - R模型参数\n")
cat("  r_ml_model_comparison.csv      - R模型间对比(测试集)\n")
cat("  r_ml_train_test_comparison.csv - R训练/测试对比\n")
cat("  r_ml_external_validation.csv   - R外部验证\n")
cat("  r_vs_python_full_comparison.csv - Python vs R全对比\n")
