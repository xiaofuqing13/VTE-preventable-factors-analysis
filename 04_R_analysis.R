# -*- coding: utf-8 -*-
setwd("C:/Users/Administrator/Desktop/2.5.388")

if (!require("tableone")) install.packages("tableone", repos="https://cloud.r-project.org")
if (!require("broom")) install.packages("broom", repos="https://cloud.r-project.org")
library(tableone)
library(broom)

train_data <- read.csv("train_data.csv", fileEncoding = "UTF-8")
var_info <- read.csv("variable_info.csv", fileEncoding = "UTF-8")

outcome <- "潜在可预防VTE"
train_data[[outcome]] <- as.factor(train_data[[outcome]])

continuous_vars <- var_info$variable[var_info$type == "continuous"]
categorical_vars <- var_info$variable[var_info$type == "categorical"]
continuous_vars <- continuous_vars[continuous_vars %in% names(train_data)]
categorical_vars <- categorical_vars[categorical_vars %in% names(train_data)]
continuous_vars <- head(continuous_vars, 20)

valid_cat <- c()
for (v in categorical_vars) {
  if (v %in% names(train_data) && is.numeric(train_data[[v]])) {
    mean_val <- mean(train_data[[v]], na.rm = TRUE)
    if (!is.na(mean_val) && mean_val >= 0.05 && mean_val <= 0.95) {
      valid_cat <- c(valid_cat, v)
    }
  }
}
categorical_vars <- head(valid_cat, 50)
all_vars <- c(continuous_vars, categorical_vars)

tryCatch({
  tab1 <- CreateTableOne(vars = all_vars, strata = outcome, data = train_data, factorVars = categorical_vars)
  tab1_print <- print(tab1, printToggle = FALSE, noSpaces = TRUE, showAllLevels = FALSE, test = TRUE)
  write.csv(tab1_print, "R_baseline_results.csv")
}, error = function(e) {})

all_predictors <- names(train_data)[names(train_data) != outcome]
all_predictors <- all_predictors[sapply(train_data[all_predictors], is.numeric)]
univariate_results <- data.frame()

for (var in all_predictors) {
  tryCatch({
    if (sd(train_data[[var]], na.rm = TRUE) == 0) next
    formula_str <- sprintf("`%s` ~ `%s`", outcome, var)
    model <- glm(as.formula(formula_str), data = train_data, family = binomial)
    result <- tidy(model, conf.int = TRUE, exponentiate = TRUE)
    result <- result[result$term != "(Intercept)", ]
    if (nrow(result) > 0) {
      result$variable <- var
      univariate_results <- rbind(univariate_results, result)
    }
  }, error = function(e) {})
}

univariate_results <- univariate_results[order(univariate_results$p.value), ]
names(univariate_results) <- c("term", "OR", "SE", "statistic", "P值", "95%CI下限", "95%CI上限", "变量")
write.csv(univariate_results, "R_univariate_results.csv", row.names = FALSE)

sig_01 <- univariate_results[univariate_results$`P值` < 0.1, ]
candidate_vars <- sig_01$变量

if (length(candidate_vars) > 1) {
  X_cand <- train_data[, candidate_vars, drop = FALSE]
  cor_matrix <- cor(X_cand, use = "pairwise.complete.obs")
  to_remove <- c()
  for (i in 1:(ncol(cor_matrix) - 1)) {
    for (j in (i + 1):ncol(cor_matrix)) {
      if (!is.na(cor_matrix[i, j]) && abs(cor_matrix[i, j]) > 0.8) {
        var_i <- colnames(cor_matrix)[i]
        var_j <- colnames(cor_matrix)[j]
        p_i <- sig_01$`P值`[sig_01$变量 == var_i]
        p_j <- sig_01$`P值`[sig_01$变量 == var_j]
        if (length(p_i) > 0 && length(p_j) > 0) {
          if (p_i > p_j) to_remove <- c(to_remove, var_i)
          else to_remove <- c(to_remove, var_j)
        }
      }
    }
  }
  candidate_vars <- candidate_vars[!candidate_vars %in% unique(to_remove)]
}

n_events <- min(sum(train_data[[outcome]] == 1), sum(train_data[[outcome]] == 0))
max_vars <- floor(n_events / 10)
if (length(candidate_vars) > max_vars) candidate_vars <- candidate_vars[1:max_vars]

if (length(candidate_vars) >= 1) {
  formula_str <- sprintf("`%s` ~ %s", outcome, paste0("`", candidate_vars, "`", collapse = " + "))
  tryCatch({
    multi_model <- glm(as.formula(formula_str), data = train_data, family = binomial)
    multi_results <- tidy(multi_model, conf.int = TRUE, exponentiate = TRUE)
    multi_results <- multi_results[multi_results$term != "(Intercept)", ]
    names(multi_results) <- c("变量", "OR", "SE", "statistic", "P值", "95%CI下限", "95%CI上限")
    multi_results <- multi_results[order(multi_results$`P值`), ]
    write.csv(multi_results, "R_multivariate_results.csv", row.names = FALSE)
    sink("R_model_summary.txt")
    print(summary(multi_model))
    sink()
  }, error = function(e) {})
}
