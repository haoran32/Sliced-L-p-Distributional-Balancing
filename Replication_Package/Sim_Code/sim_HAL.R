library(hal9001)
library(foreach)
library(doParallel)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 1) {
  BATCH <- as.numeric(args[1]) + 1
} else {
  BATCH <- 1
}

Hyperparameter <- expand.grid(
  N    = c(1000, 2000, 4000),
  SEED = 1:1100
)

if (BATCH > nrow(Hyperparameter)) {
  stop(sprintf("BATCH %d exceeds total grid size %d", BATCH, nrow(Hyperparameter)))
}

N    <- Hyperparameter[BATCH, "N"]
SEED <- Hyperparameter[BATCH, "SEED"]
set.seed(SEED)

generate_dgp <- function(n) {
  X <- matrix(runif(n * 10), nrow = n, ncol = 10)
  colnames(X) <- paste0("X", 1:10)

  C_X <- (X[, 3] - 0.5)^2 - (X[, 4] - 0.5)^2
  linear_pred_e <- 0.2 - 0.3 * X[, 1] + 0.3 * X[, 2] + 6 * C_X +
    0.05 * rowSums(X[, 5:10] - 0.5)
  prop <- plogis(linear_pred_e)

  A <- rbinom(n, size = 1, prob = prop)

  mu0 <- 0.4 * X[, 1] + 0.4 * X[, 2] + 0.2 * rowSums(X[, 5:10])
  mu1 <- mu0 + 1.5 * C_X
  eps <- rnorm(n, mean = 0, sd = 1)
  Y <- ifelse(A == 1, mu1, mu0) + eps

  aipw_oracle <- A * (Y - mu1) / prop - (1 - A) * (Y - mu0) / (1 - prop) + (mu1 - mu0)
  seb <- sd(aipw_oracle) / sqrt(n)

  return(list(Y = Y, A = A, X = X, ATE_TRUE = 0, SEB = seb))
}

estimate_ate_hal_dcar <- function(dat, K = 5, n_lambda = 20, rate = 0.95,
                                  trunc = 0.00, parallel_folds = TRUE) {
  t0_point <- Sys.time()

  Y <- dat$Y; A <- dat$A; X <- dat$X
  n <- nrow(X)

  data_YX <- cbind(A = A, X)
  mod_Q <- fit_hal(X = data_YX, Y = Y, family = "gaussian",
                   fit_control = list(cv_select = TRUE),
                   max_degree = 2, return_x_basis = FALSE)

  Q1_hat <- predict(mod_Q, new_data = cbind(A = 1, X))
  Q0_hat <- predict(mod_Q, new_data = cbind(A = 0, X))

  folds <- sample(rep(1:K, length.out = n))
  `%run_op%` <- if (parallel_folds) foreach::`%dopar%` else foreach::`%do%`

  fold_preds_list <- foreach(k = 1:K, .packages = "hal9001") %run_op% {
    idx_val <- which(folds == k)
    X_tr <- X[-idx_val, , drop = FALSE]; A_tr <- A[-idx_val]
    X_val <- X[idx_val, , drop = FALSE]

    mod_cv <- fit_hal(X = X_tr, Y = A_tr, family = "binomial",
                      fit_control = list(cv_select = TRUE),
                      max_degree = 2, return_x_basis = FALSE)
    lambda_cv <- mod_cv$lambda_star
    basis_cache <- mod_cv$basis_list

    grid_lambda <- lambda_cv * (rate ^ (1:n_lambda))

    val_preds_k <- matrix(NA, nrow = length(idx_val), ncol = n_lambda)

    for (l in 1:n_lambda) {
      mod_grid <- fit_hal(X = X_tr, Y = A_tr, family = "binomial",
                          lambda = grid_lambda[l],
                          basis_list = basis_cache,
                          fit_control = list(cv_select = FALSE),
                          max_degree = 2, return_x_basis = FALSE)
      val_preds_k[, l] <- predict(mod_grid, new_data = X_val)
    }

    list(idx = idx_val, preds = val_preds_k)
  }

  preds_under_matrix <- matrix(NA, nrow = n, ncol = n_lambda)
  for (res_k in fold_preds_list) {
    preds_under_matrix[res_k$idx, ] <- res_k$preds
  }

  preds_under_matrix <- pmax(pmin(preds_under_matrix, 1 - trunc), trunc)

  dcar_scores <- numeric(n_lambda)
  for (l in 1:n_lambda) {
    e_l <- preds_under_matrix[, l]
    aug_1 <- ((A - e_l) / e_l) * Q1_hat
    aug_0 <- ((A - e_l) / (1 - e_l)) * Q0_hat
    dcar_scores[l] <- abs(mean(aug_1 + aug_0))
  }

  best_l <- which.min(dcar_scores)
  e_opt <- preds_under_matrix[, best_l]

  mu1_ipw <- sum((A * Y) / e_opt) / sum(A / e_opt)
  mu0_ipw <- sum(((1 - A) * Y) / (1 - e_opt)) / sum((1 - A) / (1 - e_opt))
  ate_ipw <- mu1_ipw - mu0_ipw

  t1_point <- Sys.time()
  time_point <- as.numeric(difftime(t1_point, t0_point, units = "secs"))

  t0_ci <- Sys.time()

  D1 <- (A * Y / e_opt) - ((A - e_opt) / e_opt) * Q1_hat - mu1_ipw
  D0 <- ((1 - A) * Y / (1 - e_opt)) + ((A - e_opt) / (1 - e_opt)) * Q0_hat - mu0_ipw
  D_ate <- D1 - D0
  se_eic <- sqrt(mean(D_ate^2) / n)

  lower_ci <- ate_ipw - 1.96 * se_eic
  upper_ci <- ate_ipw + 1.96 * se_eic
  cover    <- as.numeric(lower_ci <= dat$ATE_TRUE && upper_ci >= dat$ATE_TRUE)

  t1_ci <- Sys.time()
  time_ci <- as.numeric(difftime(t1_ci, t0_ci, units = "secs"))


  return(list(
    ate               = ate_ipw,
    se                = se_eic,
    lower             = lower_ci,
    upper             = upper_ci,
    cover             = cover,
    best_step         = best_l,
    dcar_score        = dcar_scores[best_l],
    grid_boundary_hit = (best_l == 1 || best_l == n_lambda),
    time_point        = time_point,
    time_ci           = time_ci,
    seb               = dat$SEB
  ))
}

n_cores <- 5
cl <- makeCluster(n_cores)
registerDoParallel(cl)

dat <- generate_dgp(N)

res <- estimate_ate_hal_dcar(
  dat            = dat,
  K              = 5,
  n_lambda       = 20,
  rate           = 0.95,
  trunc          = 0.00,
  parallel_folds = TRUE
)

stopCluster(cl)

RESULT <- data.frame(
  BATCH        = BATCH,
  N            = N,
  SEED         = SEED,
  ATE_EST      = res$ate,
  SE           = res$se,
  SEB          = res$seb,
  Lower        = res$lower,
  Upper        = res$upper,
  Cover        = res$cover,
  Best_Step    = res$best_step,
  DCAR_Score   = res$dcar_score,
  Boundary_Hit = res$grid_boundary_hit,
  Time_Point   = res$time_point,
  Time_CI      = res$time_ci
)

out_filename <- sprintf("HAL_Result_N%d_S%d.csv", N, SEED)
write.csv(RESULT, file = out_filename, row.names = FALSE)
