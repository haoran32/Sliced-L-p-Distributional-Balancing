suppressPackageStartupMessages({
  library(data.table)
  library(DoubleML)
  library(hal9001)
  library(foreach)
  library(doParallel)
  library(doRNG)
})

method    <- "HAL"
inference <- "Wald"
SEED      <- 1111L
LEVEL     <- 0.95
K_FOLDS   <- 5L
N_CORES   <- 5L
N_LAMBDA  <- 20L
RATE      <- 0.95
TRUNC     <- 0.0

prepare_401k <- function() {
  d <- DoubleML::fetch_401k(return_type = "data.table", instrument = TRUE)
  features <- c("age", "inc", "educ", "fsize", "marr", "twoearn",
                "db", "pira", "hown")
  X <- scale(as.matrix(d[, features, with = FALSE]))
  list(
    Y = as.numeric(d$net_tfa),
    A = as.numeric(d$p401),
    Z = as.integer(d$e401),
    X = X,
    n = nrow(d)
  )
}

D <- prepare_401k()
X <- D$X; Z <- D$Z; Y <- D$Y; A <- D$A; n <- D$n

set.seed(SEED)

t0_point <- Sys.time()

mod_QY1 <- fit_hal(X = X[Z == 1, , drop = FALSE], Y = Y[Z == 1], family = "gaussian",
                   fit_control = list(cv_select = TRUE),
                   max_degree = 2, return_x_basis = FALSE)
mod_QY0 <- fit_hal(X = X[Z == 0, , drop = FALSE], Y = Y[Z == 0], family = "gaussian",
                   fit_control = list(cv_select = TRUE),
                   max_degree = 2, return_x_basis = FALSE)
Q1_Y <- predict(mod_QY1, new_data = X)
Q0_Y <- predict(mod_QY0, new_data = X)

mod_Q_A1 <- fit_hal(X = X[Z == 1, , drop = FALSE], Y = A[Z == 1], family = "binomial",
                    fit_control = list(cv_select = TRUE),
                    max_degree = 2, return_x_basis = FALSE)
Q1_A <- as.numeric(predict(mod_Q_A1, new_data = X))
Q0_A <- rep(0, n)

folds <- sample(rep(1:K_FOLDS, length.out = n))

cl <- makeCluster(N_CORES)
registerDoParallel(cl)

fold_preds_list <- foreach(k = 1:K_FOLDS, .packages = "hal9001") %dorng% {
  idx_val <- which(folds == k)
  X_tr  <- X[-idx_val, , drop = FALSE]
  Z_tr  <- Z[-idx_val]
  X_val <- X[idx_val, , drop = FALSE]

  mod_cv      <- fit_hal(X = X_tr, Y = Z_tr, family = "binomial",
                         fit_control = list(cv_select = TRUE),
                         max_degree = 2, return_x_basis = FALSE)
  lambda_cv   <- mod_cv$lambda_star
  basis_cache <- mod_cv$basis_list

  grid_lambda <- lambda_cv * (RATE ^ (1:N_LAMBDA))
  val_preds_k <- matrix(NA_real_, nrow = length(idx_val), ncol = N_LAMBDA)

  for (l in 1:N_LAMBDA) {
    mod_grid <- fit_hal(X = X_tr, Y = Z_tr, family = "binomial",
                        lambda = grid_lambda[l],
                        basis_list = basis_cache,
                        fit_control = list(cv_select = FALSE),
                        max_degree = 2, return_x_basis = FALSE)
    val_preds_k[, l] <- predict(mod_grid, new_data = X_val)
  }

  list(idx = idx_val, preds = val_preds_k, lambda_path = grid_lambda)
}
stopCluster(cl)

preds_under_matrix <- matrix(NA_real_, nrow = n, ncol = N_LAMBDA)
for (res_k in fold_preds_list) {
  preds_under_matrix[res_k$idx, ] <- res_k$preds
}
preds_under_matrix <- pmax(pmin(preds_under_matrix, 1 - TRUNC), TRUNC)

dcar_scores <- numeric(N_LAMBDA)
for (l in 1:N_LAMBDA) {
  e_l <- preds_under_matrix[, l]

  w1_l <- Z / e_l
  w0_l <- (1 - Z) / (1 - e_l)

  delta_Y_l <- sum(w1_l * Y) / sum(w1_l) - sum(w0_l * Y) / sum(w0_l)
  delta_A_l <- sum(w1_l * A) / sum(w1_l) - sum(w0_l * A) / sum(w0_l)
  psi_hat_l <- delta_Y_l / delta_A_l

  aug_Y1 <- ((Z - e_l) / e_l) * Q1_Y
  aug_Y0 <- ((Z - e_l) / (1 - e_l)) * Q0_Y
  D_CAR_Y <- aug_Y1 + aug_Y0

  aug_A1 <- ((Z - e_l) / e_l) * Q1_A
  aug_A0 <- ((Z - e_l) / (1 - e_l)) * Q0_A
  D_CAR_A <- aug_A1 + aug_A0

  dcar_scores[l] <- abs(mean(D_CAR_Y) - psi_hat_l * mean(D_CAR_A)) / abs(delta_A_l)
}

best_l <- which.min(dcar_scores)
e_opt  <- preds_under_matrix[, best_l]

w1 <- Z / e_opt
w0 <- (1 - Z) / (1 - e_opt)

delta_Y     <- sum(w1 * Y) / sum(w1) - sum(w0 * Y) / sum(w0)
first_stage <- sum(w1 * A) / sum(w1) - sum(w0 * A) / sum(w0)
late_est    <- delta_Y / first_stage

t1_point   <- Sys.time()
time_point <- as.numeric(difftime(t1_point, t0_point, units = "secs"))

t0_inf <- Sys.time()

D_Y <- (w1 * (Y - Q1_Y) + Q1_Y) - (w0 * (Y - Q0_Y) + Q0_Y) - delta_Y
D_A <- (w1 * (A - Q1_A) + Q1_A) - (w0 * (A - Q0_A) + Q0_A) - first_stage

D_late  <- (D_Y - late_est * D_A) / first_stage
se_late <- sqrt(mean(D_late^2) / n)

crit     <- qnorm(1 - (1 - LEVEL) / 2)
ci_lower <- late_est - crit * se_late
ci_upper <- late_est + crit * se_late
ci_len   <- ci_upper - ci_lower

t1_inf        <- Sys.time()
time_interval <- as.numeric(difftime(t1_inf, t0_inf, units = "secs"))

RESULT <- data.frame(
  Method        = method,
  Inference     = inference,
  Estimate      = round(late_est, 2),
  First_Stage   = round(first_stage, 4),
  CI_Lower      = round(ci_lower, 2),
  CI_Upper      = round(ci_upper, 2),
  Length        = round(ci_len, 2),
  Time_Point    = round(time_point, 2),
  Time_Interval = round(time_interval, 2),
  stringsAsFactors = FALSE
)

print(RESULT)
