args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  process_id <- 0
} else {
  process_id <- as.numeric(args[1])
}

suppressPackageStartupMessages({
  library(foreach)
  library(doParallel)
  library(SuperLearner)
})

source("Core/SuperLearner.R")

Hyperparameter <- expand.grid(
  N    = c(1000, 2000, 4000),
  SEED = 1:1100
)

current_row  <- process_id + 1
if (current_row > nrow(Hyperparameter)) {
  stop("Process ID exceeds the number of hyperparameter rows.")
}

current_N    <- Hyperparameter$N[current_row]
current_SEED <- Hyperparameter$SEED[current_row]

generate_dgp <- function(n, seed) {
  set.seed(seed)
  X <- matrix(runif(n * 10), nrow = n, ncol = 10)
  colnames(X) <- paste0("X", 1:10)

  C_X <- (X[, 3] - 0.5)^2 - (X[, 4] - 0.5)^2
  linear_pred_e <- 0.2 - 0.3 * X[, 1] + 0.3 * X[, 2] + 6 * C_X +
    0.05 * rowSums(X[, 5:10] - 0.5)
  e_true <- plogis(linear_pred_e)

  A <- rbinom(n, size = 1, prob = e_true)

  mu0 <- 0.4 * X[, 1] + 0.4 * X[, 2] + 0.2 * rowSums(X[, 5:10])
  mu1 <- mu0 + 1.5 * C_X
  eps <- rnorm(n, mean = 0, sd = 1)
  Y <- ifelse(A == 1, mu1, mu0) + eps

  psi_oracle <- (A * (Y - mu1) / e_true) -
    ((1 - A) * (Y - mu0) / (1 - e_true)) +
    (mu1 - mu0)
  seb <- sd(psi_oracle) / sqrt(n)

  return(list(
    Data = data.frame(Y = Y, A = A, X),
    SEB  = seb
  ))
}


AIPW_crossfit_parallel <- function(Data, locY = 1, locA = 2, locX = 3:ncol(Data),
                                   K = 5, n_cores = 5,
                                   SL.list = c(1, 4, 5), seed = 1) {
  t0_point <- Sys.time()

  set.seed(seed)
  n <- nrow(Data)
  folds <- sample(rep(1:K, length.out = n))

  Y <- Data[, locY]
  A <- Data[, locA]

  cl <- makeCluster(n_cores)
  registerDoParallel(cl)

  fold_results <- foreach(
    k = 1:K,
    .combine = rbind,
    .packages = c("SuperLearner")
  ) %dopar% {
    train_idx  <- which(folds != k)
    test_idx   <- which(folds == k)

    Data_train <- Data[train_idx, ]
    Data_test  <- Data[test_idx, ]
    newX_test  <- Data_test[, locX, drop = FALSE]

    fit_g <- MySL(Data = Data_train, locY = locA, locX = locX,
                  Ydist = stats::binomial(), SL.list = SL.list)
    g1_k <- as.numeric(predict(fit_g, newdata = newX_test)$pred)

    train_A1 <- Data_train[Data_train[, locA] == 1, ]
    fit_Q1 <- MySL(Data = train_A1, locY = locY, locX = locX,
                   Ydist = stats::gaussian(), SL.list = SL.list)
    Q1_k <- as.numeric(predict(fit_Q1, newdata = newX_test)$pred)

    train_A0 <- Data_train[Data_train[, locA] == 0, ]
    fit_Q0 <- MySL(Data = train_A0, locY = locY, locX = locX,
                   Ydist = stats::gaussian(), SL.list = SL.list)
    Q0_k <- as.numeric(predict(fit_Q0, newdata = newX_test)$pred)

    data.frame(
      idx = test_idx,
      g1  = g1_k,
      Q1  = Q1_k,
      Q0  = Q0_k
    )
  }

  stopCluster(cl)

  fold_results <- fold_results[order(fold_results$idx), ]

  g1_hat <- fold_results$g1
  Q1_hat <- fold_results$Q1
  Q0_hat <- fold_results$Q0

  g1_hat <- pmin(pmax(g1_hat, 0.05), 0.95)

  phi <- (Q1_hat - Q0_hat) +
    A * (Y - Q1_hat) / g1_hat -
    (1 - A) * (Y - Q0_hat) / (1 - g1_hat)

  ate_est <- mean(phi)

  t1_point <- Sys.time()
  time_point <- as.numeric(difftime(t1_point, t0_point, units = "secs"))
  t0_ci <- Sys.time()

  se_est   <- sqrt(as.numeric(var(phi)) / n)
  ci_lower <- ate_est - 1.96 * se_est
  ci_upper <- ate_est + 1.96 * se_est
  cover    <- as.integer(ci_lower <= 0 && 0 <= ci_upper)

  t1_ci <- Sys.time()
  time_ci <- as.numeric(difftime(t1_ci, t0_ci, units = "secs"))

  return(c(
    ATE        = ate_est,
    SE         = se_est,
    Lower      = ci_lower,
    Upper      = ci_upper,
    Cover      = cover,
    Time_Point = time_point,
    Time_CI    = time_ci
  ))
}

sim_env <- generate_dgp(n = current_N, seed = current_SEED)

res <- AIPW_crossfit_parallel(
  Data    = sim_env$Data,
  seed    = current_SEED,
  n_cores = 5,
  K       = 5,
  SL.list = c(1, 2, 4, 5)
)

result_df <- data.frame(
  Process_ID = process_id,
  N          = current_N,
  SEED       = current_SEED,
  ATE        = unname(res["ATE"]),
  SE         = unname(res["SE"]),
  SEB        = sim_env$SEB,
  Lower      = unname(res["Lower"]),
  Upper      = unname(res["Upper"]),
  Cover      = unname(res["Cover"]),
  Time_Point = unname(res["Time_Point"]),
  Time_CI    = unname(res["Time_CI"])
)

out_file <- sprintf("AIPW_Result_N%d_S%d.csv", current_N, current_SEED)
write.csv(result_df, out_file, row.names = FALSE)
