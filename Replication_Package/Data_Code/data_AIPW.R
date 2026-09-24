suppressPackageStartupMessages({
  library(data.table)
  library(DoubleML)
  library(SuperLearner)
  library(foreach)
  library(doParallel)
  library(doRNG)
})

source("Core/SuperLearner.R")

method    <- "AIPW"
inference <- "Wald"
SEED      <- 1111L
LEVEL     <- 0.95
K_FOLDS   <- 5L
N_CORES   <- 5L
SL_LIST   <- c(1, 4, 5, 7)
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

folds <- sample(rep(1:K_FOLDS, length.out = n))
Data_all <- data.frame(Y = Y, A = A, Z = Z, X)
locX <- 4:ncol(Data_all)

cl <- makeCluster(N_CORES)
registerDoParallel(cl)

fold_results <- foreach(
  k = 1:K_FOLDS,
  .combine = rbind,
  .packages = c("SuperLearner")
) %dorng% {

  train_idx  <- which(folds != k)
  test_idx   <- which(folds == k)

  Data_train <- Data_all[train_idx, ]
  Data_test  <- Data_all[test_idx, ]
  newX_test  <- Data_test[, locX, drop = FALSE]

  fit_e <- MySL(Data = Data_train, locY = 3, locX = locX,
                Ydist = stats::binomial(), SL.list = SL_LIST)
  e_k <- as.numeric(predict(fit_e, newdata = newX_test)$pred)

  train_Z1 <- Data_train[Data_train$Z == 1, ]
  fit_QY1 <- MySL(Data = train_Z1, locY = 1, locX = locX,
                  Ydist = stats::gaussian(), SL.list = SL_LIST)
  QY1_k <- as.numeric(predict(fit_QY1, newdata = newX_test)$pred)

  train_Z0 <- Data_train[Data_train$Z == 0, ]
  fit_QY0 <- MySL(Data = train_Z0, locY = 1, locX = locX,
                  Ydist = stats::gaussian(), SL.list = SL_LIST)
  QY0_k <- as.numeric(predict(fit_QY0, newdata = newX_test)$pred)

  fit_QA1 <- MySL(Data = train_Z1, locY = 2, locX = locX,
                  Ydist = stats::binomial(), SL.list = SL_LIST)
  QA1_k <- as.numeric(predict(fit_QA1, newdata = newX_test)$pred)

  QA0_k <- rep(0, length(test_idx))

  data.frame(
    idx = test_idx,
    e   = e_k,
    QY1 = QY1_k,
    QY0 = QY0_k,
    QA1 = QA1_k,
    QA0 = QA0_k
  )
}

stopCluster(cl)

fold_results <- fold_results[order(fold_results$idx), ]
e_hat   <- pmin(pmax(fold_results$e, TRUNC), 1 - TRUNC)
QY1_hat <- fold_results$QY1
QY0_hat <- fold_results$QY0
QA1_hat <- fold_results$QA1
QA0_hat <- 0

phi_Y <- (QY1_hat - QY0_hat) +
  Z * (Y - QY1_hat) / e_hat -
  (1 - Z) * (Y - QY0_hat) / (1 - e_hat)
delta_Y <- mean(phi_Y)

phi_A <- (QA1_hat - QA0_hat) +
  Z * (A - QA1_hat) / e_hat -
  (1 - Z) * (A - QA0_hat) / (1 - e_hat)
first_stage <- mean(phi_A)

late_est <- delta_Y / first_stage

t1_point   <- Sys.time()
time_point <- as.numeric(difftime(t1_point, t0_point, units = "secs"))

t0_inf <- Sys.time()

D_Y     <- phi_Y - delta_Y
D_A     <- phi_A - first_stage
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
