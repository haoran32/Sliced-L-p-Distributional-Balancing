args  <- commandArgs(trailingOnly = TRUE)
BATCH <- if (length(args) >= 1L) as.integer(args[1L]) else 1L

suppressPackageStartupMessages({
  library(Rcpp)
  library(Matrix)
  library(WeightIt)
  library(osqp)
  library(moonboot)
})

source("Core/SLDB.R")
source("Core/competitors.R")
SLDB_MAX_THREADS <- 8L
SLDB_THREADS <- sldb_set_threads(max_threads = SLDB_MAX_THREADS)
cat(sprintf("SLDB OpenMP threads = %d (cap = %d)\n",
            SLDB_THREADS, SLDB_MAX_THREADS))
sldb_load_engine()
skb_load()


## Grid

SLDB_P         <- c(SL1DB = 1L, SL2DB = 2L)
METHODS        <- c("IPW","CBPS","SKB","Gaussian", "EBW", names(SLDB_P))
INFERENCE_GRID <- c("Wald", "SS", "Boot")
KAPPA_GRID     <- c(1)          # looped inside a cell, not across batches
N_GRID         <- c(1000,2000,4000)
SEED_GRID      <- 1:1100

Hyperparameter <- expand.grid(
  method = METHODS,
  inference = INFERENCE_GRID,
  N = N_GRID,
  SEED = SEED_GRID,
  stringsAsFactors = FALSE)

## Wald inference is available only for SLDB.
Hyperparameter <- Hyperparameter[Hyperparameter$inference != "Wald" |
                                   Hyperparameter$method %in% names(SLDB_P), ]
Hyperparameter <- Hyperparameter[
  !(Hyperparameter$inference != "Boot" &
      Hyperparameter$method %in% c("CBPS", "IPW", "SKB")), ]
rownames(Hyperparameter) <- NULL

if (is.na(BATCH) || BATCH < 1L || BATCH > nrow(Hyperparameter)) {
  stop("BATCH must be an integer in 1:", nrow(Hyperparameter))
}

method    <- Hyperparameter[BATCH, "method"]
inference <- Hyperparameter[BATCH, "inference"]
N         <- Hyperparameter[BATCH, "N"]
SEED      <- Hyperparameter[BATCH, "SEED"]

R_RESAMPLE <- 500L   # bootstrap draws, and subsampling draws at the chosen m
SS_M_R   <- 200L   # draws at each candidate size while selecting m
SS_TAU_R <- 200L   # draws at each ladder rung while fitting tau
CFD_LAMBDA <- 1     # WeightIt cfd ridge; ridge = cfd_lambda/4 of the QP diagonal
LEVEL      <- 0.95

D_COVARIATE <- 10L
CONTROL_P   <- if (method %in% names(SLDB_P)) unname(SLDB_P[[method]]) else 2L

## Everything except d, n, and p is left at the sldb_control() defaults:
## L_coefficient = 1, L_exponent = 1 + 1/(4 D_COVARIATE),
## proportion_to_initial_loss = N^(-L_exponent), delta_fraction = 0.2 for
## p = 1 and NULL for p = 2, eta_step = 1, alpha = 1, max_iter = 10000,
## factr = 1e7, factr_power = 1, and a uniform start.
CONTROL <- sldb_control(d = D_COVARIATE, n = N, p = CONTROL_P)

## Gaussian and EBW build dense n by n programs. The guard admits every N used
## here; it is set to the same value in Data.R, where it binds.
DENSE_MAX_N <- 10000


##Data-generating process (DGP) for the simulation study.

expit <- function(v) 1 / (1 + exp(-v))

dgp <- function(N) {
  X <- matrix(runif(N * D_COVARIATE, 0, 1), nrow = N)
  colnames(X) <- paste0("V", seq_len(D_COVARIATE))

  ## This centered curvature is omitted from the linear IPW and CBPS models.
  curvature <- ((X[, 3] - 0.5)^2 - (X[, 4] - 0.5)^2)

  ps_link <- 0.2 -
    0.3 * X[, 1] + 0.3 * X[, 2] +
    6 * curvature +
    0.05 * rowSums(X[, 5:10, drop = FALSE] - 0.5)

  prop <- expit(ps_link)
  Z <- as.integer(rbinom(N, 1, prop))

  base.coef <- c(0.4,0.4,0,0,rep(0.2,6))
  base <- as.vector(X %*% base.coef)

  tau <- 1.5 * curvature


  outcomes <- lapply(KAPPA_GRID, function(kappa) {
    mu1 <- base + kappa*tau
    mu0 <- base

    Y1  <- mu1 + rnorm(N)
    Y0  <- mu0 + rnorm(N)
    Y   <- Z * Y1 + (1 - Z) * Y0

    aipw <- Z * (Y - mu1) / prop - (1 - Z) * (Y - mu0) / (1 - prop) + (mu1 - mu0)

    list(kappa = kappa, X = X, Z = Z, Y = Y, ATE_TRUE = 0,
         SEB = sd(aipw) / sqrt(N))
  })

  return(list(X = X, Z = Z, outcomes = outcomes))
}


covered <- function(ci, truth) {
  if (any(!is.finite(ci))) return(NA_real_)
  as.numeric(ci[1L] < truth && ci[2L] > truth)
}

## `elapsed()`, `sldb_effect()` and `sldb_summary()` come from SLDB.R.


## Shared-weight estimator closure for the competing methods
## It mirrors the single-outcome closure in competitors.R, except that the
## refit is contracted against every outcome in Y_list, returning one estimate
## per kappa instead of a scalar. SLDB needs no counterpart here:
## sldb_effect() builds its own multi-outcome closure internally.

competitor_estimator_multi <- function(X, Z, Y_list, method, opts, cache) {
  X <- as.matrix(X)
  if (is.null(colnames(X))) colnames(X) <- paste0("V", seq_len(ncol(X)))
  Z <- as.integer(Z)
  K <- length(Y_list)
  function(idx) {
    Zi <- Z[idx]
    if (sum(Zi == 1L) == 0L || sum(Zi == 0L) == 0L) return(rep(NA_real_, K))
    w <- competitor_weights(X[idx, , drop = FALSE], Zi, method, opts)
    if (is.null(w)) return(rep(NA_real_, K))
    vapply(Y_list, function(Yv) estimate_ate(w, Zi, Yv[idx]), numeric(1))
  }
}


## One cell: one (method, inference, N, SEED), all kappa

run_cell <- function(method, inference, N, SEED) {

  ## The seed keeps the dataset fixed across methods and inference procedures.
  set.seed(SEED)
  d <- dgp(N)
  X <- d$X; Z <- d$Z
  Y_list <- lapply(d$outcomes, `[[`, "Y")
  K      <- length(Y_list)

  ## WeightIt CFD ridge for Gaussian and EBW. It is passed to WeightIt
  ## unchanged and puts the ridge at cfd_lambda/4 of the QP diagonal, the same
  ## fraction at every sample size, so the point estimate and every bootstrap
  ## or subsampling refit solve the same regularized problem.
  opts  <- competitor_opts(cfd_lambda = CFD_LAMBDA,
                           dense_max_n = DENSE_MAX_N)
  cache <- competitor_cache(X, Z, method, opts)

  if (method %in% names(SLDB_P)) {

    ## SLDB: the weights, the ATE, and the requested interval in three lines.
    ## `S` carries the estimates, the interval, the Wald standard error, the
    ## subsampling diagnostics, every weight-fit diagnostic, and both timings,
    ## one row per outcome.
    pt  <- elapsed(sldb_weights(X, Z, CONTROL))
    eff <- sldb_effect(pt, Y_list, estimand = "ATE", CI = inference,
                       level = LEVEL, R = R_RESAMPLE, CI.Time = TRUE)
    S   <- sldb_summary(eff)

  } else {

    ## Competing methods: one weight fit, contracted against each kappa
    ## outcome, then the same interval routines SLDB uses. They report no
    ## weight-fit diagnostics, so those columns stay at the template's NA.
    if (identical(inference, "Wald")) {
      stop("Wald inference is implemented for SLDB only.")
    }
    pt <- elapsed(competitor_weights(X, Z, method, opts))
    w  <- pt$value
    if (is.null(w)) stop("Point estimate failed for method ", method)
    est <- vapply(Y_list, function(Yv) estimate_ate(w, Z, Yv), numeric(1))
    estimator <- competitor_estimator_multi(X, Z, Y_list, method = method,
                                            opts = opts, cache = cache)

    r <- if (inference == "SS") {
      ## Subsampling inference across outcome specifications.
      elapsed(lapply(seq_len(K), function(k) {
        sldb_moonboot_ci(function(idx) estimator(idx)[k],
                         n = N,
                         R = R_RESAMPLE,
                         level = LEVEL)
      }))
    } else {
      elapsed(sldb_boot_ci_multi(estimator, n = N, K = K,
                                 R = R_RESAMPLE, level = LEVEL))
    }
    intervals <- r$value

    S <- sldb_summary(NULL, K = K, inference = inference)
    S$Estimate     <- est
    S$CI_Lower     <- vapply(intervals, function(z) z$ci[1L], numeric(1))
    S$CI_Upper     <- vapply(intervals, function(z) z$ci[2L], numeric(1))
    S$Length       <- S$CI_Upper - S$CI_Lower
    S$m            <- do.call(c, lapply(intervals, `[[`, "m"))
    S$Tau_Exponent <- vapply(intervals, function(z) {
      if (is.null(z$tau_exponent)) NA_real_ else z$tau_exponent
    }, numeric(1))
    S$Time_Point    <- pt$time
    S$Time_Interval <- r$time
  }

  do.call(rbind, lapply(seq_len(K), function(k) {
    o  <- d$outcomes[[k]]
    ci <- c(S$CI_Lower[k], S$CI_Upper[k])
    data.frame(
      method                 = method,
      kappa                  = o$kappa,
      N                      = N,
      SEED                   = SEED,
      inference              = inference,
      ATE_Est                = S$Estimate[k],
      ATE_True               = o$ATE_TRUE,
      Bias                   = S$Estimate[k] - o$ATE_TRUE,
      CI_Lower               = ci[1L],
      CI_Upper               = ci[2L],
      Cover                  = covered(ci, o$ATE_TRUE),
      Length                 = S$Length[k],
      SE_AsymptoticNormality = S$SE_AsymptoticNormality[k],
      SEB                    = o$SEB,
      m                      = S$m[k],
      Tau_Exponent           = S$Tau_Exponent[k],
      SS_m_R                 = S$SS_m_R[k],
      lambda                 = S$lambda[k],
      L                      = S$L[k],
      L_Coefficient          = S$L_Coefficient[k],
      L_Exponent             = S$L_Exponent[k],
      p                      = S$p[k],
      Optimizer              = S$Optimizer[k],
      Optim_Convergence      = S$Optim_Convergence[k],
      Gradient_Norm          = S$Gradient_Norm[k],
      Converged               = S$Converged[k],
      Delta_Criterion_Met     = S$Delta_Criterion_Met[k],
      Iterations              = S$Iterations[k],
      Final_Gap               = S$Final_Gap[k],
      Gap_over_Delta          = S$Gap_over_Delta[k],
      Eta_Initial            = S$Eta_Initial[k],
      Eta_Final              = S$Eta_Final[k],
      Eta_Increases          = S$Eta_Increases[k],
      Eta_Decreases          = S$Eta_Decreases[k],
      Initial_Loss           = S$Initial_Loss[k],
      q_constant             = S$q_constant[k],
      Proportion_Initial_Loss = S$Proportion_Initial_Loss[k],
      Uniform_Objective      = S$Uniform_Objective[k],
      Logistic_Objective     = S$Logistic_Objective[k],
      Reference_Objective    = S$Reference_Objective[k],
      Initial_Source         = S$Initial_Source[k],
      delta                  = S$delta[k],
      Time_Point             = S$Time_Point[k],
      Time_Interval          = S$Time_Interval[k],
      row.names = NULL, stringsAsFactors = FALSE
    )
  }))
}

RESULT <- run_cell(method, inference, N, SEED)
print(RESULT)

outfile <- sprintf("ATE_%s_N%d_%s_S%d.csv",
                   method, N, inference, SEED)
write.csv(RESULT, file = outfile, row.names = FALSE)
