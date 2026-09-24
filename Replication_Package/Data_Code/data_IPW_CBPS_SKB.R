suppressPackageStartupMessages({
  library(Rcpp)
  library(Matrix)
  library(WeightIt)
  library(osqp)
  library(DoubleML)
  library(data.table)
  library(moonboot)
})

source("Core/SLDB.R")
source("Core/competitors.R")

SLDB_MAX_THREADS <- 8L
sldb_set_threads(max_threads = SLDB_MAX_THREADS)
sldb_load_engine()
skb_load()

METHODS <- c("IPW", "CBPS", "SKB")

INFERENCE_BY_METHOD <- list(
  IPW  = "Boot",
  CBPS = "Boot",
  SKB  = "Boot"
)

n <- 9915
D_COVARIATE <- 9
SEED        <- 1111L
R_RESAMPLE  <- 500L
CFD_LAMBDA  <- 1
LEVEL       <- 0.95
DENSE_MAX_N <- 10000

prepare_401k <- function() {
  d <- DoubleML::fetch_401k(return_type = "data.table", instrument = TRUE)
  features <- c("age", "inc", "educ", "fsize", "marr", "twoearn",
                "db", "pira", "hown")
  X <- scale(as.matrix(d[, features, with = FALSE]))
  list(Y = as.numeric(d$net_tfa),
       A = as.numeric(d$p401),
       Z = as.integer(d$e401),
       X = X,
       n = nrow(d))
}

D <- prepare_401k()
X <- D$X; Z <- D$Z; Y <- D$Y; A <- D$A; n <- D$n

all_results <- list()

for (method in METHODS) {

  set.seed(SEED)
  inferences <- INFERENCE_BY_METHOD[[method]]

  opts  <- competitor_opts(cfd_lambda = CFD_LAMBDA, dense_max_n = DENSE_MAX_N)
  cache <- competitor_cache(X, Z, method, opts)

  pt <- elapsed(competitor_weights(X, Z, method, opts))
  w  <- pt$value
  if (is.null(w)) stop("Point estimate failed for method ", method)

  est         <- estimate_late(w, Z, Y, A)
  first_stage <- weighted_contrast(w, Z, A)

  S_list <- lapply(inferences, function(inference) {
    estimator <- competitor_estimator(X, Z, Y, A = A, method = method,
                                      opts = opts, cache = cache)

    r <- elapsed(sldb_boot_ci(estimator, n = n, R = R_RESAMPLE, level = LEVEL))

    ci_lower <- r$value$ci[1L]
    ci_upper <- r$value$ci[2L]
    ci_len   <- ci_upper - ci_lower

    data.frame(
      Method        = method,
      Inference     = inference,
      Estimate      = round(est, 2),
      First_Stage   = round(first_stage, 4),
      CI_Lower      = round(ci_lower, 2),
      CI_Upper      = round(ci_upper, 2),
      Length        = round(ci_len, 2),
      Time_Point    = round(pt$time, 2),
      Time_Interval = round(r$time, 2),
      stringsAsFactors = FALSE
    )
  })

  all_results[[method]] <- do.call(rbind, S_list)
}

final_df <- do.call(rbind, all_results)
print(final_df)
