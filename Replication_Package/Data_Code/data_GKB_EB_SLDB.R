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

prepare_401k <- function() {
  d <- DoubleML::fetch_401k(return_type = "data.table", instrument = TRUE)
  features <- c("age", "inc", "educ", "fsize", "marr", "twoearn",
                "db", "pira", "hown")
  X <- scale(as.matrix(d[, features, with = FALSE]))
  list(Y = as.numeric(d$net_tfa),
       A = as.numeric(d$p401),
       Z = as.integer(d$e401),
       X = X,
       n = nrow(d),
       d = ncol(X))
}

D <- prepare_401k()
X <- D$X; Z <- D$Z; Y <- D$Y; A <- D$A; n <- D$n; d_cov <- D$d


SEED        <- 1111L
R_RESAMPLE  <- 500L
LEVEL       <- 0.95
CFD_LAMBDA  <- 1.0
DENSE_MAX_N <- 10000

METHODS <- c("SL1DB", "SL2DB", "EBW", "Gaussian")
INFERENCES_LIST <- list(
  SL1DB    = c("Wald", "SS"),
  SL2DB    = c("Wald", "SS"),
  EBW      = "SS",
  Gaussian = "SS"
)

all_results <- list()

for (method in METHODS) {

  set.seed(SEED)
  inferences <- INFERENCES_LIST[[method]]

  if (method %in% c("SL1DB", "SL2DB")) {

    # --- SLDB Pipeline ---
    p_order <- if (method == "SL1DB") 1L else 2L
    ctrl <- sldb_control(d = d_cov, n = n, p = p_order)

    cat("  -> Fitting weights... ")
    pt <- elapsed(sldb_weights(X, Z, ctrl))
    cat(sprintf("Done in %.2f seconds.\n", pt$time))

    cat(sprintf("  -> Running Inferences: %s... \n", paste(inferences, collapse = ", ")))
    eff <- sldb_effect(pt, Y, A = A, estimand = "LATE", CI = inferences,
                       level = LEVEL, R = R_RESAMPLE, CI.Time = TRUE)

    S <- sldb_summary(eff)
    S$Method <- method

    all_results[[method]] <- S

    for (i in seq_len(nrow(S))) {
      cat(sprintf("     [%s] LATE = %.1f, 95%% CI = [%.1f, %.1f], Length = %.1f, Time_Inf = %.2fs\n",
                  S$Inference[i], S$Estimate[i], S$CI_Lower[i], S$CI_Upper[i], S$Length[i], S$Time_Interval[i]))
    }

  } else {

    # --- EBW / Gaussian Pipeline ---
    opts <- competitor_opts(cfd_lambda = CFD_LAMBDA, dense_max_n = DENSE_MAX_N)
    cache <- competitor_cache(X, Z, method, opts)

    cat("  -> Fitting dense weights... ")
    pt <- elapsed(competitor_weights(X, Z, method, opts))
    w <- pt$value
    cat(sprintf("Done in %.2f seconds.\n", pt$time))
    if (is.null(w)) stop("Weight fit failed for ", method)

    est_late    <- estimate_late(w, Z, Y, A)
    first_stage <- weighted_contrast(w, Z, A)

    S_list <- list()
    for (inf in inferences) {
      cat(sprintf("  -> Running %s inference (this may take a while)... ", inf))
      estimator <- competitor_estimator(X, Z, Y, A = A, method = method,
                                        opts = opts, cache = cache)

      r <- elapsed(sldb_moonboot_ci(estimator, n = n, R = R_RESAMPLE, level = LEVEL))
      cat(sprintf("Done in %.2f seconds.\n", r$time))

      row <- sldb_summary(NULL, K = 1L, inference = inf)
      row$Method        <- method
      row$Estimand      <- "LATE"
      row$Estimate      <- est_late
      row$First_Stage   <- first_stage
      row$CI_Lower      <- r$value$ci[1L]
      row$CI_Upper      <- r$value$ci[2L]
      row$Length        <- row$CI_Upper - row$CI_Lower
      row$m             <- if (inf == "SS") r$value$m else n
      row$Tau_Exponent  <- if (inf == "SS") r$value$tau_exponent else NA_real_
      row$Time_Point    <- pt$time
      row$Time_Interval <- r$time


      S_list[[inf]] <- row

      cat(sprintf("     [%s] LATE = %.1f, 95%% CI = [%.1f, %.1f], Length = %.1f\n",
                  inf, row$Estimate, row$CI_Lower, row$CI_Upper, row$Length))
    }
    all_results[[method]] <- do.call(rbind, S_list)
  }
}

final_df <- as.data.frame(data.table::rbindlist(all_results, fill = TRUE))
write.csv(final_df, "Diagnostics_401k_Full.csv", row.names = FALSE)
target_cols <- c("Method", "Inference", "Estimate", "First_Stage",
                 "CI_Lower", "CI_Upper", "Length",
                 "Time_Point", "Time_Interval")

actual_cols <- intersect(target_cols, names(final_df))
clean_table <- final_df[, actual_cols, drop = FALSE]
clean_table$Estimate      <- round(clean_table$Estimate, 2)
clean_table$First_Stage   <- round(clean_table$First_Stage, 4)
clean_table$CI_Lower      <- round(clean_table$CI_Lower, 2)
clean_table$CI_Upper      <- round(clean_table$CI_Upper, 2)
clean_table$Length        <- round(clean_table$Length, 2)
clean_table$Time_Point    <- round(clean_table$Time_Point, 2)
clean_table$Time_Interval <- round(clean_table$Time_Interval, 2)
write.csv(clean_table, "Table_401k_Results.csv", row.names = FALSE)
print(clean_table, row.names = FALSE)
cat(">>> Main results saved to:       'Table_401k_Results.csv'\n")
cat(">>> Detailed diagnostics saved to: 'Diagnostics_401k_Full.csv'\n\n")
