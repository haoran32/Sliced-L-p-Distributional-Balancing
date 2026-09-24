## Summary tables for the DGP ATE simulation -------------------------------
## Produces one table per kappa with rows (N, Estimator) and columns
##   Bias, RMSE, ESE, ESE/SEB, ASE from AN, ASE/SEB,
##   Coverage (Wald / SS / Boot), Length (Wald / SS / Boot),
##   Time (point / Wald / SS / Boot).
##
## Conventions
##   Bias   mean(ATE_Est) - ATE_True
##   RMSE   sqrt(mean((ATE_Est - ATE_True)^2)) across replications
##   ESE    sd(ATE_Est) across replications
##   SEB    oracle standard-error benchmark from the true nuisance functions
##   ASE    mean SE_AsymptoticNormality (Wald rows; SLDB only)
##   Length mean CI_Upper - CI_Lower, averaged within each inference type
##   Point-estimate quantities use one row per (method, kappa, N, SEED);
##   the three inference procedures share the same point estimate.

RESULT <- read.csv("Sim_Results_Raw/sim_result_IPW_CBPS_SKB_GKB_EB_SLDB.csv")


## ---- labels and ordering -------------------------------------------------

METHOD_LABEL <- c(IPW = "IPW", CBPS = "CBPS", Gaussian = "Gaussian",
                  EBW = "EB", SKB = "SKB",
                  SL1DB = "SL1DB", SL2DB = "SL2DB")
METHOD_ORDER <- names(METHOD_LABEL)
N_ORDER      <- c(1000,2000,4000)

RESULT$Estimator <- factor(METHOD_LABEL[RESULT$method],
                           levels = unname(METHOD_LABEL))
RESULT$N         <- factor(RESULT$N, levels = N_ORDER)

## ---- helpers -------------------------------------------------------------

mean_na <- function(x) if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
sd_na   <- function(x) if (sum(!is.na(x)) < 2L) NA_real_ else sd(x, na.rm = TRUE)

## Per-cell statistics for one (kappa, N, Estimator) block.
cell_stats <- function(d) {
  ## one row per seed: the point estimate is shared across inference types
  pt <- d[!duplicated(d$SEED), ]

  by_inf <- function(inf, f, col) {
    x <- d[d$inference == inf, col]
    if (!length(x)) NA_real_ else f(x)
  }

  bias <- mean_na(pt$ATE_Est) - mean_na(pt$ATE_True)
  rmse <- sqrt(mean_na((pt$ATE_Est - pt$ATE_True)^2))
  ese  <- sd_na(pt$ATE_Est)
  seb  <- mean_na(pt$SEB)
  ase  <- by_inf("Wald", mean_na, "SE_AsymptoticNormality")
  seb_wald <- by_inf("Wald", mean_na, "SEB")

  data.frame(
    Bias        = bias,
    RMSE        = rmse,
    `RMSE/SEB`   = rmse / seb,
    `ASE (AN)`  = ase,
    `ASE/SEB`   = ase / seb_wald,
    `Cov Wald`  = by_inf("Wald", mean_na, "Cover"),
    `Cov SS`    = by_inf("SS",   mean_na, "Cover"),
    `Cov Boot`  = by_inf("Boot", mean_na, "Cover"),
    `Len Wald`  = by_inf("Wald", mean_na, "Length"),
    `Len SS`    = by_inf("SS",   mean_na, "Length"),
    `Len Boot`  = by_inf("Boot", mean_na, "Length"),
    `T point`   = mean_na(pt$Time_Point),
    `T Wald`    = by_inf("Wald", mean_na, "Time_Interval"),
    `T SS`      = by_inf("SS",   mean_na, "Time_Interval"),
    `T Boot`    = by_inf("Boot", mean_na, "Time_Interval"),
    `n Wald`    = sum(d$inference == "Wald" & !is.na(d$Cover)),
    `n SS`      = sum(d$inference == "SS" & !is.na(d$Cover)),
    `n Boot`    = sum(d$inference == "Boot" & !is.na(d$Cover)),
    Reps        = nrow(pt),
    check.names = FALSE
  )
}

## ---- assemble ------------------------------------------------------------

grid <- expand.grid(N = factor(N_ORDER, levels = N_ORDER),
                    Estimator = factor(unname(METHOD_LABEL),
                                       levels = unname(METHOD_LABEL)),
                    kappa = sort(unique(RESULT$kappa)),
                    stringsAsFactors = FALSE)
grid <- grid[order(grid$kappa, grid$N, grid$Estimator), c("kappa", "N", "Estimator")]

SUMMARY <- do.call(rbind, lapply(seq_len(nrow(grid)), function(i) {
  g <- grid[i, ]
  d <- RESULT[RESULT$kappa == g$kappa &
              RESULT$N == g$N &
              RESULT$Estimator == g$Estimator, ]
  if (!nrow(d)) {
    s <- cell_stats(RESULT[0, ])
    s$Reps <- 0L
  } else {
    s <- cell_stats(d)
  }
  cbind(g, s, row.names = NULL)
}))

## ---- per-inference detail (mean SE and coverage for every procedure) -----

DETAIL <- do.call(rbind, lapply(split(RESULT, list(RESULT$kappa, RESULT$N,
                                                   RESULT$Estimator,
                                                   RESULT$inference),
                                     drop = TRUE), function(d) {
  data.frame(kappa = d$kappa[1], N = as.integer(as.character(d$N[1])),
             Estimator = as.character(d$Estimator[1]),
             inference = d$inference[1],
             Reps = nrow(d),
             `Valid CIs` = sum(!is.na(d$Cover)),
             `Oracle SEB` = mean_na(d$SEB),
             `Mean Wald SE` = mean_na(d$SE_AsymptoticNormality),
             Coverage   = mean_na(d$Cover),
             `Mean Length` = mean_na(d$Length),
             `Time Point`  = mean_na(d$Time_Point),
             `Time Interval` = mean_na(d$Time_Interval),
             check.names = FALSE, row.names = NULL)
}))
DETAIL <- DETAIL[order(DETAIL$kappa, DETAIL$N,
                       match(DETAIL$Estimator, unname(METHOD_LABEL)),
                       DETAIL$inference), ]

## ---- print ---------------------------------------------------------------

fmt_table <- function(x) {
  out <- data.frame(N = as.character(x$N), Estimator = as.character(x$Estimator),
                    stringsAsFactors = FALSE)
  num <- function(v, d) formatC(round(v, d), format = "f", digits = d,
                                width = 1, flag = "")
  out$Bias        <- num(x$Bias*100, 3)
  out$RMSE        <- num(x$RMSE*100, 3)
  out$`RMSE/SEB`   <- num(x$`RMSE/SEB`, 3)
  out$`ASE (AN)`  <- num(x$`ASE (AN)`*100, 3)
  out$`ASE/SEB`   <- num(x$`ASE/SEB`, 3)
  out$`Cov Wald`  <- num(x$`Cov Wald`, 3)
  out$`Cov SS`    <- num(x$`Cov SS`, 3)
  out$`Cov Boot`  <- num(x$`Cov Boot`, 3)
  out$`Len Wald`  <- num(x$`Len Wald`*100, 3)
  out$`Len SS`    <- num(x$`Len SS`*100, 3)
  out$`Len Boot`  <- num(x$`Len Boot`*100, 3)
  out$`T point`   <- num(x$`T point`, 2)
  out$`T Wald`    <- num(x$`T Wald`, 2)
  out$`T SS`      <- num(x$`T SS`, 2)
  out$`T Boot`    <- num(x$`T Boot`, 2)
  out$`n Wald`    <- x$`n Wald`
  out$`n SS`      <- x$`n SS`
  out$`n Boot`    <- x$`n Boot`
  out$Reps        <- x$Reps
  out[out == "NA"] <- "--"
  out
}

for (k in sort(unique(SUMMARY$kappa))) {
  cat("\n=== Kappa = ", k, " ===\n", sep = "")
  print(fmt_table(SUMMARY[SUMMARY$kappa == k, ]), row.names = FALSE)
}

write.csv(SUMMARY, "Summary_Table.csv", row.names = FALSE)
write.csv(DETAIL,  "Summary_Detail.csv", row.names = FALSE)
