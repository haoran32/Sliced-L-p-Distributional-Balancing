library(dplyr)

summarize_benchmark <- function(file_path, est_col, method_name,
                                true_ate = 0, n_keep = 1000) {
  if (!file.exists(file_path)) {
    warning("File not found: ", file_path)
    return(NULL)
  }

  dat <- read.csv(file_path) %>%
    filter(N %in% c(1000, 2000, 4000)) %>%
    rename(est = all_of(est_col)) %>%
    filter(!is.na(est), !is.na(Cover))
  if ("SEED" %in% names(dat)) {
    dat <- dat %>% arrange(N, SEED)
  } else {
    warning("No SEED column in ", file_path, "; using row order.")
  }

  dat %>%
    group_by(N) %>%
    slice_head(n = n_keep) %>%
    summarise(
      Method        = method_name,
      Reps          = n(),
      bias_raw      = mean(est - true_ate),
      sd_raw        = sd(est),
      se_raw        = mean(SE),
      rmse_raw      = sqrt(mean((est - true_ate)^2)),
      len_raw       = mean(Upper - Lower),
      cov_raw       = mean(Cover),
      time_point    = mean(Time_Point),
      time_CI       = mean(Time_CI),
      seb_raw       = mean(SEB),
      .groups       = "drop"
    ) %>%
    transmute(
      N, Method, Reps,
      Bias          = round(bias_raw * 100, 3),
      Empirical_SD  = round(sd_raw * 100, 3),
      Mean_SE       = round(se_raw * 100, 3),
      RMSE          = round(rmse_raw * 100, 3),
      Avg_CI_Length = round(len_raw * 100, 3),
      Coverage_Pct  = round(cov_raw * 100, 1),
      time_point, time_CI,
      SEB           = round(seb_raw * 100, 4),
      Efficiency    = round(rmse_raw / seb_raw, 3)
    )
}

hal_summary  <- summarize_benchmark("Sim_Results_Raw/sim_result_HAL.csv",
                                    est_col = "ATE_EST", method_name = "HAL-IPW")
aipw_summary <- summarize_benchmark("Sim_Results_Raw/sim_result_AIPW.csv",
                                    est_col = "ATE",     method_name = "AIPW")

benchmarks_table <- bind_rows(hal_summary, aipw_summary) %>%
  arrange(N, Method)

print(as.data.frame(benchmarks_table), digits = 5)
