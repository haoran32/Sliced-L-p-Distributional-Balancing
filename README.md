# Replication Package for "Sliced $L^p$ Distributional Balancing"

This repository contains the replication code and raw simulation outputs for the manuscript.

> **CRITICAL NOTE ON REPRODUCTION:**
> All scripts should be executed with the R working directory set to this package root directory. 

## 1. System and Software Requirements

### 1.1 R Environment & Package Versions
- **R Version**: >= 4.4.1 
- **C++ Compiler**: A C++11 (or newer) compiler with **OpenMP** multi-threading support enabled.
- **R Packages**: All packages are available on CRAN. Below are the tested package versions:
  - *Rcpp* (1.1.2)
  - *RcppParallel* (5.1.11.2)
  - *Matrix* (1.7.5)
  - *WeightIt* (2.0.0)
  - *osqp* (1.0.0)
  - *DoubleML* (1.0.2)
  - *data.table* (1.17.8)
  - *dplyr* (1.2.0)
  - *foreach* (1.5.2)
  - *doParallel* (1.0.17)
  - *doRNG* (1.8.6.3)
  - *hal9001* (0.4.6)
  - *SuperLearner* (2.0.40)
  - *earth* (5.3.6)
  - *xgboost* (3.2.1.1)
  - *caret* (7.0.1)
  - *moonboot* (2.0.1)

To install missing dependencies:
```R
cran_packages <- c(
  "Rcpp", "RcppParallel", "Matrix", "WeightIt", "osqp", "DoubleML", 
  "data.table", "dplyr", "foreach", "doParallel", "doRNG", 
  "hal9001", "SuperLearner", "earth", "xgboost", "caret", "moonboot"
)
install.packages(setdiff(cran_packages, installed.packages()[, "Package"]))
```

## 2. Directory Structure

```R
Replication_Package/
├── README.md
│
├── Core/                                  # Core methodology and dependency engines
│   ├── SLDB.R                             # Proposed SLDB estimator and inference routines
│   ├── swweight.cpp                       # C++ engine for Sliced L_p distance balancing
│   ├── competitors.R                      # Benchmark weighting methods (IPW, CBPS, SKB, GKB, EB)
│   ├── RBF_kernel_C_parallel.cpp          # Parallel RBF kernel computation for SKB
│   ├── utils.R                            # Helper utility functions for SKB
│   └── SuperLearner.R                     # SuperLearner wrapper and base learners for AIPW
│
├── Data_Code/                             # Real data analysis (Section 6)
│   ├── data_IPW_CBPS_SKB.R                # Empirical estimation: IPW, CBPS, SKB
│   ├── data_GKB_EB_SLDB.R                 # Empirical estimation: GKB, EB, SLDB
│   ├── data_HAL.R                         # Empirical estimation: HAL-IPW
│   └── data_AIPW.R                        # Empirical estimation: AIPW
│
├── Sim_Code/                              # Simulation (Section 5)
│   ├── sim_IPW_CBPS_SKB_GKB_EB_SLDB.R     # Simulation driver for IPW, CBPS, SKB, GKB, EB, SLDB
│   ├── sim_HAL.R                          # Simulation driver for HAL-IPW
│   └── sim_AIPW.R                         # Simulation driver for AIPW
│
├── Sim_Results_Raw/                       # Aggregated raw CSV outputs across all replications
│   ├── sim_result_IPW_CBPS_SKB_GKB_EB_SLDB.csv
│   ├── sim_result_HAL.csv
│   └── sim_result_AIPW.csv
│
└── Sim_Results_Summary/                   # Summary scripts to reproduce manuscript exhibits
    ├── summary_IPW_CBPS_SKB_GKB_EB_SLDB.R # Summarizes Tables 1 and 2 for IPW, CBPS, SKB, GKB, EB, SLDB
    └── summary_HAL_AIPW.R                 # Summarizes Tables 1 and 2 for HAL-IPW and AIPW
```

## 3. Instructions for Reproduction

1. **Reproduce Simulation Exhibits (Tables 1 and 2)**
   - Simulation scripts in `Sim_Code/` are computationally intensive; their raw outputs are provided in `Sim_Results_Raw/`.
   - To reproduce Tables 1–2, run the scripts in `Sim_Results_Summary/`:
     ```R
     source("Sim_Results_Summary/summary_IPW_CBPS_SKB_GKB_EB_SLDB.R")
     source("Sim_Results_Summary/summary_HAL_AIPW.R")
     ```
   - Performance metrics and execution times corresponding to Table 1 and Table 2 will be displayed in the console.

2. **Reproduce Data Analysis (Section 6)**
   - The 401(k) dataset is loaded directly via `DoubleML::fetch_401k()` within the scripts; no separate download is required.
   - To reproduce Section 6, run the scripts in `Data_Code/`:
     ```R
     source("Data_Code/data_IPW_CBPS_SKB.R")
     source("Data_Code/data_GKB_EB_SLDB.R")
     source("Data_Code/data_HAL.R")
     source("Data_Code/data_AIPW.R")
     ```
   - Estimates are printed to the console. The core results are reported in Table 3.

---

## 4. Contact

For inquiries regarding code and data reproducibility, please contact Haoran Zhang (haoran32@illinois.edu).
