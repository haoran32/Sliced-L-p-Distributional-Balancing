## Gaussian, EBW, IPW, CBPS, and SKB weighting estimators.
## Each method is available through competitor_weights().

COMPETITOR_METHODS <- c("Gaussian", "EBW", "IPW", "CBPS", "SKB")


## Estimands

## Weights need not be normalized because per-arm scaling cancels.
weighted_contrast <- function(w, Z, V) {
  w1 <- w * Z
  w0 <- w * (1 - Z)
  s1 <- sum(w1)
  s0 <- sum(w0)
  if (!is.finite(s1) || !is.finite(s0) || s1 <= 0 || s0 <= 0) return(NA_real_)
  sum(w1 * V) / s1 - sum(w0 * V) / s0
}

estimate_ate <- function(w, Z, Y) weighted_contrast(w, Z, Y)

## Wald ratio: the instrument is Z, the treatment received is A.
estimate_late <- function(w, Z, Y, A) {
  num <- weighted_contrast(w, Z, Y)
  den <- weighted_contrast(w, Z, A)
  if (!is.finite(num) || !is.finite(den) ||
      abs(den) < .Machine$double.eps^0.5) return(NA_real_)
  num / den
}


## Options

## Gaussian and EBW both use WeightIt's characteristic-function-distance QP,
## with Gaussian and energy kernels respectively. `cfd_lambda` is passed to
## WeightIt unchanged; its default 1e-4 is WeightIt's own. WeightIt adds
## `lambda * (1/n_arm)^2 / 2` to the diagonal while the diagonal itself scales
## as `1/n_arm^2`, so a fixed `lambda` holds the ridge at a constant fraction
## of the diagonal at every sample size -- which is what keeps bootstrap and
## subsampling refits solving the same problem as the point estimate.
## Both methods construct dense n by n kernel matrices. NULL SKB settings use
## skb_tuning(n).
competitor_opts <- function(cfd_lambda = 1e-4,
                            gaussian_bw_scale = 1,
                            dense_max_n = Inf,
                            skb_delta = 5e-4,
                            skb_approx = NULL,
                            skb_c = NULL, skb_l = NULL, skb_s = NULL,
                            quiet = TRUE) {
  structure(as.list(environment()), class = "competitor_opts")
}

## Methods that construct dense n by n programs.
DENSE_METHODS <- c("Gaussian", "EBW")

#' @keywords internal
.check_dense_size <- function(method, n, opts) {
  if (!method %in% DENSE_METHODS) return(invisible(TRUE))
  if (n <= opts$dense_max_n) return(invisible(TRUE))
  stop(method, " builds a dense ", n, " x ", n, " program, above the ",
       "dense_max_n limit of ", opts$dense_max_n,
       ". Raise the limit deliberately if you mean to pay that cost.",
       call. = FALSE)
}

skb_tuning <- function(n, opts = competitor_opts()) {
  if (n <= 200) {
    tune <- list(approx = FALSE, c = max(10, floor(n / 10)), l = NULL, s = NULL)
  } else if (n <= 2000) {
    tune <- list(approx = TRUE, c = 100, l = NULL, s = NULL)
  } else {
    tune <- list(approx = TRUE, c = 300, l = 200, s = 100)
  }
  if (!is.null(opts$skb_approx)) tune$approx <- opts$skb_approx
  if (!is.null(opts$skb_c))      tune$c <- opts$skb_c
  if (!is.null(opts$skb_l))      tune$l <- opts$skb_l
  if (!is.null(opts$skb_s))      tune$s <- opts$skb_s
  ## Both code paths need a sketch strictly smaller than the sample.
  tune$c <- max(2L, min(as.integer(tune$c), n - 1L))
  if (!is.null(tune$l)) tune$l <- max(1L, min(as.integer(tune$l), tune$c - 1L))
  if (!is.null(tune$s) && !is.null(tune$l)) {
    tune$s <- max(1L, min(as.integer(tune$s), tune$l - 1L))
  }
  tune
}


## WeightIt CFD balancing with a common ridge convention.
cfd_weights <- function(X, Z, kernel, opts) {
  lambda <- if (is.null(opts$cfd_lambda)) 1e-4 else opts$cfd_lambda
  if (!is.numeric(lambda) || length(lambda) != 1L ||
      !is.finite(lambda) || lambda < 0) {
    stop("`cfd_lambda` must be one nonnegative finite number.", call. = FALSE)
  }
  df <- data.frame(Z = Z, X)
  args <- list(
    formula = Z ~ ., data = df, method = "cfd", estimand = "ATE",
    improved = TRUE, kernel = kernel, lambda = lambda)
  if (identical(kernel, "gaussian")) {
    args$bw_scale <- opts$gaussian_bw_scale
  }
  do.call(WeightIt::weightit, args)$weights
}


## SKB: OSQP kernel SBW (Kim 2023)

.skb <- new.env(parent = globalenv())

#' Load Kim's kernel-SBW code.
#'
#' Both files are expected in the working directory. They are sourced into a
#' private environment to avoid placing their helper functions in the global
#' environment.
#'
#' @param cpp,utils paths to the two SKB source files
skb_load <- function(cpp = "Core/RBF_kernel_C_parallel.cpp", utils = "Core/utils.R",
                     quiet = TRUE, force = FALSE) {
  if (!force && is.function(.skb$kernel.basis)) return(invisible(NULL))
  missing_files <- c(cpp, utils)[!file.exists(c(cpp, utils))]
  if (length(missing_files)) {
    stop("SKB sources not found: ", paste(missing_files, collapse = ", "),
         ". Expected them in the working directory (",
         normalizePath(".", winslash = "/"), ").", call. = FALSE)
  }
  Rcpp::sourceCpp(cpp, env = .skb, verbose = !quiet)
  sys.source(utils, envir = .skb)
  invisible(c(cpp = cpp, utils = utils))
}

skb_weights <- function(X, Z, opts = competitor_opts()) {
  n  <- nrow(X)
  n1 <- sum(Z == 1)
  n0 <- n - n1
  if (n1 == 0 || n0 == 0) return(NULL)
  skb_load()
  tune <- skb_tuning(n, opts)

  ## OSQP block structure below assumes treated rows come first.
  ord <- order(Z, decreasing = TRUE)
  Xs  <- X[ord, , drop = FALSE]
  Zs  <- Z[ord]

  B <- .skb$kernel.basis(Xs, Zs, NULL,
                         kernel.approximation = tune$approx,
                         dim.reduction = tune$approx,
                         c = tune$c, l = tune$l, s = tune$s)
  B  <- as.matrix(B)
  nB <- ncol(B)
  Bt <- B[Zs == 1, , drop = FALSE]
  Bc <- B[Zs == 0, , drop = FALSE]
  target <- colMeans(B)

  ## Build the identity directly as a sparse matrix.
  P <- Matrix::sparseMatrix(i = seq_len(n), j = seq_len(n), x = 1,
                            dims = c(n, n))
  q <- c(rep(-1 / n1, n1), rep(-1 / n0, n0))
  Amat <- Matrix::Matrix(rbind(
    c(rep(1, n1), rep(0, n0)),
    c(rep(0, n1), rep(1, n0)),
    P,
    cbind(t(Bt), matrix(0, nrow = nB, ncol = n0)),
    cbind(matrix(0, nrow = nB, ncol = n1), t(Bc))
  ), sparse = TRUE)

  d <- opts$skb_delta
  lvec <- c(1, 1, rep(0, n), target - d, target - d)
  uvec <- c(1, 1, rep(1, n), target + d, target + d)

  model <- osqp::osqp(P, q, Amat, lvec, uvec,
                      osqp::osqpSettings(alpha = 1.5, verbose = FALSE))
  res <- .osqp_solve(model)

  ## Accept OSQP solutions that satisfy its relaxed tolerances.
  if (!res$info$status %in% c("solved", "solved inaccurate")) return(NULL)

  w <- numeric(n)
  w[ord] <- res$x
  w
}

## Support both osqp model APIs.
.osqp_solve <- function(model) {
  f <- tryCatch(model@Solve, error = function(e) NULL)
  if (is.function(f)) return(f())
  model$Solve()
}


## Unified weight dispatch

#' Balancing weights from a competing method.
#'
#' @param X numeric covariate matrix (n x p)
#' @param Z 0/1 treatment or instrument vector of length n
#' @param method one of COMPETITOR_METHODS
#' @param opts competitor_opts()
#' @return numeric weight vector of length n, or NULL if the method failed
competitor_weights <- function(X, Z, method, opts = competitor_opts()) {
  method <- match.arg(method, COMPETITOR_METHODS)
  X <- as.matrix(X)
  if (is.null(colnames(X))) colnames(X) <- paste0("V", seq_len(ncol(X)))
  Z <- as.integer(Z)
  n <- nrow(X)
  if (sum(Z == 1L) == 0L || sum(Z == 0L) == 0L) return(NULL)

  run <- function(expr) if (isTRUE(opts$quiet)) suppressWarnings(expr) else expr
  .check_dense_size(method, n, opts)

  w <- switch(
    method,

    Gaussian = run(cfd_weights(X, Z, kernel = "gaussian", opts = opts)),

    EBW = run(cfd_weights(X, Z, kernel = "energy", opts = opts)),

    IPW = {
      df <- data.frame(Z = Z, X)
      run(WeightIt::weightit(Z ~ ., data = df, method = "glm",
                             estimand = "ATE")$weights)
    },

    CBPS = {
      df <- data.frame(Z = Z, X)
      run(WeightIt::weightit(Z ~ ., data = df, method = "cbps",
                             estimand = "ATE")$weights)
    },

    SKB = run(skb_weights(X, Z, opts))
  )

  if (is.null(w) || length(w) != n || anyNA(w)) return(NULL)
  as.numeric(w)
}


## Caching and estimator closures

#' Compatibility hook for competitor-specific precomputation.
#' WeightIt constructs the CFD kernel internally for each fit.
competitor_cache <- function(X, Z, method, opts = competitor_opts()) {
  NULL
}

#' Index-indexed estimator closure, matching the interface that
#' sldb_subsample_ci() and sldb_boot_ci() expect.
#'
#' @param A optional treatment-received vector; when supplied the closure
#'   returns the LATE, otherwise the ATE.
competitor_estimator <- function(X, Z, Y, A = NULL, method,
                                 opts = competitor_opts(), cache = NULL) {
  X <- as.matrix(X)
  if (is.null(colnames(X))) colnames(X) <- paste0("V", seq_len(ncol(X)))
  Z <- as.integer(Z)
  Y <- as.numeric(Y)
  if (!is.null(A)) A <- as.numeric(A)
  function(idx) {
    Zi <- Z[idx]
    if (sum(Zi == 1L) == 0L || sum(Zi == 0L) == 0L) return(NA_real_)
    w <- competitor_weights(X[idx, , drop = FALSE], Zi, method, opts)
    if (is.null(w)) return(NA_real_)
    if (is.null(A)) estimate_ate(w, Zi, Y[idx])
    else            estimate_late(w, Zi, Y[idx], A[idx])
  }
}


