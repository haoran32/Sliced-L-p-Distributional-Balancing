## SLDB: Sliced L^p Distributional Balancing
## Call sldb_load_engine() before using this file outside a package.


## Engine loading

## sourceCpp evaluates its generated glue code in this environment.
#' @keywords internal
#' @noRd
.sldb <- new.env(parent = globalenv())

#' Compile and load the SLDB C++ engine
#'
#' Compiles `swweight.cpp` with [Rcpp::sourceCpp()] into a private environment.
#' Only needed when `SLDB.R` is used standalone (i.e. via `source()`); inside an
#' installed package the engine is already available and this call is a no-op.
#'
#' @param path Path to `swweight.cpp`. If `NULL` (the default), it is looked
#'   for in the working directory, which is where the analysis scripts keep
#'   it; `src/swweight.cpp` is tried afterwards for package-style layouts.
#' @param quiet Passed to [Rcpp::sourceCpp()] as `verbose = !quiet`.
#' @param force Recompile even if an engine is already loaded.
#'
#' @return Invisibly, the path that was compiled (or `NULL` if already loaded).
#' @export
sldb_load_engine <- function(path = NULL, quiet = TRUE, force = FALSE) {
  if (!force && !is.null(.sldb$engine)) return(invisible(NULL))

  if (is.null(path)) {
    candidates <- c("swweight.cpp", "Core/swweight.cpp", "src/swweight.cpp")
    hit <- candidates[file.exists(candidates)]
    if (length(hit) == 0L) {
      stop("Could not find swweight.cpp. Pass its location via `path`.",
           call. = FALSE)
    }
    path <- hit[1L]
  }
  if (!file.exists(path)) {
    stop("No such file: ", path, call. = FALSE)
  }

  Rcpp::sourceCpp(path, env = .sldb, verbose = !quiet)
  if (!is.function(.sldb$get_sw_weights_cpp)) {
    stop("Compiled ", path, " but get_sw_weights_cpp() was not exported.",
         call. = FALSE)
  }
  if (!is.function(.sldb$prepare_sw_p2_cpp) ||
      !is.function(.sldb$sw_obj_grad_p2_cpp)) {
    stop("Compiled ", path, " but the p=2 engine was not exported.",
         call. = FALSE)
  }
  .sldb$engine <- .sldb$get_sw_weights_cpp
  invisible(path)
}

#' @keywords internal
#' @noRd
.sldb_engine <- function() {
  if (!is.null(.sldb$engine)) return(.sldb$engine)
  ## In a package, the Rcpp export is already on the search path.
  if (exists("get_sw_weights_cpp", mode = "function")) {
    return(get("get_sw_weights_cpp", mode = "function"))
  }
  stop("The SLDB C++ engine is not loaded. ",
       "Call sldb_load_engine(\"path/to/swweight.cpp\") first.", call. = FALSE)
}

#' @keywords internal
#' @noRd
.sldb_p2_engine <- function() {
  if (is.function(.sldb$prepare_sw_p2_cpp) &&
      is.function(.sldb$sw_obj_grad_p2_cpp)) {
    return(list(prepare = .sldb$prepare_sw_p2_cpp,
                evaluate = .sldb$sw_obj_grad_p2_cpp))
  }
  if (exists("prepare_sw_p2_cpp", mode = "function") &&
      exists("sw_obj_grad_p2_cpp", mode = "function")) {
    return(list(prepare = get("prepare_sw_p2_cpp", mode = "function"),
                evaluate = get("sw_obj_grad_p2_cpp", mode = "function")))
  }
  stop("The SLDB p=2 engine is not loaded. Call sldb_load_engine() first.",
       call. = FALSE)
}


## OpenMP thread selection

#' Select the number of OpenMP threads used by SLDB
#'
#' Uses no more than `max_threads`. On a scheduler, the available count is read
#' from `SLURM_CPUS_PER_TASK` or `SLURM_CPUS_ON_NODE`; otherwise it is the
#' machine's physical-core count. An `OMP_THREAD_LIMIT` supplied by the runtime
#' is also respected. Call this before [sldb_load_engine()] so the OpenMP
#' runtime sees `OMP_NUM_THREADS` when the compiled engine is loaded.
#'
#' @param max_threads Positive integer upper bound. Default `10`.
#' @param logical Passed to [parallel::detectCores()] outside a scheduler.
#'   The default `FALSE` uses physical cores.
#'
#' @return Invisibly, the positive integer thread count selected. The same
#'   value is stored in `getOption("sldb.threads")`.
#' @export
sldb_set_threads <- function(max_threads = 10L, logical = FALSE) {
  if (!is.numeric(max_threads) || length(max_threads) != 1L ||
      !is.finite(max_threads) || max_threads < 1 ||
      max_threads != as.integer(max_threads)) {
    stop("`max_threads` must be one positive integer.", call. = FALSE)
  }
  if (!is.logical(logical) || length(logical) != 1L || is.na(logical)) {
    stop("`logical` must be TRUE or FALSE.", call. = FALSE)
  }

  env_int <- function(name) {
    value <- suppressWarnings(as.integer(Sys.getenv(name, unset = NA_character_)))
    if (length(value) == 1L && is.finite(value) && value > 0L) value else NA_integer_
  }

  scheduler_counts <- vapply(c("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"),
                             env_int, integer(1L))
  scheduler_counts <- scheduler_counts[is.finite(scheduler_counts)]
  available <- if (length(scheduler_counts)) scheduler_counts[1L] else {
    detected <- parallel::detectCores(logical = logical)
    if (length(detected) != 1L || !is.finite(detected) || detected < 1L) 1L
    else as.integer(detected)
  }

  thread_limit <- env_int("OMP_THREAD_LIMIT")
  if (is.finite(thread_limit)) available <- min(available, thread_limit)

  threads <- as.integer(min(max_threads, available))
  Sys.setenv(OMP_NUM_THREADS = as.character(threads))
  options(sldb.threads = threads)
  invisible(threads)
}


## Tuning

#' Tuning parameters for SLDB
#'
#' By default, the ridge is calibrated from the unpenalized sliced-L^p
#' loss at weights that are uniform within treatment arms:
#' \deqn{\lambda_n = \rho\,
#'   \frac{\widehat{\mathcal L}_n(w_{\mathrm{constant}})}
#'        {\lVert w_{\mathrm{constant}}\rVert_2^2}.}
#' The default number of random projections depends on the sample size and the
#' covariate dimension \eqn{d}:
#' \deqn{L_n = \lceil c_L n^{\gamma_L}\rceil,}
#' where \eqn{c_L=1} and \eqn{\gamma_L=1+1/(4d)} by default. The default
#' proportion uses the same exponent,
#' \deqn{\rho = n^{-\gamma_L},}
#' evaluated at the fitted sample size unless `n` is supplied here.
#' Optimization starts from the weights that are uniform within treatment arms.
#' Setting `initial_method = "logistic"` additionally fits a main-effects
#' logistic propensity-score model, forms normalized ATE IPW weights, and
#' starts from whichever candidate has the smaller objective. The reference
#' objective is the smaller objective over the candidates actually evaluated,
#' \deqn{J_{\mathrm{ref}} = \min\{\widehat J_n(w_{\mathrm{uniform}}),
#'   \widehat J_n(w_{\mathrm{IPW}})\},}
#' which is the uniform objective under the default start. For `p = 1`, the
#' default certified-gap tolerance is
#' \deqn{\delta_n = 0.2\,J_{\mathrm{ref}}.}
#' If the logistic fit fails, the routine safely falls back to uniform weights.
#' Setting `delta_fraction = NULL` uses
#' \eqn{\delta_n=c_{\delta}\lambda_n/n} with `delta_coef`.
#' For `p = 2`, optimization always starts from uniform within-arm weights;
#' the logistic model is not fitted, delta is not computed or used, and
#' convergence is determined only by `optim()`.
#'
#' For bootstrap inference, \eqn{\lambda_N}, \eqn{L_N}, and \eqn{\delta_N}
#' retain their full-sample values. For a subsampling refit of size \eqn{m},
#' [sldb_estimator()] retains \eqn{\lambda_N} while using
#' \eqn{L_m=\lceil c_L m^{\gamma_L}\rceil} and, under the coefficient schedule,
#' \deqn{\delta_m = c_{\delta}\lambda_N/m.}
#' Under the reference-objective schedule the tolerance is instead rebuilt from
#' the subsample itself,
#' \deqn{\delta_m = \phi\,J_{\mathrm{ref}}(\text{subsample}),}
#' with the same fraction \eqn{\phi} as the full sample. The objective is an
#' average over projections rather than a sum, so it does not shrink with the
#' sample size and a tolerance carried over as \eqn{\delta_N N/m} would be
#' loosened by the factor \eqn{N/m}.
#'
#' @param d Optional positive integer covariate dimension \eqn{d}. Supplying
#'   it resolves the default `L_exponent` to \eqn{1+1/(4d)} eagerly, so the
#'   schedule is fixed in the control object rather than at fit time.
#' @param n Optional positive integer sample size. Supplying it, together with
#'   `d` or an explicit `L_exponent`, resolves the default
#'   `proportion_to_initial_loss` to \eqn{n^{-\gamma_L}} eagerly. When omitted,
#'   the same rule is applied at fit time using the number of rows of `X`.
#' @param c_opt Alias for `delta_coef`. Do not supply both with different
#'   values. Explicitly supplying either selects the
#'   `delta_coef * lambda_n / n` tolerance instead of `delta_fraction`.
#' @param alpha Weight on the treated-vs-control balance term in the objective.
#'   `alpha = 0` gives two-way balancing.
#' @param p Sliced CDF discrepancy order. Both orders are minimised by the
#'   same projected first-order solver under the same certified-gap rule:
#'   `p = 1` takes a subgradient step, `p = 2` a gradient step.
#' @param eta_step Initial projected step. Default `1`.
#' @param eta_window Even number of iterations used for each adaptive-step
#'   progress check. Default `10`.
#' @param eta_min,eta_max Lower and upper bounds for the adaptive step.
#' @param eta_increase Multiplicative increase after stable stagnation.
#' @param eta_decrease Multiplicative decrease after worsening or oscillation.
#' @param eta_progress Minimum relative objective reduction over a window that
#'   counts as adequate progress. Default `0.005`.
#' @param eta_gap_multiplier The step is not increased when the certified gap
#'   is at most this multiple of `delta_n`. Default `10`.
#' @param max_iter Maximum projected-solver iterations.
#' @param lambda Optional fixed ridge override. When omitted, initial-loss
#'   calibration is used.
#' @param L Optional lower bound on the number of projections. The actual count
#'   is at least the value specified by `L_coefficient` and `L_exponent`.
#' @param L_coefficient Positive coefficient \eqn{c_L} in the projection-count
#'   schedule \eqn{L_n=\lceil c_L n^{\gamma_L}\rceil}. Default `1`.
#' @param L_exponent Optional nonnegative exponent \eqn{\gamma_L} in the
#'   projection-count schedule. The default `NULL` evaluates dynamically to
#'   \eqn{1+1/(4d)} once the covariate dimension is known.
#' @param verbose If `TRUE`, the solver prints convergence progress.
#' @param proportion_to_initial_loss Positive proportion \eqn{\rho} used to
#'   calibrate the ridge from the initial unpenalized loss. The default
#'   `"auto"` uses \eqn{\rho = n^{-\gamma_L}}, resolved from `n` when supplied
#'   and otherwise at fit time. Values greater than one are allowed. Set to
#'   `NULL` when using a fixed `lambda`.
#' @param delta_coef Positive coefficient \eqn{c_{\delta}} in
#'   the optional schedule \eqn{\delta_n = c_{\delta}\lambda_n/n}. If omitted,
#'   `c_opt` is used. Ignored for `p = 2`.
#' @param delta_fraction Positive fraction of the reference objective used as
#'   the default certified-gap tolerance. The reference is the smaller
#'   objective over the start candidates that were evaluated. Default `0.2`
#'   (20 percent) when `p = 1`, and always `NULL` when `p = 2`, which neither
#'   computes nor uses delta.
#'   Set to `NULL` to use the `delta_coef` schedule. Ignored for `p = 2`.
#' @param initial_method Initialization method. The default `"uniform"` starts
#'   from the weights that are uniform within treatment arms and fits no
#'   propensity model. `"logistic"` fits a main-effects logistic propensity
#'   model, forms normalized ATE IPW weights, and starts from them only when
#'   their objective is below the uniform-weight objective. This setting is
#'   used only for `p = 1`; `p = 2` always starts from uniform weights.
#'
#' @return An object of class `"sldb_control"`.
#' @export
#'
#' @examples
#' ctrl <- sldb_control(d = 10, n = 1000)
#' sldb_L(ctrl, 1000, d = 10)
sldb_control <- function(d = NULL,
                         n = NULL,
                         c_opt = 1,
                         alpha = 1,
                         p = 1,
                         eta_step = 1,
                         eta_window = 10L,
                         eta_min = 1e-4,
                         eta_max = 10,
                         eta_increase = 2,
                         eta_decrease = 0.5,
                         eta_progress = 0.005,
                         eta_gap_multiplier = 10,
                         max_iter = 10000L,
                         lambda = NULL,
                         L = NULL,
                         verbose = FALSE,
                         proportion_to_initial_loss = "auto",
                         delta_coef = NULL,
                         delta_fraction = NULL,
                         L_coefficient = 1,
                         L_exponent = NULL,
                         initial_method = c("uniform", "logistic")) {
  lambda_was_supplied <- !missing(lambda) && !is.null(lambda)
  proportion_was_supplied <- !missing(proportion_to_initial_loss)
  c_opt_was_supplied <- !missing(c_opt)
  delta_coef_was_supplied <- !missing(delta_coef) && !is.null(delta_coef)
  delta_fraction_was_supplied <- !missing(delta_fraction)
  initial_method <- match.arg(initial_method)

  if (!is.null(d)) {
    if (!is.numeric(d) || length(d) != 1L || !is.finite(d) || d < 1 ||
        d != as.integer(d)) {
      stop("`d` must be one positive integer covariate dimension.",
           call. = FALSE)
    }
    d <- as.integer(d)
  }
  if (!is.null(n)) {
    if (!is.numeric(n) || length(n) != 1L || !is.finite(n) || n < 1 ||
        n != as.integer(n)) {
      stop("`n` must be one positive integer sample size.", call. = FALSE)
    }
    n <- as.integer(n)
  }

  if (!is.numeric(p) || length(p) != 1L || !is.finite(p) ||
      !(p %in% c(1, 2))) {
    stop("`p` must be either 1 or 2.", call. = FALSE)
  }

  ## Default tolerance: a fraction of the reference objective. Both orders
  ## run the same projected solver under the same certified-gap rule, so the
  ## default is the same for both.
  if (!delta_fraction_was_supplied) {
    delta_fraction <- 0.1
  }

  ## The default projection exponent is 1 + 1/(4d); supplying `d` pins it here
  ## instead of leaving it to be evaluated at fit time.
  if (is.null(L_exponent) && !is.null(d)) L_exponent <- 1 + 1 / (4 * d)

  ## The default proportion is n^(-L_exponent). It is resolved here when both
  ## the sample size and the exponent are known, and otherwise left as "auto"
  ## for the fit-time resolver, which reads them off X.
  if (!proportion_was_supplied && !lambda_was_supplied &&
      !is.null(n) && !is.null(L_exponent)) {
    proportion_to_initial_loss <- n^(-L_exponent)
  }
  if (!proportion_was_supplied && lambda_was_supplied) {
    proportion_to_initial_loss <- NULL
  }
  if (!is.null(proportion_to_initial_loss) &&
      lambda_was_supplied) {
    stop("Choose `proportion_to_initial_loss` or `lambda`, not both.",
         call. = FALSE)
  }
  if (is.null(proportion_to_initial_loss) && is.null(lambda)) {
    stop("Supply `proportion_to_initial_loss` or a fixed `lambda`.",
         call. = FALSE)
  }

  ## Explicit coefficients select the coefficient schedule by default.
  if ((c_opt_was_supplied || delta_coef_was_supplied) &&
      !delta_fraction_was_supplied) {
    delta_fraction <- NULL
  }
  if (!is.null(delta_fraction) &&
      (c_opt_was_supplied || delta_coef_was_supplied)) {
    stop("Choose either `delta_fraction` or `delta_coef`/`c_opt`, not both.",
         call. = FALSE)
  }

  if (!is.null(delta_coef)) {
    if (c_opt_was_supplied && !isTRUE(all.equal(c_opt, delta_coef))) {
      stop("Supply only one of `c_opt` and `delta_coef`, or give them the ",
           "same value.", call. = FALSE)
    }
    c_opt <- delta_coef
  }
  delta_coef <- c_opt

  stopifnot(is.numeric(c_opt), length(c_opt) == 1L, c_opt > 0)
  if (!is.null(lambda)) {
    stopifnot(is.numeric(lambda), length(lambda) == 1L,
              is.finite(lambda), lambda > 0)
  }
  if (!is.null(L)) {
    stopifnot(is.numeric(L), length(L) == 1L, is.finite(L),
              L >= 1, L == as.integer(L))
  }
  if (!is.numeric(L_coefficient) || length(L_coefficient) != 1L ||
      !is.finite(L_coefficient) || L_coefficient <= 0) {
    stop("`L_coefficient` must be one positive finite number.", call. = FALSE)
  }
  if (!is.null(L_exponent) &&
      (!is.numeric(L_exponent) || length(L_exponent) != 1L ||
       !is.finite(L_exponent) || L_exponent < 0)) {
    stop("`L_exponent` must be NULL or one nonnegative finite number.",
         call. = FALSE)
  }
  if (!is.null(proportion_to_initial_loss) &&
      !identical(proportion_to_initial_loss, "auto")) {
    stopifnot(is.numeric(proportion_to_initial_loss),
              length(proportion_to_initial_loss) == 1L,
              is.finite(proportion_to_initial_loss),
              proportion_to_initial_loss > 0)
  }
  stopifnot(is.numeric(delta_coef), length(delta_coef) == 1L,
            is.finite(delta_coef), delta_coef > 0)
  if (!is.null(delta_fraction)) {
    stopifnot(is.numeric(delta_fraction), length(delta_fraction) == 1L,
              is.finite(delta_fraction), delta_fraction > 0,
              delta_fraction <= 1)
  }
  stopifnot(is.numeric(alpha), length(alpha) == 1L, alpha >= 0)
  if (!is.numeric(eta_window) || length(eta_window) != 1L ||
      !is.finite(eta_window) || eta_window < 4 ||
      eta_window != as.integer(eta_window) || eta_window %% 2 != 0) {
    stop("`eta_window` must be an even integer of at least 4.", call. = FALSE)
  }
  if (!is.numeric(eta_min) || length(eta_min) != 1L ||
      !is.finite(eta_min) || eta_min <= 0 ||
      !is.numeric(eta_max) || length(eta_max) != 1L ||
      !is.finite(eta_max) || eta_max < eta_min) {
    stop("Require `0 < eta_min <= eta_max`.", call. = FALSE)
  }
  if (!is.numeric(eta_step) || length(eta_step) != 1L ||
      !is.finite(eta_step) || eta_step < eta_min || eta_step > eta_max) {
    stop("`eta_step` must lie between `eta_min` and `eta_max`.", call. = FALSE)
  }
  if (!is.numeric(eta_increase) || length(eta_increase) != 1L ||
      !is.finite(eta_increase) || eta_increase <= 1 ||
      !is.numeric(eta_decrease) || length(eta_decrease) != 1L ||
      !is.finite(eta_decrease) || eta_decrease <= 0 || eta_decrease >= 1 ||
      !is.numeric(eta_progress) || length(eta_progress) != 1L ||
      !is.finite(eta_progress) || eta_progress < 0 ||
      !is.numeric(eta_gap_multiplier) || length(eta_gap_multiplier) != 1L ||
      !is.finite(eta_gap_multiplier) || eta_gap_multiplier < 1) {
    stop("Invalid adaptive-eta settings.", call. = FALSE)
  }
  if (!is.numeric(max_iter) || length(max_iter) != 1L ||
      !is.finite(max_iter) || max_iter < 1 || max_iter != as.integer(max_iter)) {
    stop("`max_iter` must be one positive integer.", call. = FALSE)
  }
  structure(
    list(c_opt = delta_coef,
         d = d, n = n,
         proportion_to_initial_loss = proportion_to_initial_loss,
         delta_coef = delta_coef, delta_fraction = delta_fraction,
         initial_method = initial_method,
         alpha = alpha, p = as.integer(p), eta_step = eta_step,
         eta_window = as.integer(eta_window), eta_min = eta_min,
         eta_max = eta_max, eta_increase = eta_increase,
         eta_decrease = eta_decrease, eta_progress = eta_progress,
         eta_gap_multiplier = eta_gap_multiplier,
         max_iter = as.integer(max_iter),
         lambda = lambda, L = L,
         L_coefficient = as.numeric(L_coefficient),
         L_exponent = if (is.null(L_exponent)) NULL else as.numeric(L_exponent),
         verbose = isTRUE(verbose)),
    class = "sldb_control"
  )
}

#' @keywords internal
#' @noRd
.sldb_L_exponent <- function(control, d = NULL) {
  if (!is.null(control$L_exponent)) return(control$L_exponent)
  if (is.null(d)) d <- control$d
  if (is.null(d) || !is.numeric(d) || length(d) != 1L || !is.finite(d) ||
      d < 1) {
    stop("`d` must be one positive integer covariate dimension when ",
         "`L_exponent` is NULL.", call. = FALSE)
  }
  1 + 1 / (4 * d)
}

#' Resolve the deferred `"auto"` proportion once the data are known
#'
#' The default `proportion_to_initial_loss` is `n^(-L_exponent)`. When neither
#' `n` nor `d` was given to [sldb_control()], both are read off the fitted data
#' here. Refits that freeze the full-sample ridge set the field to `NULL`
#' before calling the solver, so they are left untouched.
#' @keywords internal
#' @noRd
.sldb_resolve_control <- function(control, n, d) {
  if (identical(control$proportion_to_initial_loss, "auto")) {
    control$proportion_to_initial_loss <- n^(-.sldb_L_exponent(control, d))
  }
  control
}

#' Ridge and projection-count schedules
#'
#' @param control An [sldb_control()] object.
#' @param n Sample size at which to evaluate the schedule.
#' @param d Positive integer covariate dimension. Required by `sldb_L()` when
#'   `control$L_exponent` is `NULL`, so the default exponent can be evaluated.
#' @param lambda Selected ridge, used by `sldb_delta`. `sldb_L` ignores it: the
#'   projection count no longer depends on the ridge.
#'
#' @return A single numeric (`sldb_lambda`, `sldb_delta`) or integer (`sldb_L`).
#' @name sldb_schedule
#' @export
sldb_lambda <- function(control, n) {
  if (!is.null(control$lambda)) return(control$lambda)
  NA_real_
}

#' @rdname sldb_schedule
#' @export
sldb_L <- function(control, n, d = NULL, lambda = NULL) {
  if (!is.finite(n) || n < 1 || n > .Machine$integer.max) {
    stop("`n` must be one positive integer below the projection-count limit.",
         call. = FALSE)
  }
  L_coefficient <- if (is.null(control$L_coefficient)) {
    1
  } else {
    control$L_coefficient
  }
  L_exponent <- .sldb_L_exponent(control, d)
  scheduled_L <- ceiling(L_coefficient * n^L_exponent)
  if (!is.finite(scheduled_L) || scheduled_L > .Machine$integer.max) {
    stop("The scheduled projection count exceeds R's integer limit.",
         call. = FALSE)
  }
  max(as.integer(scheduled_L),
      if (is.null(control$L)) 1L else as.integer(control$L))
}

#' @rdname sldb_schedule
#' @export
sldb_delta <- function(control, n) {
  p_order <- if (is.null(control$p)) 1L else as.integer(control$p)
  if (p_order == 2L) return(NA_real_)
  if (!is.null(control$delta_fraction)) return(NA_real_)
  delta_coef <- if (is.null(control$delta_coef)) control$c_opt else control$delta_coef
  delta_coef * sldb_lambda(control, n) / n
}

#' @export
print.sldb_control <- function(x, ...) {
  p_order <- if (is.null(x$p)) 1L else x$p
  cat("<sldb_control>\n")
  cat(sprintf("  p         : %d (%s)\n", p_order,
              if (p_order == 2L) "projected gradient" else
                "projected subgradient"))
  if (identical(x$proportion_to_initial_loss, "auto")) {
    cat(sprintf("  lambda   : n^-%s * Lhat(w_constant) / ||w_constant||^2\n",
                if (is.null(x$L_exponent)) "(1 + 1/(4d))" else
                  format(x$L_exponent)))
  } else if (!is.null(x$proportion_to_initial_loss)) {
    cat(sprintf("  lambda   : %g * Lhat(w_constant) / ||w_constant||^2\n",
                x$proportion_to_initial_loss))
  } else {
    cat(sprintf("  lambda   : fixed at %s\n", format(x$lambda)))
  }
  L_coefficient <- if (is.null(x$L_coefficient)) 1 else x$L_coefficient
  if (is.null(x$L_exponent)) {
    cat(sprintf("  L_n      : ceiling(%g * n^(1 + 1/(4d)))", L_coefficient))
  } else {
    cat(sprintf("  L_n      : ceiling(%g * n^%g)",
                L_coefficient, x$L_exponent))
  }
  if (!is.null(x$L)) cat(sprintf(", with lower bound %s", format(x$L)))
  cat("\n")
  if (p_order == 2L) {
    cat("  delta_n  : not used for p = 2\n")
  } else if (!is.null(x$delta_fraction)) {
    cat(sprintf("  delta_n  : %g * J_ref\n", x$delta_fraction))
  } else {
    delta_coef <- if (is.null(x$delta_coef)) x$c_opt else x$delta_coef
    cat(sprintf("  delta_n  : %g * lambda_n / n\n", delta_coef))
  }
  if (p_order == 2L) {
    cat("  start     : uniform within treatment arms\n")
  } else {
    initial_method <- if (is.null(x$initial_method)) "uniform" else x$initial_method
    if (identical(initial_method, "uniform")) {
      cat("  start     : uniform within treatment arms\n")
    } else {
      cat(sprintf("  start     : %s candidate versus uniform\n",
                  initial_method))
    }
  }
  cat(sprintf("  eta       : adaptive from %g every %d iterations [%g, %g]\n",
              x$eta_step, x$eta_window, x$eta_min, x$eta_max))
  cat(sprintf("  alpha = %g,  max_iter = %d\n", x$alpha, x$max_iter))
  invisible(x)
}


## Weights

#' @keywords internal
#' @noRd
.sldb_check_data <- function(X, Z) {
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  Z <- as.integer(Z)
  if (nrow(X) != length(Z)) {
    stop("X and Z imply different sample sizes.", call. = FALSE)
  }
  if (!all(Z %in% c(0L, 1L))) {
    stop("Z must be coded 0/1.", call. = FALSE)
  }
  if (sum(Z == 1L) == 0L || sum(Z == 0L) == 0L) {
    stop("Both treatment arms must be non-empty.", call. = FALSE)
  }
  if (anyNA(X) || anyNA(Z)) stop("X and Z must not contain NA.", call. = FALSE)
  list(X = X, Z = Z)
}

#' @keywords internal
#' @noRd
.sldb_project_simplex <- function(v) {
  u <- sort(v, decreasing = TRUE)
  css <- cumsum(u)
  keep <- which(u > (css - 1) / seq_along(u))
  theta <- (css[max(keep)] - 1) / max(keep)
  pmax(v - theta, 0)
}

#' @keywords internal
#' @noRd
.sldb_initial_weights <- function(X, Z, method = c("uniform", "logistic"),
                                  ps_clip = 1e-6) {
  method <- match.arg(method)
  n <- length(Z)
  idx1 <- which(Z == 1L)
  idx0 <- which(Z == 0L)
  w_uniform <- numeric(n)
  w_uniform[idx1] <- 1 / length(idx1)
  w_uniform[idx0] <- 1 / length(idx0)

  fallback <- function(reason = "uniform requested") {
    list(weights = w_uniform, method = "uniform",
         logistic_fit_converged = FALSE, fallback_reason = reason)
  }
  if (method == "uniform") return(fallback())

  design <- cbind(`(Intercept)` = 1, X)
  fit <- suppressWarnings(tryCatch(
    stats::glm.fit(x = design, y = Z, family = stats::binomial()),
    error = function(e) NULL
  ))
  if (is.null(fit) || !isTRUE(fit$converged) ||
      length(fit$fitted.values) != n ||
      any(!is.finite(fit$fitted.values))) {
    return(fallback("logistic propensity fit failed to converge"))
  }

  propensity <- pmin(pmax(as.numeric(fit$fitted.values), ps_clip),
                     1 - ps_clip)
  w <- ifelse(Z == 1L, 1 / propensity, 1 / (1 - propensity))
  arm_sums <- c(sum(w[idx1]), sum(w[idx0]))
  if (any(!is.finite(w)) || any(w <= 0) ||
      any(!is.finite(arm_sums)) || any(arm_sums <= 0)) {
    return(fallback("logistic IPW weights were invalid"))
  }
  w[idx1] <- w[idx1] / arm_sums[1L]
  w[idx0] <- w[idx0] / arm_sums[2L]
  list(weights = as.numeric(w), method = "logistic",
       logistic_fit_converged = TRUE, fallback_reason = "")
}

#' SLDB balancing weights
#'
#' Solves the ridge-penalised sliced-L^p balancing problem. Weights sum
#' to one within each treatment arm. This is a *design-stage* routine: the
#' outcome plays no part in choosing the weights, and none is required.
#'
#' @param X Numeric covariate matrix, `n` by `p`.
#' @param Z Integer treatment vector of length `n`, coded 0/1.
#' @param control An [sldb_control()] object.
#' @param tuning_n Reference sample size retained for compatibility and for
#'   resampling tolerance calculations. Projection counts use the actual number
#'   of rows and the fixed ridge.
#' @param delta Optional fixed certified-gap tolerance. When supplied, it
#'   overrides both `delta_fraction` and the `delta_coef` schedule. Use
#'   the full-sample fit's `delta_n` here for every resampling refit. Ignored
#'   when `p = 2`, for which delta is neither computed nor used.
#'
#' @return An object of class `"sldb_weights"`: a list with `weights`,
#'   `converged`, `iterations`, `final_gap`, `delta_n`, `S_n`, `lambda`, `L`,
#'   adaptive-step diagnostics, `uniform_objective`, `logistic_objective`,
#'   `reference_objective`, `initial_source`, `n`, and `tuning_n`.
#' @export
sldb_weights <- function(X, Z, control = sldb_control(), tuning_n = NULL,
                         delta = NULL) {
  d <- .sldb_check_data(X, Z)
  X <- d$X; Z <- d$Z
  n <- length(Z)

  if (is.null(tuning_n)) tuning_n <- n
  if (!is.numeric(tuning_n) || length(tuning_n) != 1L ||
      !is.finite(tuning_n) || tuning_n < 1 || tuning_n != as.integer(tuning_n)) {
    stop("`tuning_n` must be one positive integer.", call. = FALSE)
  }
  tuning_n <- as.integer(tuning_n)

  control <- .sldb_resolve_control(control, n = n, d = ncol(X))

  p_order <- if (is.null(control$p)) 1L else as.integer(control$p)

  initial_method <- if (is.null(control$initial_method)) {
    "uniform"
  } else {
    control$initial_method
  }
  initializer <- .sldb_initial_weights(X, Z, method = initial_method)

  engine <- .sldb_engine()
  adaptive_lambda <- !is.null(control$proportion_to_initial_loss)
  calibration <- NULL
  calibration_L <- NA_integer_
  if (adaptive_lambda) {
    calibration_L <- sldb_L(control, n, d = ncol(X))
    calibration <- engine(Y = numeric(n), Z = Z, X = X, L = calibration_L,
                          max_iter = 0L,
                          eta_step = control$eta_step,
                          c_opt = control$c_opt,
                          alpha = control$alpha,
                          lambda = -1,
                          output = 0L,
                          delta_n_override = -1,
                          delta_uniform_fraction = -1,
                          proportion_to_initial_loss =
                            control$proportion_to_initial_loss,
                          calibration_only = TRUE,
                          eta_window = control$eta_window,
                          eta_min = control$eta_min,
                          eta_max = control$eta_max,
                          eta_increase = control$eta_increase,
                          eta_decrease = control$eta_decrease,
                          eta_progress = control$eta_progress,
                          eta_gap_multiplier = control$eta_gap_multiplier,
                          initial_weights = NULL,
                          p_order = p_order)
    lambda_n <- as.numeric(calibration$lambda)
  } else {
    lambda_n <- sldb_lambda(control, n)
  }
  L_n <- sldb_L(control, n, d = ncol(X))

  if (!is.null(delta)) {
    if (!is.numeric(delta) || length(delta) != 1L ||
        !is.finite(delta) || delta <= 0) {
      stop("`delta` must be one positive finite number.", call. = FALSE)
    }
    delta_n <- as.numeric(delta)
    delta_uniform_fraction <- -1
  } else if (!is.null(control$delta_fraction)) {
    ## The C++ engine evaluates both reference candidates with the same
    ## projections used for optimization, then replaces this sentinel with the
    ## realized target based on their smaller objective.
    delta_n <- -1
    delta_uniform_fraction <- control$delta_fraction
  } else {
    delta_coef <- if (is.null(control$delta_coef)) control$c_opt else control$delta_coef
    delta_n <- delta_coef * lambda_n / n
    delta_uniform_fraction <- -1
  }

  ## The engine wants a Y argument but only reads it after the weights are
  ## frozen; a zero vector keeps this call strictly design-stage.
  out <- engine(Y = numeric(n), Z = Z, X = X, L = L_n,
                max_iter = control$max_iter,
                eta_step = control$eta_step,
                c_opt = control$c_opt,
                alpha = control$alpha,
                lambda = lambda_n,
                output = as.integer(control$verbose),
                delta_n_override = delta_n,
                delta_uniform_fraction = delta_uniform_fraction,
                proportion_to_initial_loss = -1,
                calibration_only = FALSE,
                eta_window = control$eta_window,
                eta_min = control$eta_min,
                eta_max = control$eta_max,
                eta_increase = control$eta_increase,
                eta_decrease = control$eta_decrease,
                eta_progress = control$eta_progress,
                eta_gap_multiplier = control$eta_gap_multiplier,
                initial_weights = if (identical(initializer$method, "logistic")) {
                  initializer$weights
                } else NULL,
                p_order = p_order)

  selected_initial_loss <- if (adaptive_lambda) {
    as.numeric(calibration$initial_loss)
  } else {
    as.numeric(out$initial_loss)
  }

  structure(
    list(weights = as.numeric(out$weights),
         converged = as.logical(out$converged),
         iterations = as.integer(out$converged_iter),
         final_gap = as.numeric(out$final_gap),
         delta_n = as.numeric(out$delta_n),
         initial_loss = selected_initial_loss,
         optimization_initial_loss = as.numeric(out$initial_loss),
         q_constant = as.numeric(out$q_constant),
         calibration_L = calibration_L,
         proportion_to_initial_loss = if (adaptive_lambda) {
           control$proportion_to_initial_loss
         } else NA_real_,
         uniform_objective = as.numeric(out$uniform_objective),
         logistic_objective = as.numeric(out$logistic_objective),
         reference_objective = as.numeric(out$reference_objective),
         initial_source = as.character(out$initial_source),
         logistic_fit_converged = initializer$logistic_fit_converged,
         logistic_fallback_reason = initializer$fallback_reason,
         S_n = as.numeric(out$S_n),
         eta_initial = as.numeric(out$eta_initial),
         eta_final = as.numeric(out$eta_final),
         eta_increases = as.integer(out$eta_increases),
         eta_decreases = as.integer(out$eta_decreases),
         eta_history = as.numeric(out$eta_history),
         eta_progress_history = as.numeric(out$eta_progress_history),
         objective_history = as.numeric(out$objective_history),
         lambda = as.numeric(out$lambda), L = L_n, n = n,
         tuning_n = tuning_n, p = 1L,
         optimizer = if (p_order == 2L) "projected gradient" else
           "projected subgradient",
         optim_convergence = NA_integer_, optim_message = "",
         gradient_norm = NA_real_,
         ## The fit is self-describing: sldb_effect() runs every downstream
         ## estimate and interval off the weights object alone.
         X = X, Z = Z, control = control),
    class = "sldb_weights"
  )
}

#' @export
print.sldb_weights <- function(x, ...) {
  cat("<sldb_weights>\n")
  if (!is.null(x$p)) {
    cat(sprintf("  p = %d,  optimizer = %s\n", x$p, x$optimizer))
  }
  if (!is.null(x$tuning_n) && x$tuning_n != x$n) {
    cat(sprintf("  n = %d,  tuning_n = %d,  lambda = %.6g,  L = %d\n",
                x$n, x$tuning_n, x$lambda, x$L))
  } else {
    cat(sprintf("  n = %d,  lambda = %.6g,  L = %d\n", x$n, x$lambda, x$L))
  }
  cat(sprintf("  converged = %s after %d iterations\n",
              x$converged, x$iterations))
  if (!is.null(x$p) && x$p == 2L) {
    cat(sprintf("  optim convergence = %d,  transformed-gradient norm = %.4e\n",
                x$optim_convergence, x$gradient_norm))
  } else {
    cat(sprintf("  eta: %g -> %g  (%d increases, %d decreases)\n",
                x$eta_initial, x$eta_final,
                x$eta_increases, x$eta_decreases))
  }
  if (is.null(x$p) || x$p != 2L) {
    cat(sprintf("  final gap = %.4e  (target delta_n = %.4e)\n",
                x$final_gap, x$delta_n))
  }
  if (!is.null(x$uniform_objective) && is.finite(x$uniform_objective)) {
    if (!is.null(x$initial_loss) && is.finite(x$initial_loss)) {
      cat(sprintf("  Lhat(w_constant) = %.6g,  q_constant = %.6g\n",
                  x$initial_loss, x$q_constant))
    }
    cat(sprintf("  Jhat(w_uniform) = %.6g\n", x$uniform_objective))
    if (!is.null(x$logistic_objective) &&
        is.finite(x$logistic_objective)) {
      cat(sprintf("  Jhat(w_IPW) = %.6g\n", x$logistic_objective))
    }
    if (!is.null(x$reference_objective) &&
        is.finite(x$reference_objective)) {
      cat(sprintf("  J_ref = %.6g,  selected start = %s\n",
                  x$reference_objective, x$initial_source))
    }
  }
  invisible(x)
}


## Estimation and asymptotic variance
## Write a_i = n * w_i * (2 Z_i - 1).  Because the weights sum to one within
## each arm, sum(a) = 0 exactly, which is what makes the centering below both
## estimate-preserving and available in closed form.

#' @keywords internal
#' @noRd
.sldb_contrast <- function(weights, Z, n) n * weights * (2 * Z - 1)

#' @keywords internal
#' @noRd
.sldb_reuse_weights <- function(weights, X, Z, control, n, tuning_n = NULL,
                                delta = NULL) {
  if (is.null(weights)) {
    return(sldb_weights(X, Z, control, tuning_n = tuning_n, delta = delta))
  }
  if (!inherits(weights, "sldb_weights")) {
    stop("`weights` must be an sldb_weights object from sldb_weights().",
         call. = FALSE)
  }
  if (length(weights$weights) != n) {
    stop("`weights` was fitted on ", length(weights$weights),
         " observations but the data have ", n,
         ". Refit rather than reusing across samples.", call. = FALSE)
  }
  weights
}

#' Variance-minimising centering constant for the ATE
#'
#' The influence-function family
#' \deqn{\psi_i(c) = a_i (Y_i - c) - \tau, \qquad a_i = n w_i (2 Z_i - 1)}
#' has the same mean (zero) for every \eqn{c}, because \eqn{\sum_i a_i = 0}.
#' The estimate is therefore untouched by \eqn{c} and the constant is free to be
#' chosen to minimise \eqn{\mathrm{mean}(\psi_i(c)^2)}, giving
#' \deqn{c^\star = \frac{\sum_i a_i^2 Y_i}{\sum_i a_i^2}.}
#'
#' @param a Contrast vector \eqn{a_i = n w_i (2 Z_i - 1)}.
#' @param Y Numeric outcome vector.
#'
#' @return A single numeric.
#' @export
sldb_center_ate <- function(a, Y) sum(a^2 * Y) / sum(a^2)

#' Variance-minimising centering constant for the LATE
#'
#' With \eqn{a_i = n w_i (2 Z_i - 1)} and the Wald-ratio estimate
#' \eqn{\hat\theta}, the influence-function family is
#' \deqn{\psi_i(c) = a_i (Y_i - c) - \hat\theta\, a_i (A_i - 1),}
#' whose mean is exactly zero for every \eqn{c}. Minimising
#' \eqn{\mathrm{mean}(\psi_i(c)^2)} gives
#' \deqn{c^\star = \frac{\sum_i a_i^2 \{Y_i - \hat\theta (A_i - 1)\}}{\sum_i a_i^2}.}
#'
#' @param a Contrast vector \eqn{a_i = n w_i (2 Z_i - 1)}.
#' @param Y Numeric outcome vector.
#' @param A Numeric treatment-received vector.
#' @param late The Wald-ratio point estimate \eqn{\hat\theta}.
#'
#' @return A single numeric.
#' @export
sldb_center_late <- function(a, Y, A, late) {
  sum(a^2 * (Y - late * (A - 1))) / sum(a^2)
}

#' SLDB estimate of the average treatment effect
#'
#' @param X Numeric covariate matrix, `n` by `p`.
#' @param Z Integer treatment vector, coded 0/1.
#' @param Y Numeric outcome vector.
#' @param control An [sldb_control()] object.
#' @param weights Optionally, a pre-computed [sldb_weights()] object, to avoid
#'   re-solving the balancing problem.
#' @param tuning_n Optional reference sample size for the tuning schedule;
#'   passed to [sldb_weights()] when `weights` is not supplied.
#' @param delta Optional fixed certified-gap tolerance passed to
#'   [sldb_weights()] when `weights` is not supplied.
#'
#' @return An object of class `"sldb_ate"` (inheriting `"sldb_fit"`) with
#'   elements `estimate`, `se`, `se_min`, `center`, `weights`, `fit` and `n`.
#'   `se` uses \eqn{c = 0}; `se_min` uses the variance-minimising \eqn{c^\star}
#'   from [sldb_center_ate()]. Both are consistent for the same asymptotic
#'   variance bound, but `se_min` is the sharper finite-sample estimator and is
#'   the one the Wald interval should normally be built from.
#' @export
sldb_ate <- function(X, Z, Y, control = sldb_control(), weights = NULL,
                     tuning_n = NULL, delta = NULL) {
  d <- .sldb_check_data(X, Z)
  Z <- d$Z
  n <- length(Z)
  Y <- as.numeric(Y)
  if (length(Y) != n) stop("Y has the wrong length.", call. = FALSE)

  weights <- .sldb_reuse_weights(weights, d$X, Z, control, n, tuning_n, delta)
  w <- weights$weights
  a <- .sldb_contrast(w, Z, n)

  est    <- sum(w * (2 * Z - 1) * Y)
  se     <- sqrt(mean((a * Y - est)^2) / n)
  center <- sldb_center_ate(a, Y)
  se_min <- sqrt(mean((a * (Y - center) - est)^2) / n)

  structure(
    list(estimand = "ATE", estimate = est, se = se, se_min = se_min,
         center = center, weights = w, fit = weights, n = n),
    class = c("sldb_ate", "sldb_fit")
  )
}

#' SLDB estimate of the local average treatment effect
#'
#' Wald-ratio (instrumental-variable) estimate in which the balancing weights
#' are built from the instrument `Z`:
#' \deqn{\hat\theta = \frac{\sum_i w_i (2 Z_i - 1) Y_i}{\sum_i w_i (2 Z_i - 1) A_i}.}
#'
#' @param X Numeric covariate matrix, `n` by `p`.
#' @param Z Integer instrument vector, coded 0/1.
#' @param Y Numeric outcome vector.
#' @param A Numeric treatment-received vector.
#' @param control An [sldb_control()] object.
#' @param weights Optionally, a pre-computed [sldb_weights()] object.
#' @param tuning_n Optional reference sample size for the tuning schedule;
#'   passed to [sldb_weights()] when `weights` is not supplied.
#' @param delta Optional fixed certified-gap tolerance passed to
#'   [sldb_weights()] when `weights` is not supplied.
#'
#' @return An object of class `"sldb_late"` (inheriting `"sldb_fit"`) with
#'   elements `estimate`, `se`, `se_min`, `center`, `numerator`, `denominator`,
#'   `weights`, `fit` and `n`. `se` uses \eqn{c = \hat\theta}, which reproduces
#'   the usual IV influence function \eqn{a_i (Y_i - \hat\theta A_i)}; `se_min`
#'   uses the variance-minimising \eqn{c^\star} from [sldb_center_late()].
#' @export
sldb_late <- function(X, Z, Y, A, control = sldb_control(), weights = NULL,
                      tuning_n = NULL, delta = NULL) {
  d <- .sldb_check_data(X, Z)
  Z <- d$Z
  n <- length(Z)
  Y <- as.numeric(Y); A <- as.numeric(A)
  if (length(Y) != n || length(A) != n) {
    stop("Y and A must have length nrow(X).", call. = FALSE)
  }

  weights <- .sldb_reuse_weights(weights, d$X, Z, control, n, tuning_n, delta)
  w <- weights$weights
  a <- .sldb_contrast(w, Z, n)

  num <- mean(a * Y)
  den <- mean(a * A)
  if (!is.finite(den) || abs(den) < .Machine$double.eps^0.5) {
    stop("The weighted first stage is numerically zero; the LATE is not ",
         "identified in this sample.", call. = FALSE)
  }
  est <- num / den

  ## psi_i(c) = a_i (Y_i - c) - est * a_i (A_i - 1); mean(psi) = 0 for any c.
  psi <- function(cc) a * (Y - cc) - est * a * (A - 1)
  se     <- sqrt(mean(psi(est)^2) / n) / abs(den)
  center <- sldb_center_late(a, Y, A, est)
  se_min <- sqrt(mean(psi(center)^2) / n) / abs(den)

  structure(
    list(estimand = "LATE", estimate = est, se = se, se_min = se_min,
         center = center, numerator = num, denominator = den,
         weights = w, fit = weights, n = n),
    class = c("sldb_late", "sldb_fit")
  )
}

#' @export
print.sldb_fit <- function(x, ...) {
  cat(sprintf("<sldb %s>  n = %d\n", x$estimand, x$n))
  cat(sprintf("  estimate = %.6g\n", x$estimate))
  cat(sprintf("  se       = %.6g   (reference centering)\n", x$se))
  cat(sprintf("  se_min   = %.6g   (c* = %.6g)\n", x$se_min, x$center))
  ci <- sldb_ci_wald(x)
  cat(sprintf("  95%% Wald  = [%.6g, %.6g]\n", ci[1L], ci[2L]))
  invisible(x)
}


## Inference

#' Wald confidence interval for an SLDB fit
#'
#' @param fit An `"sldb_fit"` object from [sldb_ate()] or [sldb_late()].
#' @param level Confidence level.
#' @param centered If `TRUE` (the default) use the variance-minimising
#'   `se_min`; if `FALSE` use the reference `se`.
#'
#' @return A length-2 numeric vector, `c(lower, upper)`.
#' @export
sldb_ci_wald <- function(fit, level = 0.95, centered = TRUE) {
  stopifnot(inherits(fit, "sldb_fit"))
  se <- if (centered) fit$se_min else fit$se
  z  <- stats::qnorm(1 - (1 - level) / 2)
  fit$estimate + c(-1, 1) * z * se
}

#' Percentile bootstrap confidence interval
#'
#' Generic in the estimator, in the same way as [sldb_moonboot_ci()].
#'
#' @param estimator A function of an integer index vector returning a scalar.
#' @param n Full sample size.
#' @param R Number of bootstrap resamples.
#' @param level Confidence level.
#'
#' @return A list with `ci` (length-2 numeric), `replicates` and `n_valid`.
#' @export
sldb_boot_ci <- function(estimator, n, R = 1000L, level = 0.95) {
  stopifnot(is.function(estimator))
  alpha <- 1 - level
  vals <- rep(NA_real_, R)
  for (r in seq_len(R)) {
    idx <- sample.int(n, n, replace = TRUE)
    vals[r] <- tryCatch(as.numeric(estimator(idx)), error = function(e) NA_real_)
  }
  ok <- vals[is.finite(vals)]
  ci <- if (length(ok) < 2L) c(NA_real_, NA_real_) else {
    stats::quantile(ok, c(alpha / 2, 1 - alpha / 2), names = FALSE)
  }
  list(ci = ci, replicates = vals, n_valid = length(ok))
}

#' Subsample sizes for the rate estimate
#'
#' A geometric ladder centred on \eqn{\sqrt{n}}, spanning `span` either way:
#' \eqn{m_j = \sqrt{n}\,\mathrm{span}^{\,j}} for `n_grid` values of \eqn{j}
#' equally spaced on \eqn{[-1, 1]}. The defaults give
#' \eqn{\sqrt{n}/2,\ \sqrt{n}/\sqrt2,\ \sqrt{n},\ \sqrt2\sqrt{n},\ 2\sqrt{n}}:
#' 22-89 at \eqn{n = 2000}, 32-126 at \eqn{n = 4000} and 50-199 at
#' \eqn{n = 9915}.
#'
#' \eqn{\sqrt{n}} is the anchor because it is the scale the subsampling
#' interval actually runs at -- it sits inside the Bickel-Sakov candidate
#' range from [sldb_m_grid_bickel()] at every sample size used here -- so the
#' fitted exponent is a local fit over the sizes it will be applied to rather
#' than an extrapolation from sizes the procedure never visits. The ladder is
#' geometric because the fit regresses a log spread on \eqn{\log m}, which
#' weights every rung equally only when the rungs are equally spaced in logs.
#'
#' @param n Full sample size.
#' @param n_grid Number of rungs.
#' @param span Multiplicative half-width: the ladder runs from
#'   \eqn{\sqrt{n}/\mathrm{span}} to \eqn{\mathrm{span}\sqrt{n}}.
#' @param min_m Floor on the smallest rung.
#'
#' @return An integer vector of sizes, ascending.
#' @export
sldb_tau_grid <- function(n, n_grid = 5L, span = 2, min_m = 20L) {
  if (!is.numeric(n) || length(n) != 1L || !is.finite(n) || n < 1) {
    stop("`n` must be one positive finite sample size.", call. = FALSE)
  }
  if (!is.numeric(span) || length(span) != 1L || !is.finite(span) ||
      span <= 1) {
    stop("`span` must be a finite multiplier greater than one.", call. = FALSE)
  }
  if (n_grid < 2L) stop("`n_grid` needs at least two rungs.", call. = FALSE)
  anchor <- sqrt(n)
  sizes  <- anchor * span^seq(-1, 1, length.out = n_grid)
  sizes  <- pmin(n, pmax(min_m, round(sizes)))
  sizes  <- unique(as.integer(sizes))
  if (length(sizes) < 2L) {
    stop("`n` is too small for the requested tau ladder.", call. = FALSE)
  }
  sizes
}

#' Estimate the convergence rate tau from the data
#'
#' Subsampling inverts \eqn{\tau_m(\hat\theta_m - \hat\theta_n)}, so it needs
#' the rate \eqn{\tau_m = m^{a}} at which \eqn{\hat\theta_m} concentrates. This
#' fits \eqn{a} rather than asserting it.
#'
#' Draw `R` subsamples at each size on the [sldb_tau_grid()] ladder and measure
#' the spread of the replicates by symmetric interquantile ranges,
#' \deqn{D_p(m) = q_{1-p}(\hat\theta_m) - q_p(\hat\theta_m),\quad p \in
#'   \mathtt{probs}.}
#' Under \eqn{\hat\theta_m - \theta = O_p(m^{-a})} every \eqn{D_p(m)} is
#' proportional to \eqn{m^{-a}}, so
#' \deqn{y(m) = \frac{1}{|\mathtt{probs}|}\sum_p \log D_p(m) = c - a \log m,}
#' and the least-squares slope of \eqn{y(m)} on \eqn{\log m} estimates
#' \eqn{-a}. The returned rate is \eqn{\tau(x) = x^{\hat a}}.
#'
#' Interquantile ranges rather than the variance: a weighting estimator on a
#' small subsample can produce a near-zero weighted first stage, and the
#' resulting outlying replicate moves a variance far more than it moves a
#' 5th-to-95th percentile range. Averaging several `probs` in logs uses more of
#' the distribution than any single pair.
#'
#' @param statistic A `boot`-style `function(data, indices)`.
#' @param n Full sample size.
#' @param m_grid Ladder of sizes. Defaults to [sldb_tau_grid()] at `n`.
#' @param R Subsamples drawn at each rung.
#' @param probs Lower tail probabilities defining the interquantile ranges.
#' @param replace Passed to [moonboot::mboot()].
#' @param min_valid Minimum usable replicates for a rung to enter the fit.
#' @param exponent_range Admissible range for \eqn{\hat a}; a fit outside it is
#'   clamped, so the exponent used is
#'   \eqn{\mathrm{median}(0.45, \hat a, 0.5)} at the default. The upper end
#'   caps the rate at the parametric \eqn{\sqrt{m}}: the subsampling
#'   half-width is \eqn{(m/n)^{a} D(m)} and \eqn{m/n < 1}, so a larger
#'   \eqn{a} shortens the interval, and \eqn{a > 1/2} would report an interval
#'   shorter than the root-\eqn{n} one. The lower end bounds how far the
#'   fitted rate may widen it: at \eqn{a = 0.45} the half-width carries a
#'   factor \eqn{(n/m)^{0.05}} over the root-\eqn{n} one.
#' @param n_grid,span Passed to [sldb_tau_grid()] when `m_grid` is `NULL`.
#'
#' @return A list with `tau` (the rate function), `exponent` (\eqn{\hat a} as
#'   used), `exponent_raw` (before clamping), `clamped` (whether the two
#'   differ), `grid`, `log_spread`, `n_valid` and `r_squared`.
#' @export
sldb_estimate_tau <- function(statistic, n, m_grid = NULL, R = 500L,
                              probs = c(0.05, 0.10, 0.25), replace = FALSE,
                              min_valid = 30L, exponent_range = c(0.45, 0.5),
                              n_grid = 5L, span = 2) {
  if (!requireNamespace("moonboot", quietly = TRUE)) {
    stop("Package 'moonboot' is required by sldb_estimate_tau().", call. = FALSE)
  }
  if (is.null(m_grid)) {
    m_grid <- sldb_tau_grid(n, n_grid = n_grid, span = span)
  }
  probs <- sort(unique(probs))
  if (any(probs <= 0) || any(probs >= 0.5)) {
    stop("`probs` must be lower tail probabilities in (0, 0.5).", call. = FALSE)
  }

  G <- length(m_grid)
  y <- rep(NA_real_, G)
  n_valid <- integer(G)
  for (g in seq_len(G)) {
    v <- moonboot::mboot(seq_len(n), statistic, m = m_grid[g], R = R,
                         replace = replace)$t
    v <- v[is.finite(v)]
    n_valid[g] <- length(v)
    if (length(v) < min_valid) next
    d <- stats::quantile(v, 1 - probs, names = FALSE) -
         stats::quantile(v, probs, names = FALSE)
    d <- d[is.finite(d) & d > 0]
    if (length(d)) y[g] <- mean(log(d))
  }

  ok <- is.finite(y)
  fallback <- function(msg) {
    warning(msg, " Falling back to tau(x) = sqrt(x).", call. = FALSE)
    list(tau = sqrt, exponent = 0.5, exponent_raw = NA_real_, clamped = NA,
         grid = m_grid, log_spread = y, n_valid = n_valid,
         r_squared = NA_real_)
  }
  if (sum(ok) < 2L) {
    return(fallback("Fewer than two usable sizes in the tau ladder."))
  }

  fit <- stats::lm(y[ok] ~ log(m_grid[ok]))
  a_raw <- -unname(stats::coef(fit)[2L])
  if (!is.finite(a_raw)) return(fallback("The tau fit did not return a slope."))

  ## Both ends are design bounds, not diagnostics: the rate used is
  ## median(exponent_range[1], a_raw, exponent_range[2]). The cap keeps the
  ## interval from coming out shorter than the root-n one; the floor caps how
  ## far a noisy fit can widen it. Hitting either is expected and is recorded
  ## in `clamped` rather than warned about.
  a <- min(max(a_raw, exponent_range[1L]), exponent_range[2L])
  list(tau = function(x) x^a, exponent = a, exponent_raw = a_raw,
       clamped = !isTRUE(all.equal(a, a_raw)),
       grid = m_grid, log_spread = y, n_valid = n_valid,
       r_squared = unname(summary(fit)$r.squared))
}

#' Candidate subsample sizes for the Bickel-Sakov search
#'
#' `n_grid` sizes spaced evenly on the log scale from
#' \eqn{\max(20, n^{a})} to \eqn{n^{b}}, with \eqn{a} and \eqn{b} given by
#' `range_exp`. The defaults \eqn{a = 1/3}, \eqn{b = 2/3} keep the whole search
#' inside \eqn{m/n \le n^{-1/3}}: 0.079 at \eqn{n = 2000}, 0.063 at
#' \eqn{n = 4000} and 0.047 at \eqn{n = 9915}. That is well below the
#' \eqn{0.25n} (at `q = 0.5`) or \eqn{0.5625n} (at `q = 0.75`) top of the
#' geometric grid [moonboot::estimate.m()] builds for itself, and it sends
#' \eqn{m/n \to 0} at the \eqn{n^{-1/3}} rate while still leaving
#' \eqn{m \to \infty}. The grids are 20-159 at \eqn{n = 2000}, 20-252 at
#' \eqn{n = 4000} and 21-462 at \eqn{n = 9915}.
#'
#' Log spacing is what the Bickel-Sakov rule expects: it compares
#' *consecutive* candidates, so a constant ratio between neighbours gives every
#' comparison the same meaning. The lower floor of 20 keeps both treatment arms
#' non-empty in most draws; \eqn{n^{1/3}} only exceeds it above \eqn{n = 8000},
#' so the floor binds at \eqn{n \le 4000} and the schedule binds at
#' \eqn{n = 9915}.
#'
#' @param n Full sample size.
#' @param n_grid Number of candidate sizes.
#' @param range_exp Length-2 numeric, the lower and upper exponents.
#' @param lower_min Floor on the smallest candidate.
#'
#' @return An integer vector of candidate subsample sizes, ascending.
#' @export
sldb_m_grid_bickel <- function(n, n_grid = 20L, range_exp = c(1 / 3, 2 / 3),
                               lower_min = 20) {
  if (!is.numeric(n) || length(n) != 1L || !is.finite(n) || n < 1) {
    stop("`n` must be one positive finite sample size.", call. = FALSE)
  }
  if (!is.numeric(range_exp) || length(range_exp) != 2L ||
      !all(is.finite(range_exp)) || range_exp[1L] >= range_exp[2L]) {
    stop("`range_exp` must be two increasing finite exponents.", call. = FALSE)
  }
  lower <- max(lower_min, n^range_exp[1L])
  upper <- n^range_exp[2L]
  if (upper <= lower) {
    stop("`n` is too small for the requested Bickel grid.", call. = FALSE)
  }
  unique(round(exp(seq(log(lower), log(upper), length.out = n_grid))))
}

#' Bickel-Sakov subsample size over a supplied grid
#'
#' The Bickel and Sakov (2008) rule: standardise the subsampling distribution
#' at each candidate size as \eqn{\tau_m(\hat\theta_m - \hat\theta_n)}, take the
#' Kolmogorov-Smirnov distance between *consecutive* candidates, and keep the
#' size where that distance is smallest. Ties go to the smallest \eqn{m}, as in
#' [moonboot::estimate.m()].
#'
#' This differs from [moonboot::estimate.m()] in two ways. The candidate sizes
#' are supplied rather than fixed to \eqn{\lceil q^j n\rceil}, so the search can
#' be confined to a range where \eqn{m/n} stays small. And each candidate is
#' resampled once and reused by both of its neighbouring comparisons;
#' `moonboot` resamples every interior size twice, so this costs
#' `length(m_grid)` runs where `moonboot` costs `2 (length(m_grid) - 1)`.
#'
#' @param statistic A `boot`-style `function(data, indices)`.
#' @param n Full sample size.
#' @param m_grid Ascending candidate sizes, e.g. from [sldb_m_grid_bickel()].
#' @param tau Rate function, e.g. `sqrt`.
#' @param R Subsamples drawn per candidate.
#' @param replace Passed to [moonboot::mboot()].
#' @param min_valid Minimum usable replicates for a candidate to be compared.
#'
#' @return A list with `m`, `grid`, `distances` and `n_valid`.
#' @export
sldb_bickel_m <- function(statistic, n, m_grid, tau, R = 200L,
                          replace = FALSE, min_valid = 10L) {
  G <- length(m_grid)
  if (G < 2L) stop("`m_grid` needs at least two candidates.", call. = FALSE)
  boots <- lapply(m_grid, function(m) {
    moonboot::mboot(seq_len(n), statistic, m = m, R = R, replace = replace)
  })
  t0 <- boots[[1L]]$t0
  n_valid <- integer(G)
  std <- vector("list", G)
  for (g in seq_len(G)) {
    v <- boots[[g]]$t
    v <- v[is.finite(v)]
    n_valid[g] <- length(v)
    if (length(v) >= min_valid && is.finite(t0)) {
      std[[g]] <- tau(m_grid[g]) * (v - t0)
    }
  }
  distances <- rep(NA_real_, G - 1L)
  for (g in seq_len(G - 1L)) {
    a <- std[[g]]; b <- std[[g + 1L]]
    if (is.null(a) || is.null(b)) next
    pooled <- c(a, b)
    distances[g] <- max(abs(stats::ecdf(a)(pooled) - stats::ecdf(b)(pooled)))
  }
  if (all(is.na(distances))) {
    return(list(m = NA_integer_, grid = m_grid, distances = distances,
                n_valid = n_valid))
  }
  ## Ascending grid, and which.min returns the first minimiser, so ties give
  ## the smallest m -- the same tie rule moonboot applies to its descending one.
  best <- which.min(distances)
  list(m = as.integer(m_grid[best]), grid = m_grid, distances = distances,
       n_valid = n_valid)
}

#' Subsampling interval via the moonboot package
#'
#' Wraps [moonboot::mboot()], [moonboot::mboot.ci()] and
#' [moonboot::estimate.m()] behind the same estimator interface used by
#' [sldb_boot_ci()], so the same call serves SLDB and every competing method.
#' This is the subsampling routine both drivers use, and it runs in three
#' stages, each feeding the next:
#'
#' 1. the rate \eqn{\tau(x) = x^{\hat a}} is fitted by [sldb_estimate_tau()]
#'    over the \eqn{\sqrt{n}} ladder;
#' 2. that \eqn{\tau} standardises the candidate subsampling distributions in
#'    the Bickel-Sakov search of [sldb_bickel_m()], which returns \eqn{m};
#' 3. the same \eqn{\tau} and that \eqn{m} build the interval.
#'
#' Fitting \eqn{\hat a} rather than asserting \eqn{1/2} is what makes the
#' reported length insensitive to \eqn{m}. The half-width is
#' \eqn{(m/n)^{a} D(m)} for a spread \eqn{D(m) \propto m^{-a_0}} at the true
#' rate \eqn{a_0}, so it varies with \eqn{m} as \eqn{m^{a - a_0}} and is flat
#' in \eqn{m} only when \eqn{a = a_0}. Under a fixed \eqn{a = 1/2} any gap
#' between \eqn{a_0} and \eqn{1/2} is charged to the interval at the rate
#' \eqn{(m/n)^{1/2 - a_0}}.
#'
#' `moonboot` expects a `boot`-style `statistic(data, indices)`, whereas the
#' estimators here are already closures over the data. The wrapper therefore
#' passes the row indices as `data` and ignores it inside `statistic`, and
#' caches the full-sample value so the repeated `t0` evaluation that every
#' [moonboot::mboot()] call performs costs one fit rather than dozens.
#'
#' The `"basic"` interval it returns inverts the quantiles of
#' \eqn{\tau_m(\hat\theta_m - \hat\theta_n)} and rescales by
#' \eqn{\tau_m/\tau_n}. It applies no finite-population correction for the
#' overlap between the subsample and the full sample.
#'
#' @param estimator A function of an integer index vector returning a scalar.
#'   May return `NA` for a degenerate subsample; such draws are dropped.
#' @param n Full sample size.
#' @param R Number of subsamples drawn at the selected size.
#' @param level Confidence level.
#' @param tau Rate function. `NULL`, the default, fits it from the data with
#'   [sldb_estimate_tau()]. Pass a function to assert a rate instead, e.g.
#'   `sqrt` for the \eqn{\sqrt{m}} rate the Wald intervals assume.
#' @param m Subsample size. When `NULL` (the default) it is chosen by the rule
#'   named in `m_method`, standardising with the `tau` above.
#' @param m_method Selection rule. `"bickel"` runs [sldb_bickel_m()] over
#'   `m_grid`; `"goetze"`, `"politis"` and `"sherman"` are passed through to
#'   [moonboot::estimate.m()], which uses its own \eqn{\lceil q^j n\rceil}
#'   grid instead.
#' @param m_R Number of resamples drawn at *each* candidate size while
#'   selecting `m`. This is where nearly all the cost sits: the search spends
#'   `n_grid * m_R` refits against the `R` refits of the interval itself, so
#'   the default 200 is deliberately below `R`.
#' @param tau_R Number of resamples drawn at *each* rung of the
#'   [sldb_tau_grid()] ladder while fitting `tau`. Ignored when `tau` is given.
#' @param tau_n_grid,tau_span,tau_probs Passed to [sldb_estimate_tau()].
#' @param tau_exponent_range Admissible range for the fitted exponent, passed
#'   to [sldb_estimate_tau()] as `exponent_range`. The default bounds it to
#'   \eqn{[0.45, 0.5]}: the cap keeps the interval from coming out shorter
#'   than the root-\eqn{n} one, and the floor bounds how far a noisy fit may
#'   widen it. Ignored when `tau` is given.
#' @param min_m Smallest admissible subsample size, and the `lower_min` floor
#'   of the Bickel-Sakov grid. Sizes much below 20 produce many degenerate
#'   draws for weighting estimators, since both treatment arms must be
#'   non-empty.
#' @param m_grid Candidate sizes for `m_method = "bickel"`. Defaults to
#'   [sldb_m_grid_bickel()] evaluated with `n_grid` and `range_exp`.
#' @param n_grid,range_exp Passed to [sldb_m_grid_bickel()] when `m_grid` is
#'   `NULL`.
#' @param params Extra parameters for the `moonboot` selection rules, e.g.
#'   `list(q = 0.5)`. Ignored by `m_method = "bickel"`, which takes its
#'   candidates from `m_grid`.
#' @param type Interval type from [moonboot::mboot.ci()]: `"basic"`, `"norm"`
#'   or `"sherman"`.
#'
#' @return A list with `ci` (length-2 numeric), `m`, `tau_exponent`,
#'   `n_valid`, `replicates`, `tau_fit` (the [sldb_estimate_tau()] output when
#'   the rate was fitted) and `search` (the [sldb_bickel_m()] output when that
#'   rule was used).
#' @export
sldb_moonboot_ci <- function(estimator, n, R = 1000L, level = 0.95,
                             tau = NULL, m = NULL, m_method = "bickel",
                             m_R = 200L, tau_R = 200L, min_m = 20L,
                             m_grid = NULL, n_grid = 20L,
                             range_exp = c(1 / 3, 2 / 3),
                             tau_n_grid = 5L, tau_span = 2,
                             tau_probs = c(0.05, 0.10, 0.25),
                             tau_exponent_range = c(0.45, 0.5),
                             params = NULL, type = "basic") {
  stopifnot(is.function(estimator))
  if (!requireNamespace("moonboot", quietly = TRUE)) {
    stop("Package 'moonboot' is required by sldb_moonboot_ci().", call. = FALSE)
  }
  type <- match.arg(type, c("basic", "norm", "sherman"))

  ## moonboot calls statistic(data, 1:n) inside every mboot(); cache it.
  full_value <- NULL
  statistic <- function(data, indices, ...) {
    if (length(indices) == n) {
      if (is.null(full_value)) {
        full_value <<- tryCatch(as.numeric(estimator(indices)),
                                error = function(e) NA_real_)
      }
      return(full_value)
    }
    tryCatch(as.numeric(estimator(indices)), error = function(e) NA_real_)
  }
  idx_data <- seq_len(n)

  ## Stage 1: the rate. Fitted over the sqrt(n) ladder unless asserted.
  tau_fit <- NULL
  if (is.null(tau)) {
    tau_fit <- sldb_estimate_tau(statistic, n, R = tau_R, probs = tau_probs,
                                 replace = FALSE, n_grid = tau_n_grid,
                                 span = tau_span,
                                 exponent_range = tau_exponent_range)
    tau <- tau_fit$tau
  }
  ## tau(x) = x^a  =>  log tau(e) = a. Read back off `tau` rather than taken
  ## from the fit, so it describes an asserted rate as well as a fitted one.
  tau_exponent <- log(tau(exp(1)))

  m_search <- NULL
  if (is.null(m)) {
    if (identical(m_method, "bickel")) {
      ## Stage 2: Bickel-Sakov over our own grid rather than moonboot's q^j n
      ## sequence, standardising with the tau fitted above.
      if (is.null(m_grid)) {
        m_grid <- sldb_m_grid_bickel(n, n_grid = n_grid, range_exp = range_exp,
                                     lower_min = min_m)
      }
      m_search <- sldb_bickel_m(statistic, n, m_grid = m_grid, tau = tau,
                                R = m_R)
      m <- m_search$m
    } else {
      m <- moonboot::estimate.m(idx_data, statistic, tau = tau, R = m_R,
                                replace = FALSE, min.m = min_m,
                                method = m_method, params = params)
    }
  }
  if (!is.finite(m)) {
    return(list(ci = c(NA_real_, NA_real_), m = NA_integer_,
                tau_exponent = tau_exponent, n_valid = 0L,
                replicates = numeric(), tau_fit = tau_fit, search = m_search))
  }
  m <- as.integer(m)

  ## Stage 3: the interval, at that m and under the same tau.
  boot_out <- moonboot::mboot(idx_data, statistic, m = m, R = R,
                              replace = FALSE)
  ## mboot.ci has no na.rm; drop degenerate draws before inverting.
  keep <- is.finite(boot_out$t)
  n_valid <- sum(keep)
  if (n_valid < 2L || !is.finite(boot_out$t0)) {
    return(list(ci = c(NA_real_, NA_real_), m = m, tau_exponent = tau_exponent,
                n_valid = n_valid, replicates = boot_out$t, tau_fit = tau_fit,
                search = m_search))
  }
  replicates <- boot_out$t
  boot_out$t <- boot_out$t[keep]

  ci <- tryCatch(
    moonboot::mboot.ci(boot_out, conf = level, tau = tau, types = type)[[type]],
    error = function(e) c(NA_real_, NA_real_))

  list(ci = as.numeric(ci), m = m, tau_exponent = tau_exponent,
       n_valid = n_valid, replicates = replicates, tau_fit = tau_fit,
       search = m_search)
}

#' SLDB point estimator as an index-indexed closure
#'
#' Builds the `estimator` argument that [sldb_moonboot_ci()] and
#' [sldb_boot_ci()] expect. The returned closure holds \eqn{\lambda_N} fixed at
#' `tuning_n`, even when its index vector contains fewer rows. By default,
#' `tuning_n` is the size of the original data supplied when the closure is
#' created. For subsampling, `refit_L = TRUE` gives
#' \eqn{L_m=\lceil m^{1+1/(4d)}\rceil}, while
#' `refit_delta = TRUE` gives \eqn{\delta_m=c_{\delta}\lambda_N/m} under
#' the coefficient schedule.
#'
#' @param X Numeric covariate matrix.
#' @param Z Integer instrument/treatment vector, coded 0/1.
#' @param Y Numeric outcome vector.
#' @param A Optional treatment-received vector. When supplied, the closure
#'   returns the LATE; otherwise the ATE.
#' @param control An [sldb_control()] object.
#' @param tuning_n Reference sample size used for `lambda` and `delta` in every
#'   refit, and for `L` unless `refit_L = TRUE`. Defaults to the original number
#'   of rows in `X`.
#' @param lambda Fixed full-sample ridge used in every indexed refit. Required
#'   when `control$proportion_to_initial_loss` is active.
#' @param delta Fixed certified-gap tolerance used in every refit. It is used
#'   only when `refit_delta = FALSE`, which freezes the full-sample value:
#'   pass the full-sample weights fit's `delta_n` there. With
#'   `refit_delta = TRUE` each refit derives its own tolerance and this
#'   argument is ignored, so it may be `NULL`. Ignored when `p = 2`.
#' @param refit_L If `TRUE`, each indexed refit of size \eqn{m} uses
#'   \eqn{L_m=\lceil m^{1+1/(4d)}\rceil} while retaining the full-sample ridge.
#'   Use this for subsampling, not bootstrap. Default `FALSE`.
#' @param refit_delta If `TRUE`, an indexed refit of size \eqn{m} derives its
#'   own certified-gap tolerance. Under the coefficient schedule that is
#'   \eqn{\delta_m=c_{\delta}\lambda_N/m}, without changing
#'   \eqn{c_{\delta}}. Under the reference-objective schedule it is
#'   \eqn{\delta_m=\phi\,J_{\mathrm{ref}}(\text{subsample})}, recomputed by
#'   the engine from the subsample's own reference objective at the frozen
#'   ridge, so the tolerance keeps the same meaning at every size. Use this for
#'   subsampling, not bootstrap. Default `FALSE`.
#'
#' @return A function of an integer index vector returning a scalar estimate.
#' @export
sldb_estimator <- function(X, Z, Y, A = NULL, control = sldb_control(),
                           tuning_n = NULL, lambda = NULL, delta = NULL,
                           refit_L = FALSE, refit_delta = FALSE) {
  d <- .sldb_check_data(X, Z)
  X <- d$X; Z <- d$Z
  Y <- as.numeric(Y)
  if (!is.null(A)) A <- as.numeric(A)
  if (is.null(tuning_n)) tuning_n <- length(Z)
  if (!is.numeric(tuning_n) || length(tuning_n) != 1L ||
      !is.finite(tuning_n) || tuning_n < 1 || tuning_n != as.integer(tuning_n)) {
    stop("`tuning_n` must be one positive integer.", call. = FALSE)
  }
  tuning_n <- as.integer(tuning_n)
  p_order <- if (is.null(control$p)) 1L else as.integer(control$p)
  uses_delta <- p_order == 1L
  if (!uses_delta) delta <- NULL
  adaptive_lambda <- !is.null(control$proportion_to_initial_loss)
  if (adaptive_lambda && is.null(lambda)) {
    stop("The initial-loss ridge must be fixed before resampling. Pass ",
         "`lambda = full_sample_weights$lambda`.", call. = FALSE)
  }
  if (!is.null(lambda) &&
      (!is.numeric(lambda) || length(lambda) != 1L ||
       !is.finite(lambda) || lambda <= 0)) {
    stop("`lambda` must be one positive finite number.", call. = FALSE)
  }
  if (!is.logical(refit_L) || length(refit_L) != 1L || is.na(refit_L)) {
    stop("`refit_L` must be TRUE or FALSE.", call. = FALSE)
  }
  if (!is.logical(refit_delta) || length(refit_delta) != 1L ||
      is.na(refit_delta)) {
    stop("`refit_delta` must be TRUE or FALSE.", call. = FALSE)
  }
  ## With `refit_delta = TRUE` the reference-objective tolerance is rebuilt
  ## inside each refit, so no full-sample value is needed. Freezing it
  ## (`refit_delta = FALSE`) still requires one.
  if (uses_delta && is.null(delta) && !is.null(control$delta_fraction) &&
      !refit_delta) {
    stop("The reference-objective tolerance must be fixed before resampling. ",
         "Pass `delta = full_sample_weights$delta_n`, or set ",
         "`refit_delta = TRUE` to recompute it from each subsample's own ",
         "reference objective.", call. = FALSE)
  }
  if (uses_delta && !is.null(delta) &&
      (!is.numeric(delta) || length(delta) != 1L ||
       !is.finite(delta) || delta <= 0)) {
    stop("`delta` must be one positive finite number.", call. = FALSE)
  }

  coefficient_delta <- is.null(control$delta_fraction)
  delta_coef <- if (is.null(control$delta_coef)) control$c_opt else control$delta_coef
  lambda_N <- if (is.null(lambda)) sldb_lambda(control, tuning_n) else lambda

  function(idx) {
    Zi <- Z[idx]
    if (sum(Zi == 1L) == 0L || sum(Zi == 0L) == 0L) return(NA_real_)
    Xi <- X[idx, , drop = FALSE]
    refit_control <- control
    if (!is.null(lambda)) {
      refit_control$lambda <- lambda
      refit_control$proportion_to_initial_loss <- NULL
    }
    if (refit_L) {
      refit_control$L <- sldb_L(refit_control, length(idx), d = ncol(X))
    }
    refit_delta_value <- if (!uses_delta) {
      NULL
    } else if (refit_delta && coefficient_delta) {
      delta_coef * lambda_N / length(idx)
    } else if (refit_delta) {
      ## Reference-objective schedule: passing NULL makes sldb_weights() ask
      ## the engine for delta_m = delta_fraction * J_ref(subsample), computed
      ## from the subsample's own reference objective at the frozen ridge.
      NULL
    } else {
      delta
    }
    if (is.null(A)) {
      sldb_ate(Xi, Zi, Y[idx], refit_control, tuning_n = tuning_n,
               delta = refit_delta_value)$estimate
    } else {
      sldb_late(Xi, Zi, Y[idx], A[idx], refit_control,
                tuning_n = tuning_n, delta = refit_delta_value)$estimate
    }
  }
}

#' Bootstrap interval for SLDB
#'
#' @inheritParams sldb_estimator
#' @param R,level Passed to [sldb_boot_ci()].
#'
#' @return As [sldb_boot_ci()].
#' @export
sldb_ci_boot <- function(X, Z, Y, A = NULL, control = sldb_control(),
                         R = 1000L, level = 0.95,
                         lambda = NULL, delta = NULL) {
  sldb_boot_ci(sldb_estimator(X, Z, Y, A, control, lambda = lambda,
                              delta = delta),
               n = length(Z), R = R, level = level)
}


#' SLDB point estimator as a multi-outcome index-indexed closure
#'
#' The multi-outcome form of [sldb_estimator()]. The balancing weights never
#' see the outcome, so one weight fit serves every outcome in `Y_list`: the
#' returned closure fits the weights once per index draw and contracts them
#' against each outcome, returning a length-`K` vector instead of a scalar.
#' Feed it to [sldb_boot_ci_multi()], which reads each outcome off its own
#' column of the replicate matrix, or to [sldb_moonboot_ci()] one outcome at a
#' time.
#'
#' Tuning follows [sldb_estimator()] exactly: \eqn{\lambda_N} is frozen at
#' `tuning_n`, `refit_L` gives \eqn{L_m = \lceil c_L m^{\gamma_L}\rceil}, and
#' `refit_delta` rebuilds \eqn{\delta_m} from the subsample. A degenerate draw
#' -- one treatment arm empty, or a failed fit -- returns `NA` for every
#' outcome, and the interval routines drop it.
#'
#' @param X Numeric covariate matrix.
#' @param Z Integer treatment vector, coded 0/1.
#' @param Y_list List of numeric outcome vectors, all of length `nrow(X)`.
#' @param control An [sldb_control()] object.
#' @param tuning_n Reference sample size for `lambda` and `delta`.
#' @param lambda Fixed full-sample ridge used in every refit.
#' @param delta Fixed certified-gap tolerance, used only when
#'   `refit_delta = FALSE`. Ignored for `p = 2`.
#' @param refit_L,refit_delta As in [sldb_estimator()]. Use both for
#'   subsampling, neither for the bootstrap.
#'
#' @return A function of an integer index vector returning `length(Y_list)`
#'   estimates.
#' @export
sldb_estimator_multi <- function(X, Z, Y_list, control, tuning_n, lambda,
                                 delta, refit_L = FALSE, refit_delta = FALSE) {
  X <- as.matrix(X)
  Z <- as.integer(Z)
  d <- ncol(X)
  K <- length(Y_list)
  uses_delta <- as.integer(control$p) == 1L
  coefficient_delta <- is.null(control$delta_fraction)
  delta_coef <- if (is.null(control$delta_coef)) control$c_opt else control$delta_coef

  function(idx) {
    Zi <- Z[idx]
    if (sum(Zi == 1L) == 0L || sum(Zi == 0L) == 0L) return(rep(NA_real_, K))
    refit_control <- control
    refit_control$lambda <- lambda
    refit_control$proportion_to_initial_loss <- NULL
    if (refit_L) {
      refit_control$L <- sldb_L(refit_control, length(idx), d = d)
    }
    refit_delta_value <- if (!uses_delta) {
      NULL
    } else if (refit_delta && coefficient_delta) {
      delta_coef * lambda / length(idx)
    } else if (refit_delta) {
      ## Reference-objective schedule: NULL makes the engine rebuild
      ## delta_m = delta_fraction * J_ref(subsample) from the subsample.
      NULL
    } else {
      delta
    }
    wfit <- tryCatch(
      sldb_weights(X[idx, , drop = FALSE], Zi, refit_control,
                   tuning_n = tuning_n, delta = refit_delta_value),
      error = function(e) NULL)
    if (is.null(wfit)) return(rep(NA_real_, K))
    ## sldb_ate()$estimate is exactly sum(w * (2*Z - 1) * Y).
    contrast <- wfit$weights * (2 * Zi - 1)
    vapply(Y_list, function(Yv) sum(contrast * Yv[idx]), numeric(1))
  }
}

#' Multi-outcome percentile bootstrap interval
#'
#' [sldb_boot_ci()] for an estimator that returns `K` estimates per draw. One
#' index draw yields one row of `K` replicates and each outcome is inverted off
#' its own column, so every outcome sees the same resamples and the same weight
#' fits.
#'
#' @param estimator A function of an integer index vector returning `K`
#'   estimates, e.g. from [sldb_estimator_multi()].
#' @param n Full sample size.
#' @param K Number of outcomes.
#' @param R Number of bootstrap resamples.
#' @param level Confidence level.
#'
#' @return A length-`K` list, each element as [sldb_boot_ci()], carrying that
#'   outcome's column of the replicate matrix plus `m = n`.
#' @export
sldb_boot_ci_multi <- function(estimator, n, K, R = 1000L, level = 0.95) {
  stopifnot(is.function(estimator))
  alpha <- 1 - level
  vals <- matrix(NA_real_, R, K)
  for (r in seq_len(R)) {
    idx <- sample.int(n, n, replace = TRUE)
    vals[r, ] <- tryCatch(as.numeric(estimator(idx)),
                          error = function(e) rep(NA_real_, K))
  }
  lapply(seq_len(K), function(k) {
    ok <- vals[is.finite(vals[, k]), k]
    ci <- if (length(ok) < 2L) c(NA_real_, NA_real_) else {
      stats::quantile(ok, c(alpha / 2, 1 - alpha / 2), names = FALSE)
    }
    list(ci = ci, replicates = vals[, k], n_valid = length(ok), m = n)
  })
}


## ============================================================================
## Unified user-facing interface
##
## Everything above is the machinery: the balancing solver, the two estimands,
## the Wald standard errors, and the three interval routines. Everything below
## is the single entry point the analysis scripts use. The intended workflow is
##
##     pt  <- elapsed(sldb_weights(X, Z, CONTROL))
##     eff <- sldb_effect(pt, Y, A = A, estimand = "LATE",
##                        CI = c("Wald", "SS"), level = 0.95, R = 500L)
##     S   <- sldb_summary(eff)
##
## after which `S` is a tidy data frame with one row per (inference, outcome)
## carrying the estimate, the interval, the standard error, the subsampling
## diagnostics, every weight-fit diagnostic, and both timings.
## ============================================================================


## Timing

#' Time an expression and keep its value
#'
#' The drivers wrap the weight fit in this so `sldb_effect()` can report
#' `Time_Point` alongside the interval timings it measures itself.
#'
#' @param expr Any expression. It is forced exactly once.
#'
#' @return A list with `value` (the result of `expr`) and `time` (elapsed
#'   wall-clock seconds, from `proc.time()[["elapsed"]]`).
#' @export
elapsed <- function(expr) {
  t0  <- proc.time()[["elapsed"]]
  val <- force(expr)
  list(value = val, time = proc.time()[["elapsed"]] - t0)
}


## Tuning bundles

#' Subsampling tuning options
#'
#' Collects every tuning argument of [sldb_moonboot_ci()] other than the
#' estimator, `n`, `R` and `level`, which [sldb_effect()] supplies itself. The
#' defaults are exactly the [sldb_moonboot_ci()] defaults, so
#' `sldb_ss_control()` reproduces a bare `sldb_moonboot_ci(estimator, n, R,
#' level)` call.
#'
#' @param tau Rate function, or `NULL` to fit it with [sldb_estimate_tau()].
#' @param m Subsample size, or `NULL` to select it with `m_method`.
#' @param m_method Selection rule: `"bickel"`, `"goetze"`, `"politis"` or
#'   `"sherman"`.
#' @param m_R Resamples drawn at each candidate size during the selection.
#' @param tau_R Resamples drawn at each rung of the rate ladder.
#' @param min_m Smallest admissible subsample size.
#' @param m_grid Candidate sizes for `m_method = "bickel"`, or `NULL`.
#' @param n_grid,range_exp Passed to [sldb_m_grid_bickel()].
#' @param tau_n_grid,tau_span,tau_probs,tau_exponent_range Passed to
#'   [sldb_estimate_tau()].
#' @param params Extra parameters for the `moonboot` selection rules.
#' @param type Interval type from [moonboot::mboot.ci()].
#'
#' @return A list of class `"sldb_ss_control"`, suitable as the `ss` argument
#'   of [sldb_effect()].
#' @export
sldb_ss_control <- function(tau = NULL, m = NULL, m_method = "bickel",
                            m_R = 200L, tau_R = 200L, min_m = 20L,
                            m_grid = NULL, n_grid = 20L,
                            range_exp = c(1 / 3, 2 / 3),
                            tau_n_grid = 5L, tau_span = 2,
                            tau_probs = c(0.05, 0.10, 0.25),
                            tau_exponent_range = c(0.45, 0.5),
                            params = NULL, type = "basic") {
  structure(
    list(tau = tau, m = m, m_method = m_method, m_R = m_R, tau_R = tau_R,
         min_m = min_m, m_grid = m_grid, n_grid = n_grid,
         range_exp = range_exp, tau_n_grid = tau_n_grid, tau_span = tau_span,
         tau_probs = tau_probs, tau_exponent_range = tau_exponent_range,
         params = params, type = type),
    class = "sldb_ss_control")
}


## Internal helpers

#' Arm-normalised weighted contrast
#'
#' The `weighted_contrast()` of the driver scripts, reproduced here so that
#' `SLDB.R` carries no dependency on `competitors.R`. Kept distinct from
#' `sum(w * (2 Z - 1) * Y)`: the two agree to within rounding whenever the
#' weights sum to one in each arm, but not bit for bit.
#' @keywords internal
#' @noRd
.sldb_weighted_contrast <- function(w, Z, V) {
  w1 <- w * Z
  w0 <- w * (1 - Z)
  s1 <- sum(w1)
  s0 <- sum(w0)
  if (!is.finite(s1) || !is.finite(s0) || s1 <= 0 || s0 <= 0) return(NA_real_)
  sum(w1 * V) / s1 - sum(w0 * V) / s0
}

#' Full-sample point estimates, one per outcome
#'
#' ATE uses `sum(w (2Z - 1) Y)`, which is bit for bit `sldb_ate()$estimate`.
#' LATE uses the arm-normalised Wald ratio of `.sldb_weighted_contrast()`,
#' which is what `Data.R` reports.
#' @keywords internal
#' @noRd
.sldb_point_estimates <- function(w, Z, Y_list, A, estimand) {
  if (identical(estimand, "ATE")) {
    contrast <- w * (2 * Z - 1)
    return(list(estimate = vapply(Y_list, function(Yv) sum(contrast * Yv),
                                  numeric(1)),
                first_stage = NA_real_))
  }
  den <- .sldb_weighted_contrast(w, Z, A)
  degenerate <- !is.finite(den) || abs(den) < .Machine$double.eps^0.5
  est <- vapply(Y_list, function(Yv) {
    num <- .sldb_weighted_contrast(w, Z, Yv)
    if (!is.finite(num) || degenerate) NA_real_ else num / den
  }, numeric(1))
  list(estimate = est, first_stage = den)
}

#' Flat diagnostic summary of a weight fit
#'
#' Every scalar the drivers report off a weight fit, including the two derived
#' `p = 1` quantities.
#' @keywords internal
#' @noRd
.sldb_weight_diagnostics <- function(wfit) {
  p_order   <- as.integer(wfit$p)
  final_gap <- as.numeric(wfit$final_gap)
  delta_n   <- as.numeric(wfit$delta_n)

  ## Reproduced verbatim from the drivers: both are p = 1 quantities and stay
  ## NA for p = 2, which neither computes nor uses a certified gap.
  delta_criterion_met <- NA
  gap_over_delta      <- NA_real_
  if (p_order == 1L) {
    delta_criterion_met <- is.finite(final_gap) && is.finite(delta_n) &&
      final_gap <= delta_n
    if (is.finite(delta_n) && delta_n > 0) gap_over_delta <- final_gap / delta_n
  }

  ctrl <- wfit$control
  list(lambda                  = as.numeric(wfit$lambda),
       L                       = as.integer(wfit$L),
       L_Coefficient           = if (is.null(ctrl$L_coefficient)) NA_real_ else
                                   as.numeric(ctrl$L_coefficient),
       L_Exponent              = if (is.null(ctrl$L_exponent)) NA_real_ else
                                   as.numeric(ctrl$L_exponent),
       p                       = p_order,
       Optimizer               = as.character(wfit$optimizer),
       Optim_Convergence       = as.integer(wfit$optim_convergence),
       Gradient_Norm           = as.numeric(wfit$gradient_norm),
       Converged               = wfit$converged,
       Delta_Criterion_Met     = delta_criterion_met,
       Iterations              = as.integer(wfit$iterations),
       Final_Gap               = final_gap,
       Gap_over_Delta          = gap_over_delta,
       Eta_Initial             = as.numeric(wfit$eta_initial),
       Eta_Final               = as.numeric(wfit$eta_final),
       Eta_Increases           = as.integer(wfit$eta_increases),
       Eta_Decreases           = as.integer(wfit$eta_decreases),
       Initial_Loss            = as.numeric(wfit$initial_loss),
       q_constant              = as.numeric(wfit$q_constant),
       Proportion_Initial_Loss = as.numeric(wfit$proportion_to_initial_loss),
       Uniform_Objective       = as.numeric(wfit$uniform_objective),
       Logistic_Objective      = as.numeric(wfit$logistic_objective),
       Reference_Objective     = as.numeric(wfit$reference_objective),
       Initial_Source          = as.character(wfit$initial_source),
       delta                   = delta_n)
}

#' Resampling closure covering both estimands and any number of outcomes
#'
#' The one closure behind every SLDB subsampling and bootstrap run. It is the
#' `sldb_estimator_multi()` refit rule -- freeze the ridge, optionally rebuild
#' `L` and `delta` at the resample size -- fitting the weights once per index
#' draw and contracting them against each outcome. With `A = NULL` it returns
#' the ATE contrasts, matching `sldb_estimator_multi()` bit for bit; with `A`
#' supplied it returns the Wald ratios `mean(a Y) / mean(a A)`, matching
#' `sldb_estimator()` bit for bit.
#' @keywords internal
#' @noRd
.sldb_resample_estimator <- function(X, Z, Y_list, A, control, tuning_n,
                                     lambda, delta, refit_L, refit_delta) {
  X <- as.matrix(X)
  Z <- as.integer(Z)
  d <- ncol(X)
  K <- length(Y_list)
  if (!is.null(A)) A <- as.numeric(A)
  uses_delta <- as.integer(control$p) == 1L
  coefficient_delta <- is.null(control$delta_fraction)
  delta_coef <- if (is.null(control$delta_coef)) control$c_opt else control$delta_coef

  function(idx) {
    Zi <- Z[idx]
    if (sum(Zi == 1L) == 0L || sum(Zi == 0L) == 0L) return(rep(NA_real_, K))
    refit_control <- control
    refit_control$lambda <- lambda
    refit_control$proportion_to_initial_loss <- NULL
    if (refit_L) {
      refit_control$L <- sldb_L(refit_control, length(idx), d = d)
    }
    refit_delta_value <- if (!uses_delta) {
      NULL
    } else if (refit_delta && coefficient_delta) {
      delta_coef * lambda / length(idx)
    } else if (refit_delta) {
      ## Reference-objective schedule: NULL makes the engine rebuild
      ## delta_m = delta_fraction * J_ref(subsample) from the subsample.
      NULL
    } else {
      delta
    }
    wfit <- tryCatch(
      sldb_weights(X[idx, , drop = FALSE], Zi, refit_control,
                   tuning_n = tuning_n, delta = refit_delta_value),
      error = function(e) NULL)
    if (is.null(wfit)) return(rep(NA_real_, K))
    if (is.null(A)) {
      ## sldb_ate()$estimate is exactly sum(w * (2*Z - 1) * Y).
      contrast <- wfit$weights * (2 * Zi - 1)
      return(vapply(Y_list, function(Yv) sum(contrast * Yv[idx]), numeric(1)))
    }
    ## sldb_late()$estimate is exactly mean(a * Y) / mean(a * A).
    a   <- .sldb_contrast(wfit$weights, Zi, length(idx))
    den <- mean(a * A[idx])
    if (!is.finite(den) || abs(den) < .Machine$double.eps^0.5) {
      return(rep(NA_real_, K))
    }
    vapply(Y_list, function(Yv) mean(a * Yv[idx]) / den, numeric(1))
  }
}

#' Normalise the `Y` argument of sldb_effect() to a list of outcome vectors
#' @keywords internal
#' @noRd
.sldb_outcome_list <- function(Y, n) {
  Y_list <- if (is.list(Y)) Y else list(Y)
  Y_list <- lapply(Y_list, as.numeric)
  if (!length(Y_list)) stop("`Y` must hold at least one outcome.", call. = FALSE)
  if (any(vapply(Y_list, length, integer(1)) != n)) {
    stop("Every outcome in `Y` must have length ", n, ".", call. = FALSE)
  }
  Y_list
}


## The single entry point

#' SLDB effect estimation and inference, from a fitted set of weights
#'
#' One call that takes the balancing weights and returns the point estimate,
#' every requested confidence interval, all interval diagnostics, and the
#' computation times. It is the only SLDB routine the analysis scripts call
#' after fitting the weights.
#'
#' The weights are a *design-stage* object: they never see an outcome, so one
#' fit serves every outcome, both estimands and all three interval procedures.
#' `sldb_weights()` records the `X`, `Z` and resolved `control` it was fitted
#' on, which is what lets this function refit for resampling without being
#' handed the data again.
#'
#' **Interval procedures.**
#' * `"Wald"` reuses the fitted weights, so it costs one pass over the outcome,
#'   no refit, and no random numbers. It is built from [sldb_ate()] or
#'   [sldb_late()] and [sldb_ci_wald()].
#' * `"SS"` is the subsampling interval of [sldb_moonboot_ci()]: fit the rate,
#'   choose `m` by Bickel-Sakov, then invert at that `m`. Each refit rebuilds
#'   `L` at the subsample size, and for `p = 1` rebuilds `delta` too.
#' * `"Boot"` is the percentile bootstrap of [sldb_boot_ci_multi()]. Refits
#'   keep the full-sample `L` and `delta` schedule.
#'
#' Requested procedures run in the order given, which fixes the random-number
#' stream: `"Wald"` draws none, so `CI = c("Wald", "SS")` leaves the `"SS"`
#' draws identical to a standalone `"SS"` run.
#'
#' **One detail worth knowing about the LATE.** The reported `estimate` is the
#' arm-normalised Wald ratio
#' `weighted_contrast(w, Z, Y) / weighted_contrast(w, Z, A)`, matching the
#' estimator the comparison methods use, while the `"Wald"` interval is centred
#' on `sldb_late()$estimate`, which is `mean(a Y) / mean(a A)`. The two agree to
#' within rounding -- the weights sum to one in each arm -- but not bit for bit,
#' so a LATE Wald interval is not exactly symmetric about `estimate`. The
#' resampling replicates use the `sldb_late()` form. The ATE has no such split:
#' `sum(w (2Z - 1) Y)` is bit for bit `sldb_ate()$estimate`.
#'
#' @param weights An `"sldb_weights"` object from [sldb_weights()], or the
#'   `list(value, time)` that [elapsed()] returns around one. In the latter case
#'   the recorded time becomes the returned `time_point`.
#' @param Y A numeric outcome vector, or a list of numeric outcome vectors for
#'   the multi-outcome case. One weight fit is shared across the list.
#' @param A Treatment-received vector. Required for `estimand = "LATE"`,
#'   ignored for `"ATE"`.
#' @param estimand `"ATE"` (default) or `"LATE"`. For `"LATE"` the weights are
#'   built from the instrument `Z` and the estimate is the Wald ratio.
#' @param CI Which intervals to compute: any subset of `"Wald"`, `"SS"` and
#'   `"Boot"`, in the order they should run. Omitting the argument computes all
#'   three. Use `character(0)` for a point estimate with no inference.
#' @param level Confidence level. Default `0.95`.
#' @param R Resample count: bootstrap draws, and subsampling draws at the
#'   selected `m`. Default `1000L`; the analysis scripts use `500L`.
#' @param CI.Time If `TRUE` (the default), record the wall-clock seconds each
#'   interval procedure took, in `time_ci` and the summary's `Time_Interval`.
#'   If `FALSE` those fields are `NA_real_`.
#' @param wald_centered Passed to [sldb_ci_wald()] as `centered`. `TRUE`, the
#'   default, uses the variance-minimising `se_min`.
#' @param ss Subsampling tuning, from [sldb_ss_control()]. Its defaults are the
#'   [sldb_moonboot_ci()] defaults.
#' @param refit_L,refit_delta Override the per-procedure refit rule, which is
#'   `TRUE` for `"SS"` (and `refit_delta` additionally requires `p = 1`) and
#'   `FALSE` for `"Boot"`. Leave `NULL` to keep that rule.
#' @param tuning_n Reference sample size that pins the ridge in every refit.
#'   Defaults to the size the weights were fitted on.
#' @param lambda Ridge frozen across refits. Defaults to the fitted
#'   `weights$lambda`.
#' @param delta Certified-gap tolerance frozen across refits, used only where
#'   `refit_delta` is `FALSE`. Defaults to the fitted `weights$delta_n` for
#'   `p = 1` and `NULL` for `p = 2`.
#' @param X,Z,control Only needed when `weights` predates the fields that
#'   [sldb_weights()] now records; otherwise taken from the fit.
#'
#' @return An object of class `"sldb_effect"`: a list with
#'   \describe{
#'     \item{`estimand`, `level`, `R`, `n`, `K`}{what was run.}
#'     \item{`estimate`}{length-`K` numeric, one point estimate per outcome.}
#'     \item{`first_stage`}{weighted first stage, `NA_real_` for the ATE.}
#'     \item{`time_point`}{seconds spent fitting the weights, when `weights`
#'       came through [elapsed()]; `NA_real_` otherwise.}
#'     \item{`time_ci`}{named numeric, seconds per interval procedure.}
#'     \item{`inference`}{named list, one entry per requested procedure, each
#'       with `type`, `level`, length-`K` `lower`, `upper`, `length`, `se`,
#'       `m`, `tau_exponent`, `n_valid`, plus `m_R`, `R`, `time` and `detail`
#'       (the raw [sldb_moonboot_ci()] / [sldb_boot_ci_multi()] / [sldb_ate()]
#'       output, including replicates, `tau_fit` and the Bickel-Sakov
#'       `search`).}
#'     \item{`weights`}{the `"sldb_weights"` object it ran off.}
#'     \item{`diagnostics`}{every weight-fit scalar the drivers report.}
#'   }
#'   [sldb_summary()] flattens all of this into one row per
#'   (procedure, outcome).
#' @export
sldb_effect <- function(weights, Y, A = NULL,
                        estimand = c("ATE", "LATE"),
                        CI = c("Wald", "SS", "Boot"),
                        level = 0.95,
                        R = 1000L,
                        CI.Time = TRUE,
                        wald_centered = TRUE,
                        ss = sldb_ss_control(),
                        refit_L = NULL,
                        refit_delta = NULL,
                        tuning_n = NULL,
                        lambda = NULL,
                        delta = NULL,
                        X = NULL, Z = NULL, control = NULL) {

  ## ---- unwrap the (optionally timed) weight fit ----------------------------
  time_point <- NA_real_
  if (!inherits(weights, "sldb_weights") && is.list(weights) &&
      all(c("value", "time") %in% names(weights))) {
    time_point <- as.numeric(weights$time)
    weights    <- weights$value
  }
  if (!inherits(weights, "sldb_weights")) {
    stop("`weights` must be an sldb_weights object from sldb_weights(), or ",
         "elapsed(sldb_weights(...)).", call. = FALSE)
  }
  if (is.null(X))       X       <- weights$X
  if (is.null(Z))       Z       <- weights$Z
  if (is.null(control)) control <- weights$control
  if (is.null(X) || is.null(Z) || is.null(control)) {
    stop("This weights object does not carry `X`, `Z` and `control`; supply ",
         "them to sldb_effect() explicitly.", call. = FALSE)
  }

  estimand <- match.arg(estimand)
  CI <- if (length(CI)) match.arg(CI, c("Wald", "SS", "Boot"),
                                  several.ok = TRUE) else character(0)
  if (!is.numeric(level) || length(level) != 1L || !is.finite(level) ||
      level <= 0 || level >= 1) {
    stop("`level` must be one number strictly between 0 and 1.", call. = FALSE)
  }
  if (!inherits(ss, "sldb_ss_control")) {
    stop("`ss` must come from sldb_ss_control().", call. = FALSE)
  }

  n      <- length(weights$weights)
  w      <- weights$weights
  Z      <- as.integer(Z)
  Y_list <- .sldb_outcome_list(Y, n)
  K      <- length(Y_list)
  outcome_names <- if (!is.null(names(Y_list))) names(Y_list) else NULL

  if (identical(estimand, "LATE")) {
    if (is.null(A)) stop("`A` is required for estimand = \"LATE\".",
                         call. = FALSE)
    A <- as.numeric(A)
    if (length(A) != n) stop("`A` must have length ", n, ".", call. = FALSE)
  } else {
    A <- NULL
  }

  ## ---- point estimates and weight diagnostics ------------------------------
  point <- .sldb_point_estimates(w, Z, Y_list, A, estimand)
  diagnostics <- .sldb_weight_diagnostics(weights)

  ## ---- resampling schedule, frozen at the full-sample fit ------------------
  p_order <- as.integer(weights$p)
  if (is.null(tuning_n)) tuning_n <- weights$n
  if (is.null(lambda))   lambda   <- weights$lambda
  if (is.null(delta))    delta    <- if (p_order == 1L) weights$delta_n else NULL

  make_estimator <- function(default_refit_L, default_refit_delta) {
    .sldb_resample_estimator(
      X, Z, Y_list, A, control,
      tuning_n    = tuning_n,
      lambda      = lambda,
      delta       = delta,
      refit_L     = if (is.null(refit_L))     default_refit_L     else refit_L,
      refit_delta = if (is.null(refit_delta)) default_refit_delta else refit_delta)
  }

  ## ---- one entry per requested procedure, in the order given ---------------
  na_K <- function() rep(NA_real_, K)
  entry <- function(type, lower, upper, se, m, tau_exponent, n_valid, m_R,
                    detail, time) {
    list(type = type, level = level, R = R,
         lower = lower, upper = upper, length = upper - lower,
         se = se, m = m, tau_exponent = tau_exponent, n_valid = n_valid,
         m_R = m_R, time = time, detail = detail)
  }

  run_wald <- function() {
    fits <- lapply(seq_len(K), function(k) {
      fit <- if (identical(estimand, "ATE")) {
        sldb_ate(X, Z, Y_list[[k]], control, weights = weights)
      } else {
        sldb_late(X, Z, Y_list[[k]], A, control, weights = weights)
      }
      list(fit = fit, ci = sldb_ci_wald(fit, level, centered = wald_centered))
    })
    fits
  }

  run_ss <- function() {
    estimator <- make_estimator(TRUE, p_order == 1L)
    lapply(seq_len(K), function(k) {
      do.call(sldb_moonboot_ci,
              c(list(estimator = function(idx) estimator(idx)[k],
                     n = n, R = R, level = level),
                unclass(ss)))
    })
  }

  run_boot <- function() {
    estimator <- make_estimator(FALSE, FALSE)
    sldb_boot_ci_multi(estimator, n = n, K = K, R = R, level = level)
  }

  inference <- list()
  time_ci   <- numeric(0)
  for (type in CI) {
    timed <- elapsed(switch(type, Wald = run_wald(), SS = run_ss(),
                            Boot = run_boot()))
    out   <- timed$value
    secs  <- if (isTRUE(CI.Time)) timed$time else NA_real_

    inference[[type]] <- switch(
      type,
      Wald = entry(
        "Wald",
        lower        = vapply(out, function(z) z$ci[1L], numeric(1)),
        upper        = vapply(out, function(z) z$ci[2L], numeric(1)),
        se           = vapply(out, function(z) {
          if (isTRUE(wald_centered)) z$fit$se_min else z$fit$se
        }, numeric(1)),
        ## No resampling: the interval is read at the full sample size.
        m            = rep(as.numeric(n), K),
        tau_exponent = na_K(),
        n_valid      = rep(NA_integer_, K),
        m_R          = NA_integer_,
        detail       = lapply(out, `[[`, "fit"),
        time         = secs),
      SS = entry(
        "SS",
        lower        = vapply(out, function(z) z$ci[1L], numeric(1)),
        upper        = vapply(out, function(z) z$ci[2L], numeric(1)),
        se           = na_K(),
        m            = vapply(out, function(z) as.integer(z$m), integer(1)),
        tau_exponent = vapply(out, function(z) as.numeric(z$tau_exponent),
                              numeric(1)),
        n_valid      = vapply(out, function(z) as.integer(z$n_valid),
                              integer(1)),
        m_R          = ss$m_R,
        detail       = out,
        time         = secs),
      Boot = entry(
        "Boot",
        lower        = vapply(out, function(z) z$ci[1L], numeric(1)),
        upper        = vapply(out, function(z) z$ci[2L], numeric(1)),
        se           = na_K(),
        m            = rep(as.numeric(n), K),
        tau_exponent = na_K(),
        n_valid      = vapply(out, function(z) as.integer(z$n_valid),
                              integer(1)),
        m_R          = NA_integer_,
        detail       = out,
        time         = secs))
    time_ci[[type]] <- secs
  }

  structure(
    list(estimand = estimand, level = level, R = R, n = n, K = K,
         outcome_names = outcome_names,
         estimate = point$estimate, first_stage = point$first_stage,
         time_point = time_point, time_ci = time_ci,
         inference = inference, weights = weights,
         diagnostics = diagnostics),
    class = "sldb_effect")
}

#' @export
print.sldb_effect <- function(x, ...) {
  cat(sprintf("<sldb_effect %s>  n = %d,  outcomes = %d\n",
              x$estimand, x$n, x$K))
  cat(sprintf("  estimate = %s\n",
              paste(sprintf("%.6g", x$estimate), collapse = ", ")))
  if (identical(x$estimand, "LATE")) {
    cat(sprintf("  weighted first stage = %.6g\n", x$first_stage))
  }
  for (e in x$inference) {
    cat(sprintf("  %-4s %g%% CI = %s%s\n", e$type, 100 * e$level,
                paste(sprintf("[%.6g, %.6g]", e$lower, e$upper),
                      collapse = " "),
                if (identical(e$type, "SS")) {
                  sprintf("   (m = %d, tau exponent = %.3f)",
                          as.integer(e$m[1L]), e$tau_exponent[1L])
                } else ""))
  }
  invisible(x)
}


## Tidy output

#' Column template shared by [sldb_summary()] and its empty form
#' @keywords internal
#' @noRd
.sldb_summary_columns <- c(
  "Estimand", "Inference", "Outcome",
  "Estimate", "First_Stage", "CI_Lower", "CI_Upper", "Length",
  "SE_AsymptoticNormality", "m", "Tau_Exponent", "SS_m_R", "N_Valid",
  "Level", "R",
  "lambda", "L", "L_Coefficient", "L_Exponent", "p",
  "Optimizer", "Optim_Convergence", "Gradient_Norm",
  "Converged", "Delta_Criterion_Met", "Iterations",
  "Final_Gap", "Gap_over_Delta",
  "Eta_Initial", "Eta_Final", "Eta_Increases", "Eta_Decreases",
  "Initial_Loss", "q_constant", "Proportion_Initial_Loss",
  "Uniform_Objective", "Logistic_Objective", "Reference_Objective",
  "Initial_Source", "delta",
  "Time_Point", "Time_Interval")

#' Flatten an SLDB effect into one row per (procedure, outcome)
#'
#' The bridge between [sldb_effect()] and the analysis scripts: everything the
#' drivers write to disk, in a data frame with stable column names and stable
#' column types. Rows are ordered procedure-major, matching the order the
#' procedures were requested in, with the outcomes in their original order
#' inside each procedure.
#'
#' Called with `effect = NULL` it returns the same frame filled with `NA`,
#' which lets a driver that also runs non-SLDB methods build its output rows
#' through one code path. `SS_m_R` is still filled in that case, because the
#' competing methods reach [sldb_moonboot_ci()] with the same `m_R`.
#'
#' @param effect An `"sldb_effect"` object, or `NULL` for the empty template.
#' @param K Number of outcome rows in the template. Ignored when `effect` is
#'   given.
#' @param inference Procedure name(s) to label the template rows with.
#' @param m_R The `m_R` recorded on template `"SS"` rows.
#'
#' @return A data frame with `length(CI) * K` rows (or `length(inference) * K`
#'   for the template) and the columns listed in `.sldb_summary_columns`.
#' @export
sldb_summary <- function(effect = NULL, K = 1L, inference = NA_character_,
                         m_R = sldb_ss_control()$m_R) {

  ## ---- empty template -----------------------------------------------------
  if (is.null(effect)) {
    rows <- lapply(inference, function(type) {
      data.frame(
        Estimand = NA_character_, Inference = as.character(type),
        Outcome = seq_len(K),
        Estimate = NA_real_, First_Stage = NA_real_,
        CI_Lower = NA_real_, CI_Upper = NA_real_, Length = NA_real_,
        SE_AsymptoticNormality = NA_real_, m = NA_real_,
        Tau_Exponent = NA_real_,
        SS_m_R = if (identical(type, "SS")) as.numeric(m_R) else NA_integer_,
        N_Valid = NA_integer_, Level = NA_real_, R = NA_integer_,
        lambda = NA_real_, L = NA_integer_, L_Coefficient = NA_real_,
        L_Exponent = NA_real_, p = NA_integer_,
        Optimizer = NA_character_, Optim_Convergence = NA_integer_,
        Gradient_Norm = NA_real_, Converged = NA,
        Delta_Criterion_Met = NA, Iterations = NA_integer_,
        Final_Gap = NA_real_, Gap_over_Delta = NA_real_,
        Eta_Initial = NA_real_, Eta_Final = NA_real_,
        Eta_Increases = NA_integer_, Eta_Decreases = NA_integer_,
        Initial_Loss = NA_real_, q_constant = NA_real_,
        Proportion_Initial_Loss = NA_real_, Uniform_Objective = NA_real_,
        Logistic_Objective = NA_real_, Reference_Objective = NA_real_,
        Initial_Source = NA_character_, delta = NA_real_,
        Time_Point = NA_real_, Time_Interval = NA_real_,
        row.names = NULL, stringsAsFactors = FALSE)
    })
    out <- do.call(rbind, rows)
    return(out[, .sldb_summary_columns, drop = FALSE])
  }

  stopifnot(inherits(effect, "sldb_effect"))
  g <- effect$diagnostics
  K <- effect$K

  ## With no interval requested there is still a point estimate and a full set
  ## of weight diagnostics to report; the interval columns stay NA.
  entries <- effect$inference
  if (!length(entries)) {
    entries <- list(list(type = NA_character_, level = effect$level,
                         R = effect$R, lower = rep(NA_real_, K),
                         upper = rep(NA_real_, K), length = rep(NA_real_, K),
                         se = rep(NA_real_, K), m = rep(NA_real_, K),
                         tau_exponent = rep(NA_real_, K),
                         n_valid = rep(NA_integer_, K), m_R = NA_integer_,
                         time = NA_real_))
  }

  rows <- lapply(entries, function(e) {
    data.frame(
      Estimand = effect$estimand, Inference = e$type, Outcome = seq_len(K),
      Estimate = effect$estimate, First_Stage = effect$first_stage,
      CI_Lower = e$lower, CI_Upper = e$upper, Length = e$length,
      SE_AsymptoticNormality = e$se, m = e$m, Tau_Exponent = e$tau_exponent,
      SS_m_R = if (identical(e$type, "SS")) as.numeric(e$m_R) else NA_integer_,
      N_Valid = e$n_valid, Level = e$level, R = e$R,
      lambda = g$lambda, L = g$L, L_Coefficient = g$L_Coefficient,
      L_Exponent = g$L_Exponent, p = g$p,
      Optimizer = g$Optimizer, Optim_Convergence = g$Optim_Convergence,
      Gradient_Norm = g$Gradient_Norm, Converged = g$Converged,
      Delta_Criterion_Met = g$Delta_Criterion_Met, Iterations = g$Iterations,
      Final_Gap = g$Final_Gap, Gap_over_Delta = g$Gap_over_Delta,
      Eta_Initial = g$Eta_Initial, Eta_Final = g$Eta_Final,
      Eta_Increases = g$Eta_Increases, Eta_Decreases = g$Eta_Decreases,
      Initial_Loss = g$Initial_Loss, q_constant = g$q_constant,
      Proportion_Initial_Loss = g$Proportion_Initial_Loss,
      Uniform_Objective = g$Uniform_Objective,
      Logistic_Objective = g$Logistic_Objective,
      Reference_Objective = g$Reference_Objective,
      Initial_Source = g$Initial_Source, delta = g$delta,
      Time_Point = effect$time_point, Time_Interval = e$time,
      row.names = NULL, stringsAsFactors = FALSE)
  })
  out <- do.call(rbind, unname(rows))
  out[, .sldb_summary_columns, drop = FALSE]
}
