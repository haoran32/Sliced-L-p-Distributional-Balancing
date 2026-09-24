#include <Rcpp.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <vector>
#include <cmath>

using namespace Rcpp;

// ============================================================
// Helper Function: Project onto the Unit Simplex
// (unchanged)
// ============================================================
void project_simplex_cpp(std::vector<double>& v, const std::vector<int>& indices, double target_sum, NumericVector& w_out) {
  if (indices.empty()) return;
  std::vector<double> u;
  u.reserve(indices.size());
  for (int idx : indices) {
    u.push_back(v[idx]);
  }
  std::sort(u.begin(), u.end(), std::greater<double>());
  double css = 0.0;
  int rho = -1;
  for (size_t i = 0; i < u.size(); ++i) {
    css += u[i];
    if (u[i] > (css - target_sum) / (i + 1.0)) {
      rho = i;
    }
  }
  double theta = 0.0;
  if (rho != -1) {
    double css_rho = 0.0;
    for(int k=0; k<=rho; k++) css_rho += u[k];
    theta = (css_rho - target_sum) / (rho + 1.0);
  }
  for (int idx : indices) {
    double val = v[idx] - theta;
    w_out[idx] = (val > 0.0) ? val : 0.0;
  }
}

// ============================================================
// Core Function: Sliced Wasserstein (SW) Objective + Gradient
// Supports Three-Way Balancing (Treated vs Sample, Control vs Sample, Treated vs Control)
// By setting alpha=0.0, we obtain Two-Way Balancing.
//
// NOTE: modified to also return the objective value Jhat(w), since the
// certified gap needs both g^(t) and Jhat(w^(t)) at the same iterate.
// This avoids a second, separate pass over the data to evaluate Jhat.
// ============================================================
// [[Rcpp::plugins(openmp)]]
List sw_obj_grad_cpp_parallel(NumericVector w, IntegerMatrix idx_sorted,
                              IntegerVector A, NumericMatrix Delta,
                              int L, double lambda, double alpha = 1.0) {
  int n = w.size();
  NumericVector grad(n, 0.0);
  int n_minus_1 = n - 1;
  double total_loss = 0.0;

  std::vector<double> target_cdf(n_minus_1);
  for (int i = 0; i < n_minus_1; i++) target_cdf[i] = (i + 1.0) / n;

  const int* p_idx = &idx_sorted[0];
  const double* p_delta = &Delta[0];
  const int* p_A = &A[0];
  const double* p_w = &w[0];
  double* p_grad = &grad[0];

#pragma omp parallel
{
  std::vector<double> local_grad(n, 0.0);
  double local_loss = 0.0;
  std::vector<double> vk1S(n_minus_1), vk0S(n_minus_1), vk10(n_minus_1);

#pragma omp for nowait
  for (int l = 0; l < L; l++) {
    int col_offset_idx = l * n;
    int col_offset_delta = l * n_minus_1;

    double cumw1 = 0.0, cumw0 = 0.0;
    double W1S = 0.0, W0S = 0.0, W10 = 0.0;
    double s1S = 0.0, s0S = 0.0, s10 = 0.0;

    for (int i = 0; i < n_minus_1; i++) {
      int orig = p_idx[col_offset_idx + i] - 1;
      if (p_A[orig] == 1) cumw1 += p_w[orig];
      else cumw0 += p_w[orig];

      double e1S = cumw1 - target_cdf[i];
      double e0S = cumw0 - target_cdf[i];
      double e10 = cumw1 - cumw0;

      double d_val = p_delta[col_offset_delta + i];
      W1S += d_val * std::abs(e1S);
      W0S += d_val * std::abs(e0S);
      W10 += d_val * std::abs(e10);

      vk1S[i] = d_val * (e1S > 0 ? 1.0 : (e1S < 0 ? -1.0 : 0.0));
      vk0S[i] = d_val * (e0S > 0 ? 1.0 : (e0S < 0 ? -1.0 : 0.0));
      vk10[i] = d_val * (e10 > 0 ? 1.0 : (e10 < 0 ? -1.0 : 0.0));

      s1S += vk1S[i];
      s0S += vk0S[i];
      s10 += vk10[i];
    }

    // Accumulate this direction's contribution to Jhat (before the 1/L average)
    local_loss += W1S * W1S + W0S * W0S + alpha * W10 * W10;

    double cvk1S = 0.0, cvk0S = 0.0, cvk10 = 0.0;
    for (int i = 0; i < n; i++) {
      int orig = p_idx[col_offset_idx + i] - 1;

      if (p_A[orig] == 1) {
        local_grad[orig] += (s1S - cvk1S) * (2.0 * W1S);
        local_grad[orig] += (s10 - cvk10) * (2.0 * W10 * alpha);
      } else {
        local_grad[orig] += (s0S - cvk0S) * (2.0 * W0S);
        local_grad[orig] -= (s10 - cvk10) * (2.0 * W10 * alpha);
      }

      if (i < n_minus_1) {
        cvk1S += vk1S[i];
        cvk0S += vk0S[i];
        cvk10 += vk10[i];
      }
    }
  }

#pragma omp critical
{
  for (int i = 0; i < n; i++) p_grad[i] += local_grad[i];
  total_loss += local_loss;
}
}

double invL = 1.0 / L;
double w_sq_sum = 0.0;
for (int i = 0; i < n; i++) {
  p_grad[i] = p_grad[i] * invL + 2.0 * lambda * p_w[i];
  w_sq_sum += p_w[i] * p_w[i];
}
double Jhat = total_loss * invL + lambda * w_sq_sum;

return List::create(Named("grad") = grad, Named("Jhat") = Jhat);
}

// [[Rcpp::plugins(openmp)]]
// [[Rcpp::export]]
List prepare_sw_p2_cpp(NumericMatrix X, int L) {
  int n = X.nrow();
  int p = X.ncol();
  NumericMatrix proj(p, L);
  for (int j = 0; j < L; j++) {
    double sum_sq = 0.0;
    for (int i = 0; i < p; i++) {
      proj(i, j) = R::rnorm(0, 1);
      sum_sq += proj(i, j) * proj(i, j);
    }
    double norm = std::sqrt(sum_sq);
    for (int i = 0; i < p; i++) proj(i, j) /= norm;
  }

  IntegerMatrix idx_sorted(n, L);
  NumericMatrix Delta(n - 1, L);
#pragma omp parallel
  {
    std::vector<std::pair<double, int>> col_data(n);
#pragma omp for
    for (int l = 0; l < L; l++) {
      for (int i = 0; i < n; i++) {
        double value = 0.0;
        for (int k = 0; k < p; k++) value += X(i, k) * proj(k, l);
        col_data[i] = std::make_pair(value, i);
      }
      std::sort(col_data.begin(), col_data.end());
      for (int i = 0; i < n; i++) {
        idx_sorted(i, l) = col_data[i].second + 1;
        if (i < n - 1) {
          Delta(i, l) = col_data[i + 1].first - col_data[i].first;
        }
      }
    }
  }
  return List::create(Named("idx_sorted") = idx_sorted,
                      Named("Delta") = Delta);
}

// [[Rcpp::plugins(openmp)]]
// [[Rcpp::export]]
List sw_obj_grad_p2_cpp(NumericVector w, IntegerMatrix idx_sorted,
                        IntegerVector A, NumericMatrix Delta,
                        double lambda, double alpha = 1.0) {
  int n = w.size();
  int L = idx_sorted.ncol();
  int n_minus_1 = n - 1;
  NumericVector grad(n, 0.0);
  double total_loss = 0.0;

  const int* p_idx = &idx_sorted[0];
  const double* p_delta = &Delta[0];
  const int* p_A = &A[0];
  const double* p_w = &w[0];
  double* p_grad = &grad[0];

#pragma omp parallel
  {
    std::vector<double> local_grad(n, 0.0);
    std::vector<double> vk1(n_minus_1), vk0(n_minus_1), vk10(n_minus_1);
    double local_loss = 0.0;

#pragma omp for nowait
    for (int l = 0; l < L; l++) {
      int idx_offset = l * n;
      int delta_offset = l * n_minus_1;
      double cumw1 = 0.0, cumw0 = 0.0;
      double s1 = 0.0, s0 = 0.0, s10 = 0.0;

      for (int i = 0; i < n_minus_1; i++) {
        int orig = p_idx[idx_offset + i] - 1;
        if (p_A[orig] == 1) cumw1 += p_w[orig];
        else cumw0 += p_w[orig];

        double target = (i + 1.0) / n;
        double e1 = cumw1 - target;
        double e0 = cumw0 - target;
        double e10 = cumw1 - cumw0;
        double d = p_delta[delta_offset + i];

        local_loss += d * (e1 * e1 + e0 * e0 + alpha * e10 * e10);
        vk1[i] = 2.0 * d * e1;
        vk0[i] = 2.0 * d * e0;
        vk10[i] = 2.0 * d * e10 * alpha;
        s1 += vk1[i];
        s0 += vk0[i];
        s10 += vk10[i];
      }

      double c1 = 0.0, c0 = 0.0, c10 = 0.0;
      for (int i = 0; i < n; i++) {
        int orig = p_idx[idx_offset + i] - 1;
        if (p_A[orig] == 1) {
          local_grad[orig] += (s1 - c1) + (s10 - c10);
        } else {
          local_grad[orig] += (s0 - c0) - (s10 - c10);
        }
        if (i < n_minus_1) {
          c1 += vk1[i];
          c0 += vk0[i];
          c10 += vk10[i];
        }
      }
    }

#pragma omp critical
    {
      for (int i = 0; i < n; i++) p_grad[i] += local_grad[i];
      total_loss += local_loss;
    }
  }

  double invL = 1.0 / L;
  double w_sq_sum = 0.0;
  for (int i = 0; i < n; i++) {
    p_grad[i] = p_grad[i] * invL + 2.0 * lambda * p_w[i];
    w_sq_sum += p_w[i] * p_w[i];
  }
  double loss = total_loss * invL;
  return List::create(Named("grad") = grad,
                      Named("Jhat") = loss + lambda * w_sq_sum,
                      Named("loss") = loss);
}

// ============================================================
// Design-stage scale S_n = 3/L * sum_l (X_(n)^theta_l - X_(1)^theta_l)^2 + 2*lambda
// Computed once, before any outcome is touched.
// ============================================================
double compute_Sn_cpp(NumericMatrix Delta, int L, double lambda) {
  int n_minus_1 = Delta.nrow();
  double sum_sq_range = 0.0;
  for (int l = 0; l < L; l++) {
    double range_l = 0.0;
    for (int i = 0; i < n_minus_1; i++) range_l += Delta(i, l);
    sum_sq_range += range_l * range_l;
  }
  return (3.0 / L) * sum_sq_range + 2.0 * lambda;
}

// ============================================================
// Main Optimization Function
//
// Outcome-free certified stopping rule:
//   Gamma_t = U_t - L_t <= delta_n.
// By default the R wrapper requests
//   delta_n = delta_uniform_fraction * J_ref,
// where J_ref is the smaller objective at the uniform and supplied logistic-
// IPW candidates. The argument name is retained for backward compatibility.
// An explicit
// delta_n_override takes precedence and is used for resampling so every refit
// can retain the full-sample delta_n. Y is not read until after optimization.
// ============================================================
// delta_n_override: if >= 0, use this fixed tolerance.
// delta_uniform_fraction: if > 0 and there is no override, set the tolerance
// to this fraction of the reference objective.
// Objective/gradient dispatch. p_order = 1 uses the sliced-L1 objective,
// p_order = 2 the sliced-L2 one. Both are convex in w with the same lambda
// ridge, so the same projected solver and certified-gap rule serve both.
static inline List sw_obj_grad_dispatch(int p_order,
                                        NumericVector w,
                                        IntegerMatrix idx_sorted,
                                        IntegerVector Z,
                                        NumericMatrix Delta,
                                        int L,
                                        double lambda,
                                        double alpha) {
  if (p_order == 2) {
    return sw_obj_grad_p2_cpp(w, idx_sorted, Z, Delta, lambda, alpha);
  }
  return sw_obj_grad_cpp_parallel(w, idx_sorted, Z, Delta, L, lambda, alpha);
}

// [[Rcpp::plugins(openmp)]]
// [[Rcpp::export]]
List get_sw_weights_cpp(NumericVector Y,
                        IntegerVector Z,
                        NumericMatrix X,
                        int L,
                        int max_iter = 10000,
                        double eta_step = 1.0,
                        double c_opt = 1.0,
                        double alpha = 1.0,
                        double lambda = 0.1,
                        int output = 0,
                        double delta_n_override = -1.0,
                        double delta_uniform_fraction = -1.0,
                        double proportion_to_initial_loss = -1.0,
                        bool calibration_only = false,
                        int eta_window = 10,
                        double eta_min = 1e-4,
                        double eta_max = 10.0,
                        double eta_increase = 2.0,
                        double eta_decrease = 0.5,
                        double eta_progress = 0.005,
                        double eta_gap_multiplier = 10.0,
                        Nullable<NumericVector> initial_weights = R_NilValue,
                        int p_order = 1)
{
  int n = X.nrow();
  int p = X.ncol();

  if (p_order != 1 && p_order != 2) stop("p_order must be 1 or 2.");

  if (eta_window < 4 || eta_window % 2 != 0) {
    stop("eta_window must be an even integer of at least 4.");
  }
  if (!R_finite(eta_step) || !R_finite(eta_min) || !R_finite(eta_max) || eta_min <= 0.0 ||
      eta_max < eta_min || eta_step < eta_min || eta_step > eta_max) {
    stop("eta_step must lie between eta_min and eta_max.");
  }
  if (!R_finite(eta_increase) || eta_increase <= 1.0 ||
      !R_finite(eta_decrease) || eta_decrease <= 0.0 || eta_decrease >= 1.0 ||
      !R_finite(eta_progress) || eta_progress < 0.0 ||
      !R_finite(eta_gap_multiplier) || eta_gap_multiplier < 1.0) {
    stop("Invalid adaptive-eta settings.");
  }

  // 1. Pre-processing: Generate Random Projections (design-stage, no Y)
  NumericMatrix proj(p, L);
  for (int j = 0; j < L; j++) {
    double sum_sq = 0.0;
    for (int i = 0; i < p; i++) {
      proj(i, j) = R::rnorm(0, 1);
      sum_sq += proj(i, j) * proj(i, j);
    }
    double norm = sqrt(sum_sq);
    for (int i = 0; i < p; i++) proj(i, j) /= norm;
  }

  IntegerMatrix idx_sorted(n, L);
  NumericMatrix Delta(n - 1, L);
#pragma omp parallel
{
  std::vector<std::pair<double, int>> col_data(n);
#pragma omp for
  for (int l = 0; l < L; l++) {
    for (int i = 0; i < n; i++) {
      double val = 0.0;
      for (int k = 0; k < p; k++) val += X(i, k) * proj(k, l);
      col_data[i] = std::make_pair(val, i);
    }
    std::sort(col_data.begin(), col_data.end());
    for (int i = 0; i < n; i++) {
      idx_sorted(i, l) = col_data[i].second + 1;
      if (i < n - 1) Delta(i, l) = col_data[i + 1].first - col_data[i].first;
    }
  }
}

// 2. Construct weights that are uniform within each treatment arm.
NumericVector w_uniform(n);
std::vector<int> idx_1, idx_0;
for(int i = 0; i < n; i++) {
  if (Z[i] == 1) idx_1.push_back(i);
  else idx_0.push_back(i);
}
for(int idx : idx_1) w_uniform[idx] = 1.0 / idx_1.size();
for(int idx : idx_0) w_uniform[idx] = 1.0 / idx_0.size();

double q_constant = 0.0;
for (int i = 0; i < n; i++) {
  q_constant += w_uniform[i] * w_uniform[i];
}

// 3. Optionally calibrate lambda from the unpenalized loss at uniform weights.
bool adaptive_lambda = proportion_to_initial_loss > 0.0;
double initial_loss = NA_REAL;
if (adaptive_lambda) {
  List initial = sw_obj_grad_dispatch(p_order, w_uniform, idx_sorted, Z, Delta,
                                      L, 0.0, alpha);
  initial_loss = as<double>(initial["Jhat"]);
  lambda = proportion_to_initial_loss * initial_loss / q_constant;
  if (!R_finite(lambda) || lambda <= 0.0) {
    stop("Initial-loss calibration produced a nonpositive or nonfinite lambda.");
  }
}

if (calibration_only) {
  if (!adaptive_lambda) {
    stop("Calibration-only mode requires proportion_to_initial_loss > 0.");
  }
  return List::create(
    Named("lambda") = lambda,
    Named("initial_loss") = initial_loss,
    Named("q_constant") = q_constant,
    Named("proportion_to_initial_loss") = proportion_to_initial_loss
  );
}

// 4. Evaluate both start candidates with exactly the projections and ridge
// used by this optimization. Use the lower-objective candidate and define
// J_ref as the smaller objective.
List uniform_eval = sw_obj_grad_dispatch(p_order,
  w_uniform, idx_sorted, Z, Delta, L, lambda, alpha);
double uniform_objective = as<double>(uniform_eval["Jhat"]);
NumericVector uniform_grad = as<NumericVector>(uniform_eval["grad"]);
if (!adaptive_lambda) initial_loss = uniform_objective - lambda * q_constant;

double logistic_objective = NA_REAL;
NumericVector w = clone(w_uniform);
NumericVector start_grad = clone(uniform_grad);
double start_objective = uniform_objective;
std::string initial_source = "uniform";

if (initial_weights.isNotNull()) {
  NumericVector candidate(initial_weights);
  if (candidate.size() != n) {
    stop("initial_weights must have one value per observation.");
  }
  double sum_1 = 0.0, sum_0 = 0.0;
  for (int i = 0; i < n; i++) {
    if (!R_finite(candidate[i]) || candidate[i] < 0.0) {
      stop("initial_weights must be finite and nonnegative.");
    }
    if (Z[i] == 1) sum_1 += candidate[i];
    else sum_0 += candidate[i];
  }
  if (!R_finite(sum_1) || !R_finite(sum_0) ||
      sum_1 <= 0.0 || sum_0 <= 0.0) {
    stop("initial_weights must have positive weight in each treatment arm.");
  }
  for (int idx : idx_1) candidate[idx] /= sum_1;
  for (int idx : idx_0) candidate[idx] /= sum_0;

  List candidate_eval = sw_obj_grad_dispatch(p_order,
    candidate, idx_sorted, Z, Delta, L, lambda, alpha);
  logistic_objective = as<double>(candidate_eval["Jhat"]);
  if (R_finite(logistic_objective) &&
      logistic_objective < uniform_objective) {
    w = clone(candidate);
    start_grad = as<NumericVector>(candidate_eval["grad"]);
    start_objective = logistic_objective;
    initial_source = "logistic";
  }
}
double reference_objective = std::min(
  uniform_objective,
  R_finite(logistic_objective) ? logistic_objective : uniform_objective);

double S_n = compute_Sn_cpp(Delta, L, lambda);
double delta_n = (delta_n_override >= 0.0) ? delta_n_override
  : c_opt * lambda / (double(n));
if (delta_n_override < 0.0 && delta_uniform_fraction > 0.0) {
  delta_n = delta_uniform_fraction * reference_objective;
}

std::vector<double> v(n);
NumericVector w_best = clone(w);
double U = start_objective;
double Lbound = R_NegInf;
int final_iter = max_iter;
bool converged = false;
std::vector<double> gamma_history;
gamma_history.reserve(max_iter);
std::vector<double> objective_history;
objective_history.reserve(max_iter);
std::vector<double> eta_history;
eta_history.reserve(max_iter);
std::vector<double> eta_progress_history;
eta_progress_history.reserve(max_iter / eta_window);
double curr_eta = eta_step;
int eta_increases = 0;
int eta_decreases = 0;

// 5. Main Optimization Loop (outcome-independent: Y is never touched here)
for (int iter = 1; iter <= max_iter; iter++) {
  // Reuse the selected candidate's evaluation at the first iterate.
  NumericVector grad;
  double Jt;
  if (iter == 1) {
    grad = clone(start_grad);
    Jt = start_objective;
  } else {
    List og = sw_obj_grad_dispatch(p_order, w, idx_sorted, Z, Delta, L,
                                   lambda, alpha);
    grad = as<NumericVector>(og["grad"]);
    Jt = as<double>(og["Jhat"]);
  }

  // u^(t) = P_Omega( w^(t) - g^(t) / (2*lambda) )
  for (int i = 0; i < n; i++) v[i] = w[i] - grad[i] / (2.0 * lambda);
  NumericVector u(n);
  project_simplex_cpp(v, idx_1, 1.0, u);
  project_simplex_cpp(v, idx_0, 1.0, u);

  // Strong-convexity lower bound:
  // lower_t = Jhat(w^(t)) + <g^(t), u^(t)-w^(t)> + lambda*||u^(t)-w^(t)||_2^2
  double inner = 0.0, sq_norm = 0.0;
  for (int i = 0; i < n; i++) {
    double diff = u[i] - w[i];
    inner += grad[i] * diff;
    sq_norm += diff * diff;
  }
  double lower_t = Jt + inner + lambda * sq_norm;
  if (lower_t > Lbound) Lbound = lower_t;

  // Track best primal iterate
  if (Jt < U) {
    U = Jt;
    w_best = clone(w);
  }

  double Gamma_t = U - Lbound;
  gamma_history.push_back(Gamma_t);
  objective_history.push_back(Jt);

  if (output == 1 && (iter % 200 == 0)) {
    Rcpp::Rcout << "iter " << iter << "  Gamma_t=" << Gamma_t
                << "  U=" << U << "  Lbound=" << Lbound
                << "  eta=" << curr_eta << "\n";
  }

  if (Gamma_t <= delta_n) {
    eta_history.push_back(curr_eta);
    if (output == 1) {
      Rcpp::Rcout << "\n>>> Certified-Gap Convergence Triggered! <<<\n";
      Rcpp::Rcout << "Gamma_t: " << Gamma_t << " <= delta_n: " << delta_n << "\n";
      Rcpp::Rcout << "Converged at Iteration: " << iter << "\n";
    }
    final_iter = iter;
    converged = true;
    break;
  }

  bool restart_from_best = false;
  if (iter % eta_window == 0) {
    int start = iter - eta_window;
    int half = eta_window / 2;
    double old_mean = 0.0;
    double new_mean = 0.0;
    for (int k = start; k < start + half; k++) old_mean += objective_history[k];
    for (int k = start + half; k < iter; k++) new_mean += objective_history[k];
    old_mean /= half;
    new_mean /= half;
    double r_t = (old_mean - new_mean) /
      std::max(std::abs(old_mean), 1e-12);
    eta_progress_history.push_back(r_t);

    int increases = 0;
    for (int k = start + 1; k < iter; k++) {
      double tolerance = 1e-10 * std::max(1.0, std::abs(objective_history[k - 1]));
      if (objective_history[k] > objective_history[k - 1] + tolerance) increases++;
    }
    int oscillation_limit = std::max(3, eta_window / 3);
    bool unstable = (r_t < 0.0) || (increases >= oscillation_limit);

    if (unstable) {
      double new_eta = std::max(eta_min, curr_eta * eta_decrease);
      if (new_eta < curr_eta) eta_decreases++;
      curr_eta = new_eta;
      w = clone(w_best);
      restart_from_best = true;
    } else if (r_t < eta_progress &&
               Gamma_t > eta_gap_multiplier * delta_n) {
      double new_eta = std::min(eta_max, curr_eta * eta_increase);
      if (new_eta > curr_eta) eta_increases++;
      curr_eta = new_eta;
    }

    if (output == 1 && (unstable || r_t < eta_progress)) {
      Rcpp::Rcout << "eta check at iter " << iter << ": r_t=" << r_t
                  << ", increases=" << increases
                  << ", eta=" << curr_eta;
      if (restart_from_best) Rcpp::Rcout << " (restored best iterate)";
      Rcpp::Rcout << "\n";
    }
  }

  eta_history.push_back(curr_eta);
  if (restart_from_best) continue;

  // Update: w^(t+1) = P_Omega( w^(t) - eta_t * g^(t) )
  for (int i = 0; i < n; i++) v[i] = w[i] - curr_eta * grad[i];
  project_simplex_cpp(v, idx_1, 1.0, w);
  project_simplex_cpp(v, idx_0, 1.0, w);
}

if (!converged && output == 1) {
  Rcpp::Rcout << "\n>>> WARNING: max_iter reached WITHOUT certified convergence. <<<\n";
  Rcpp::Rcout << "Achieved gap: " << gamma_history.back() << "  (target delta_n: " << delta_n << ")\n";
  Rcpp::Rcout << "Run should be labeled nonconverged; report the achieved gap.\n";
}

// 6. Outcome is read ONLY now, after weights are frozen at w_best.
double ate = 0.0;
for (int i = 0; i < n; i++) {
  if (Z[i] == 1) ate += w_best[i] * Y[i];
  else ate -= w_best[i] * Y[i];
}

return List::create(
  Named("weights") = w_best,
  Named("ate") = ate,
  Named("converged") = converged,
  Named("converged_iter") = final_iter,
  Named("delta_n") = delta_n,
  Named("lambda") = lambda,
  Named("initial_loss") = initial_loss,
  Named("q_constant") = q_constant,
  Named("proportion_to_initial_loss") = proportion_to_initial_loss,
  Named("uniform_objective") = uniform_objective,
  Named("logistic_objective") = logistic_objective,
  Named("reference_objective") = reference_objective,
  Named("initial_source") = initial_source,
  Named("S_n") = S_n,
  Named("final_gap") = gamma_history.back(),
  Named("gamma_history") = gamma_history,
  Named("objective_history") = objective_history,
  Named("eta_initial") = eta_step,
  Named("eta_final") = curr_eta,
  Named("eta_increases") = eta_increases,
  Named("eta_decreases") = eta_decreases,
  Named("eta_history") = eta_history,
  Named("eta_progress_history") = eta_progress_history
);
}
