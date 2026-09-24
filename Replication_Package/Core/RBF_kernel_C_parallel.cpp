// [[Rcpp::depends(RcppParallel)]]
#include <Rcpp.h>
#include <omp.h>
#include <RcppParallel.h>
using namespace RcppParallel;


struct RBFKernel : public Worker
{
  // source matrix
  const RMatrix<double> X;
  
  const int c;
  const RVector<int> set_c;
  
  // destination matrix
  RMatrix<double> out;
  
  // initialize with source and destination
  RBFKernel(const Rcpp::NumericMatrix X, const int c_, Rcpp::IntegerVector set_c_, Rcpp::NumericMatrix out) 
    : X(X), c(c_), set_c(set_c_), out(out) {}
  
  // calculate the range of elements requested
  void operator()(std::size_t begin, std::size_t end) {
    int p = X.ncol();
    // default value following Hazlett (2020)
    double gamma = 1/double(2*p);
    for (std::size_t i = begin; i < end; ++i) {
      for (std::size_t j = 0; j < c; ++j) {
        int cj = set_c[j]-1;
        double dist = 0;
        for (int k = 0; k < p; ++k) {
          double xi = X(i, k), xj = X(cj, k);
          dist += (xi - xj)*(xi - xj);
        }
        out(i, j) = exp(-gamma*dist);
      }
    }
  }
};

// [[Rcpp::export]]
Rcpp::NumericMatrix RBF_kernel_C_parallel(Rcpp::NumericMatrix X,
                                          int c, 
                                          Rcpp::IntegerVector set_c) {
  
  // allocate the output matrix
  Rcpp::NumericMatrix out(X.nrow(), c);
  
  // RBFKernel functor (pass input and output matrixes)
  RBFKernel rbfkernel(X, c, set_c, out);
  
  // call parallelFor to do the work
  parallelFor(0, X.nrow(), rbfkernel);
  
  // return the output matrix
  return out;
}