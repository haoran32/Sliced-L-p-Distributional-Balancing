## OSQP kernel SBW with a Nystrom approximation.
## The kernel construction follows the kbal implementation by Chad Hazlett.

makeK <- function(allx, useasbases=NULL, b=NULL, linkernel = FALSE, scale = TRUE){
  N=nrow(allx)
  if(is.null(useasbases)) {useasbases = rep(1, N)}
  
  if (is.null(b)){ b=2*ncol(allx) }
  
  if(scale) {
    allx = scale(allx)
  } 
  bases = allx[useasbases==1, ]
  
  if(linkernel == TRUE) {
    K = allx
  } else {
    if(sum(useasbases) == N) {
      K = kernel_parallel(X = allx, b = b)
    } else {
      K = kernel_parallel_2(X = allx, Y = bases, b = b)
    }
  }
  return(K)
}

kernel_parallel <- function(X, b) {
  .Call('_kbal_kernel_parallel', PACKAGE = 'kbal', X, b)
}

kernel_parallel_2 <- function(X, Y, b) {
  .Call('_kbal_kernel_parallel_2', PACKAGE = 'kbal', X, Y, b)
}

kernel_parallel_old <- function(X, Y, b) {
  .Call('_kbal_kernel_parallel_old', PACKAGE = 'kbal', X, Y, b)
}


## Gaussian kernel basis using either the full matrix (Hazlett, 2020) or a
## Nystrom approximation (Wang, 2019).

kernel.basis <- function(X,A,Y, 
                         kernel.approximation=TRUE,
                         dim.reduction=FALSE,
                         c = NULL, l=NULL, s=NULL, gamma=NULL, U.mat = NULL) {
  n <- nrow(X)
  if (kernel.approximation) {
    if (is.null(c)) {
      c = ifelse(n<1e+5, 100, 250) 
    }
    if (is.null(l)) {
      l=round(c/2)
      s=round(l/2)
    }
    if (is.null(gamma)) {
      gamma = 1/(2*ncol(X)) 
    }
    
    set_c = sample(x = 1:n, size = c, replace = FALSE)
    set_c = sort(set_c)
    X <- scale(X)
    C <- RBF_kernel_C_parallel(X, c, set_c)
    W <- C[set_c,]
    ## Keep the first l values to match a rank-l truncated SVD.
    SVD_W <- base::svd(W, nu = l, nv = 0L)
    SVD_W$d <- SVD_W$d[seq_len(l)]
    if (dim.reduction) {
      R <- C %*% SVD_W$u %*% diag(1/sqrt(SVD_W$d))
      SVD_R <- base::svd(R, nu = 0L, nv = s)
      X_ <- R %*% SVD_R$v # B
    } else {
      X_ <- C %*% SVD_W$u # %*% diag(1/SVD_W$d)
    }
  } else{
    if (is.null(gamma)) { gamma <- 1 / (2 * ncol(X)) }
    gram.mat <- exp(-gamma * as.matrix(dist(X))^2)
    
    ## Retain the c eigenpairs with the largest absolute eigenvalues.
    res <- base::eigen(gram.mat, symmetric = TRUE)
    keep <- order(abs(res$values), decreasing = TRUE)[seq_len(c)]
    res$values <- res$values[keep]
    res$vectors <- res$vectors[, keep, drop = FALSE]
    X_ <- if(is.null(U.mat)) {
      res$vectors %*% diag(1/sqrt(res$values))
    } else{
      U.mat %*% diag(1/sqrt(res$values))
    }
  }
  return(X_)
}

## Power-series basis through order K, with optional interactions.

power.basis <- function(X,A,Y,
                        K=2, interactions=FALSE) {
  for (k in 1:K){
    X.k <- X^k
    colnames(X.k) <- paste(colnames(X),".",k,sep = "")
    X_ <- cbind(X_, X.k)
  }
  
  if (interactions==TRUE & K >=2) {
    for (k in 2:K){
      indx <- combn(colnames(X),k)
      int <- as.data.frame(do.call(cbind,
                                   lapply(split(indx, col(indx)), 
                                          function(x) rowProds(as.matrix(X[,x])))
      ))
      colnames(int) <- apply(indx, 2, function(x) paste(x,collapse="."))
      X_ <- cbind(X_, int)
    }
  }
  return(X_)
}

## OSQP kernel SBW.

osqp_kernel_sbw <- function(X,A,Y,
                            delta.v=0.005, 
                            X_=NULL,
                            osqp.setting=NULL,
                            basis="kernel", kernel.approximation=TRUE,
                            c = NULL, l=NULL, gamma=NULL, U.mat = NULL,
                            dim.reduction=FALSE, s=NULL,
                            K=2, interactions=FALSE) {
  
  res.list <- list()
  if (is.null(X_)) {
    if (basis=="kernel"){
      X_ <- kernel.basis(X,A,Y, 
                         kernel.approximation=kernel.approximation, 
                         c=c, l=l, gamma=gamma, U.mat=U.mat,
                         s=s, dim.reduction=dim.reduction)
    } 
    
    else if (basis=="power") {
      X_ <- power.basis(X,A,Y, 
                        K=K, interactions=interactions)
    } 
    
    else {
      stop("not available yet")
    }
  }
  
  nX <- ncol(X_)
  n1 <- sum(A); n0 <- sum(1-A)
  Xt <- X_[A==1,]; Xc <- X_[A==0,]
  Yt <- Y[A==1]; Yc <- Y[A==0]
  n <- n1 + n0 
  target_mean <- colMeans(X_) 
  P.mat <- as(.symDiagonal(n=n, x=1.), "dgCMatrix")
  q.vec <- c(rep(-1./n1, n1), rep(-1./n0, n0))
  A.mat <- Matrix(rbind(
    c(rep(1., n1), rep(0., n0)),                   
    c(rep(0., n1), rep(1., n0)),                   
    P.mat,                                       
    cbind(t(Xt), matrix(0., nrow=nX, ncol=n0)),   
    cbind(matrix(0., nrow=nX, ncol=n1), t(Xc))    
  ), sparse = TRUE)
  
  if (is.null(osqp.setting)) {
    settings <- osqpSettings(alpha = 1.5, verbose = FALSE)  
  } else {
    settings <- osqp.setting
  }
  
  for (j in 1:length(delta.v)) {

    l.vec <- c(1., 1.,                             
               rep(0., n),                        
               target_mean - delta.v[j],           
               target_mean - delta.v[j])          
    
    u.vec <- c(1., 1.,                             
               rep(1., n),                        
               target_mean + delta.v[j],          
               target_mean + delta.v[j])          
    
    if (j==1) {
      model <- osqp(P.mat, q.vec, A.mat, l.vec, u.vec, settings)
    } else {
      model$Update(l = l.vec, u = u.vec)
    }
    
    res <- model$Solve()
    if (res$info$status != "solved") {
      warning(res$info$status)
    }
    
    w_all <- res$x
    w_t <- w_all[1:n1]          
    w_c <- w_all[(n1+1):n]      
    y_hat_t <- sum(w_t * Yt)
    y_hat_c <- sum(w_c * Yc)
    ate_estimate <- y_hat_t - y_hat_c
    res.list[[j]] <- list(w_t=w_t, w_c=w_c, ate=ate_estimate, t=res$info$solve_time)
  }
  return(res.list)
}   
