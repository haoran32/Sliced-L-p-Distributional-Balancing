#' Super Learner Wrapper
#'
#' Fit an ensemble of machine learning algorithms using the Super Learner
#' algorithm (van der Laan et al., 2007) with a configurable library of
#' base learners.
#'
#' @param Data A data frame containing the outcome and covariates.
#' @param locY Column index for the outcome variable.
#' @param locX Column index (or indices) for the covariates.
#' @param Ydist A family object (e.g., \code{gaussian()}, \code{binomial()}).
#' @param SL.list Integer vector selecting which learner groups to include (1--9).
#'   Available groups: 1 = GLM, 2 = lasso/ridge, 3 = earth,
#'   4 = GAM, 5 = xgboost, 6 = polynomial spline, 7 = random forest,
#'   8 = gbm, 9 = 1-layer MLP.
#' @param obsWeights Optional observation weights.
#' @param PS.thr Propensity score threshold (default 1e-3).
#' @param CVlist Optional list of cross-validation fold assignments.
#'
#' @details
#' \code{MySL} is a convenience wrapper of the \code{SuperLearner}
#' algorithm (van der Laan et al., 2007) implemented in the
#' \code{SuperLearner} package (Polley et al., 2025). The function builds
#' a library of candidate learners from the groups specified in \code{SL.list}:
#' \enumerate{
#'   \item \strong{GLM}: generalized linear model.
#'   \item \strong{Lasso/Ridge}: elastic net via \code{glmnet} with
#'     \eqn{\alpha \in \{0, 0.5, 1\}}.
#'   \item \strong{MARS}: multivariate adaptive regression splines via
#'     \code{earth} with polynomial degrees 1--5.
#'   \item \strong{GAM}: generalized additive model via \code{mgcv} with
#'     spline degrees 1--5.
#'   \item \strong{XGBoost}: gradient boosted trees via \code{xgboost} with
#'     a grid over the number of trees and tree depth.
#'   \item \strong{Polynomial spline}: polynomial MARS via \code{polymars}
#'     with 2--4 knots.
#'   \item \strong{Random forest}: via \code{ranger} with a grid over the
#'     number of trees and \code{mtry}.
#'   \item \strong{GBM}: gradient boosted model via \code{caret} with
#'     \code{gbm} backend.
#'   \item \strong{1-layer MLP}: single hidden-layer neural network via
#'     \code{caret} with \code{mlpML} backend.
#' }
#' The Super Learner computes the optimal convex combination of these base
#' learners using 5-fold cross-validation by default.
#'
#' For Poisson outcomes (\code{Ydist = poisson()}), only learner groups
#' 1 (GLM), 2 (lasso/ridge), 4 (GAM), and 5 (xgboost) are used.
#'
#' @return A fitted \code{SuperLearner} object.
#'
#' @references
#' \itemize{
#'   \item Polley, E., LeDell, E., Kennedy, C., Lendle, S., & van der Laan, M. (2025).
#'     SuperLearner: Super Learner Prediction. R package version 2.0-40.
#'   \item van der Laan, M. J., Polley, E. C., & Hubbard, A. E. (2007).
#'     Super learner.
#'     \emph{Statistical Applications in Genetics and Molecular Biology}, 6(1).
#' }
#'
#' @export
MySL <- function( Data, locY, locX, Ydist=stats::gaussian(),
                  SL.list=c(1:9), obsWeights=NULL,
                  PS.thr = 10^(-3), CVlist=NULL ){

  ## Poisson c(2,4,5)
  if(Ydist$family=="poisson"){
    SL.list <- intersect(c(1,2,4,5),SL.list)
  }

  Learners <- list()

  ##############
  # Caret Based
  ##############

  SL.caret.SLP <- function (Y, X, newX, family, obsWeights, method = "mlpML",
                            L1,L2,L3,decay,
                            trControl = caret::trainControl(method = "none"),
                            metric = ifelse(family$family == "gaussian", "RMSE", "Accuracy"), ...)
  {
    if (family$family == "gaussian") {
      fit.train <- caret::train(x = X, y = Y, weights = obsWeights,
                                metric = metric, method = method,
                                tuneGrid = expand.grid(layer1=L1,layer2=L2,layer3=L3),
                                preProc =  NULL,
                                hiddenActFunc = "Act_Logistic",linOut=TRUE,
                                learnFuncParams=decay,
                                trControl = trControl)
      pred <- stats::predict(fit.train, newdata = newX, type = "raw")
    }
    if (family$family == "binomial") {
      Y.f <- as.factor(Y)
      levels(Y.f) <- c("A0", "A1")
      fit.train <- caret::train(x = X, y = Y.f, weights = obsWeights,
                                metric = metric, method = method,
                                tuneGrid = expand.grid(layer1=L1,layer2=L2,layer3=L3),
                                hiddenActFunc = "Act_Identity",
                                preProc =  NULL,
                                trControl = trControl)
      pred <- stats::predict(fit.train, newdata = newX, type = "prob")[,2]
    }
    fit <- list(object = fit.train)
    out <- list(pred = pred, fit = fit)
    class(out$fit) <- c("SL.caret")
    return(out)
  }


  SL.caret.MLP <- function (Y, X, newX, family, obsWeights, method = "mlpML",
                            L1,L2,L3,decay,
                            trControl = caret::trainControl(method = "none"),
                            metric = ifelse(family$family == "gaussian", "RMSE", "Accuracy"), ...)
  {
    if (family$family == "gaussian") {
      fit.train <- caret::train(x = X, y = Y, weights = obsWeights,
                                metric = metric, method = method,
                                tuneGrid = expand.grid(layer1=L1,layer2=L2,layer3=L3),
                                learnFuncParams=decay,
                                hiddenActFunc = "Act_Logistic",linOut=TRUE,
                                preProc =  NULL,
                                trControl = trControl)
      pred <- stats::predict(fit.train, newdata = newX, type = "raw")
    }
    if (family$family == "binomial") {
      Y.f <- as.factor(Y)
      levels(Y.f) <- c("A0", "A1")
      fit.train <- caret::train(x = X, y = Y.f, weights = obsWeights,
                                metric = metric, method = method,
                                tuneGrid = expand.grid(layer1=L1,layer2=L2,layer3=L3),
                                hiddenActFunc = "Act_Identity",
                                preProc =  NULL,
                                trControl = trControl)
      pred <- stats::predict(fit.train, newdata = newX, type = "prob")[,2]
    }
    fit <- list(object = fit.train)
    out <- list(pred = pred, fit = fit)
    class(out$fit) <- c("SL.caret")
    return(out)
  }



  SL.caret.gbm <- function (Y, X, newX, family, obsWeights, method = "gbm",
                            ntree,intdepth,sh,nmn,
                            trControl = caret::trainControl(method = "none"),
                            metric = ifelse(family$family == "gaussian", "RMSE", "Accuracy"), ...)
  {
    if (family$family == "gaussian") {

      fit.train <- caret::train(x = X, y = Y, weights = obsWeights,
                                metric = metric, method = method,
                                tuneGrid = expand.grid(n.trees=ntree,interaction.depth=intdepth,shrinkage=sh,n.minobsinnode=nmn),
                                preProc =  NULL,
                                trControl = trControl)
      pred <- stats::predict(fit.train, newdata = newX, type = "raw")
    }
    if (family$family == "binomial") {
      Y.f <- as.factor(Y)
      levels(Y.f) <- c("A0", "A1")
      fit.train <- caret::train(x = X, y = Y.f, weights = obsWeights,
                                metric = metric, method = method,
                                tuneGrid = expand.grid(n.trees=ntree,interaction.depth=intdepth,shrinkage=sh,n.minobsinnode=nmn),
                                preProc =  NULL,
                                trControl = trControl)
      pred <- stats::predict(fit.train, newdata = newX, type = "prob")[,2]
    }
    fit <- list(object = fit.train)
    out <- list(pred = pred, fit = fit)
    class(out$fit) <- c("SL.caret")
    return(out)
  }


  #############################################
  # SL-based
  #############################################

  SL.new.earth <- function (Y, X, newX, family, obsWeights, id, degree = 2, penalty = 3,
                            nk = max(21, 2 * ncol(X) + 1), pmethod = "backward",
                            nfold = 0, ncross = 1, minspan = 0, endspan = 0, ...)
  {
    if (family$family == "gaussian") {
      fit.earth <- earth::earth(x = X, y = Y, degree = degree, weights=obsWeights,
                                nk = nk, penalty = penalty, pmethod = pmethod, nfold = nfold,
                                ncross = ncross, minspan = minspan, endspan = endspan)
    }
    if (family$family == "binomial") {
      fit.earth <- earth::earth(x = X, y = Y, degree = degree, weights=obsWeights,
                                nk = nk, penalty = penalty, pmethod = pmethod, nfold = nfold,
                                ncross = ncross, minspan = minspan, endspan = endspan,
                                glm = list(family = stats::binomial))
    }
    pred <- stats::predict(fit.earth, newdata = newX, type = "response")
    fit <- list(object = fit.earth)
    out <- list(pred = pred, fit = fit)
    class(out$fit) <- c("SL.earth")
    return(out)
  }

  SL.new.xgboost <- function (Y, X, newX, family, obsWeights, id, ntrees = 1000,
                              max_depth = 4, shrinkage = 0.1, minobspernode = 10, params = list(),
                              nthread = 1, verbose = 0, save_period = NULL, ...)
  {
    if (!is.matrix(X)) {
      X = stats::model.matrix(~. - 1, X)
    }

    xgb_ver <- as.numeric(sub("^(\\d+\\.\\d+).*", "\\1", utils::packageVersion("xgboost")))

    if (xgb_ver >= 2.1) {
      if (family$family == "gaussian") {
        model = xgboost::xgboost(x = X, y = Y, weight = obsWeights,
                                 objective = "reg:squarederror",
                                 nrounds = ntrees, max_depth = max_depth,
                                 min_child_weight = minobspernode,
                                 learning_rate = shrinkage, nthread = nthread,
                                 verbose = verbose)
      }
      if (family$family == "binomial") {
        y_fac <- factor(if (is.factor(Y)) as.numeric(Y) - 1 else Y, levels = c(0, 1))
        model = xgboost::xgboost(x = X, y = y_fac, weight = obsWeights,
                                 nrounds = ntrees, max_depth = max_depth,
                                 min_child_weight = minobspernode,
                                 learning_rate = shrinkage, nthread = nthread)
      }
      if (family$family == "multinomial") {
        y_num = as.numeric(as.factor(Y)) - 1
        model = xgboost::xgboost(x = X, y = y_num, weight = obsWeights,
                                 objective = "multi:softmax",
                                 nrounds = ntrees, max_depth = max_depth,
                                 min_child_weight = minobspernode,
                                 learning_rate = shrinkage,
                                 num_class = length(unique(Y)), nthread = nthread,
                                 verbose = verbose)
      }
      if (family$family == "poisson") {
        model = xgboost::xgboost(x = X, y = Y, weight = obsWeights,
                                 objective = "count:poisson",
                                 nrounds = ntrees, max_depth = max_depth,
                                 min_child_weight = minobspernode,
                                 learning_rate = shrinkage, nthread = nthread,
                                 verbose = verbose)
      }
    } else {
      if (family$family == "gaussian") {
        xgmat = xgboost::xgb.DMatrix(data = X, label = Y, weight = obsWeights)
        model = xgboost::xgboost(data = xgmat, objective = "reg:squarederror",
                                 nrounds = ntrees, max_depth = max_depth,
                                 min_child_weight = minobspernode,
                                 eta = shrinkage, verbose = verbose, nthread = nthread,
                                 params = params, save_period = save_period)
      }
      if (family$family == "binomial") {
        y_num = if (is.factor(Y)) as.numeric(Y) - 1 else as.numeric(Y)
        xgmat = xgboost::xgb.DMatrix(data = X, label = y_num, weight = obsWeights)
        model = xgboost::xgboost(data = xgmat, objective = "binary:logistic",
                                 nrounds = ntrees, max_depth = max_depth,
                                 min_child_weight = minobspernode,
                                 eta = shrinkage, verbose = verbose, nthread = nthread,
                                 params = params, save_period = save_period)
      }
      if (family$family == "multinomial") {
        y_num = as.numeric(as.factor(Y)) - 1
        xgmat = xgboost::xgb.DMatrix(data = X, label = y_num, weight = obsWeights)
        model = xgboost::xgboost(data = xgmat, objective = "multi:softmax",
                                 nrounds = ntrees, max_depth = max_depth,
                                 min_child_weight = minobspernode,
                                 eta = shrinkage, verbose = verbose,
                                 num_class = length(unique(Y)),
                                 nthread = nthread, params = params, save_period = save_period)
      }
      if (family$family == "poisson") {
        xgmat = xgboost::xgb.DMatrix(data = X, label = Y, weight = obsWeights)
        model = xgboost::xgboost(data = xgmat, objective = "count:poisson",
                                 nrounds = ntrees, max_depth = max_depth,
                                 min_child_weight = minobspernode,
                                 eta = shrinkage, verbose = verbose,
                                 nthread = nthread, params = params, save_period = save_period)
      }
    }

    if (!is.matrix(newX)) {
      newX = stats::model.matrix(~. - 1, newX)
    }
    pred = stats::predict(model, newdata = newX)
    fit = list(object = model)
    class(fit) = c("SL.xgboost")
    out = list(pred = pred, fit = fit)
    return(out)
  }



  ## Build a local environment that contains all learner and screen functions
  ## so SuperLearner can find them by name.
  sl_env <- list2env(list(
    ## Screen functions
    All               = SuperLearner::All,
    ## Base learners from SuperLearner
    SL.glm            = SuperLearner::SL.glm,
    SL.glmnet         = SuperLearner::SL.glmnet,
    SL.gam            = SuperLearner::SL.gam,
    SL.polymars       = SuperLearner::SL.polymars,
    SL.ranger         = SuperLearner::SL.ranger,
    SL.nnet           = SuperLearner::SL.nnet,
    SL.xgboost        = SuperLearner::SL.xgboost,
    ## Predict methods
    predict.SL.glm       = SuperLearner:::predict.SL.glm,
    predict.SL.glmnet    = SuperLearner:::predict.SL.glmnet,
    predict.SL.gam       = SuperLearner:::predict.SL.gam,
    predict.SL.polymars  = SuperLearner:::predict.SL.polymars,
    predict.SL.ranger    = SuperLearner:::predict.SL.ranger,
    predict.SL.nnet      = SuperLearner:::predict.SL.nnet,
    predict.SL.xgboost   = SuperLearner:::predict.SL.xgboost,
    predict.SL.caret     = SuperLearner:::predict.SL.caret,
    predict.SL.earth     = SuperLearner:::predict.SL.earth,
    ## Custom learners defined in this function
    SL.caret.SLP      = SL.caret.SLP,
    SL.caret.MLP      = SL.caret.MLP,
    SL.caret.gbm      = SL.caret.gbm,
    SL.new.earth       = SL.new.earth,
    SL.new.xgboost     = SL.new.xgboost
  ), parent = globalenv())

  Learners[[1]] <- SuperLearner::create.Learner("SL.glm")
  TOTAL.M <- 1

  Learners[[2]] <- SuperLearner::create.Learner("SL.glmnet",tune=list(alpha=c(1,0.5,0),useMin=c(TRUE,FALSE)))

  Learners[[3]] <- SuperLearner::create.Learner("SL.new.earth",tune=list(degree=c(1,2,3,4,5)))

  Learners[[4]] <- SuperLearner::create.Learner("SL.gam",tune=list(deg.gam=c(1,2,3,4,5)))

  Learners[[5]] <- SuperLearner::create.Learner("SL.new.xgboost",tune=list(ntrees=c(100,300,500),max_depth=c(1,2,3,4)))

  Learners[[6]] <- SuperLearner::create.Learner("SL.polymars",tune=list(knots=c(2,3,4)))

  num_x_var <- length(locX)

  Learners[[7]] <- SuperLearner::create.Learner("SL.ranger",tune=list(num.trees=c(500,1000,1500),mtry=c(1,ceiling(sqrt(num_x_var)))))

  # Learners[[TOTAL.M+1]] <- SuperLearner::create.Learner("SL.nnet",tune=list(linout=c(TRUE,FALSE), decay=c(0,0.1),size=MTRY))

  Learners[[8]] <- SuperLearner::create.Learner("SL.caret.gbm",tune=list(ntree=c(100,300,500),intdepth=c(1,2,3),sh=c(0.1,0.01),nmn=10))

  Learners[[9]] <- SuperLearner::create.Learner("SL.caret.SLP", tune = list(L1=c(2,4),L2=c(0),L3=c(0),decay=10^c(-1,-3)))

  # Learners[[10]] <- SuperLearner::create.Learner("SL.caret.MLP", tune = list(L1=c(2,4),L2=MLPL,L3=MLPL,decay=MLPdecay))

  BaseLearner <- Learners[[ SL.list[1] ]]$names

  if( length(SL.list)>1 ){
    for( METHOD in 2:length(SL.list) ){
      BaseLearner <- c(BaseLearner, Learners[[ SL.list[METHOD] ]]$names)
    }
  }

  ## Copy all generated learner wrapper functions into sl_env
  ## and re-parent them so they can find base learners (SL.ranger, etc.)
  calling_env <- environment()
  for (lname in BaseLearner) {
    if (exists(lname, envir = calling_env, inherits = FALSE)) {
      fn <- get(lname, envir = calling_env)
      if (is.function(fn)) {
        environment(fn) <- sl_env
      }
      assign(lname, fn, envir = sl_env)
    }
  }

  if( length(locX)==1 ){
    dX <- data.frame(matrix( Data[,locX], dim(Data)[1], 1))
    colnames(dX) <- colnames(Data)[locX]
  } else {
    dX <- Data[,locX]
  }

  if(is.null(CVlist)){
    CVL <- list(V = 5)
  } else {
    CVL <- list(V = 5, shuffle = FALSE, validRows = CVlist)
  }

  if(Ydist$family!="binomial"){
    utils::capture.output( Fitted.SL <- SuperLearner::SuperLearner(Y=Data[,locY],X=dX,family=Ydist,
                                                                   SL.library=BaseLearner, cvControl = CVL, obsWeights=obsWeights,
                                                                   env = sl_env) , file=NULL )
  } else {
    utils::capture.output( Fitted.SL <- SuperLearner::SuperLearner(Y=Data[,locY],X=dX,family=Ydist,
                                                                   SL.library=BaseLearner, cvControl = CVL, obsWeights=obsWeights,
                                                                   env = sl_env) , file=NULL )
  }

  return(Fitted.SL)
}




#' @keywords internal
PS.Adjust <- function (Fitted.SL2,POS.Z,
                       Y, X, newX = NULL, family = stats::gaussian(), SL.library,
                       method = "method.NNLS", id = NULL, verbose = FALSE,
                       control = list(), cvControl = list(), obsWeights = NULL,
                       env = parent.frame()) {

  SL.library <- SL.library[POS.Z]

  .createLibrary <- function(SL.library) {
    if (is.character(SL.library)) {
      k <- length(SL.library)
      whichScreen <- matrix(1, nrow = 1, ncol = k)
      screenAlgorithm <- "All"
      library <- data.frame(predAlgorithm = SL.library, rowScreen = 1, stringsAsFactors=FALSE)
    } else if (is.list(SL.library)) {
      predNames <- sapply(SL.library, FUN = "[", 1)
      NumberScreen <- (sapply(SL.library, FUN = length) - 1)
      if (sum(NumberScreen == 0) > 0) {
        for(ii in which(NumberScreen == 0)) {
          SL.library[[ii]] <- c(SL.library[[ii]], "All")
          NumberScreen[ii] <- 1
        }
      }
      screenAlgorithmFull <- unlist(lapply(SL.library, FUN="[", -1))
      screenAlgorithm <- unique(screenAlgorithmFull)

      library <- data.frame(predAlgorithm = rep(predNames, times=NumberScreen), rowScreen = match(screenAlgorithmFull, screenAlgorithm), stringsAsFactors = FALSE)
    } else {
      stop('format for SL.library is not recognized')
    }

    out <- list(library = library, screenAlgorithm = screenAlgorithm)
    return(out)
  }
  library <- .createLibrary(SL.library)

  if (is.character(method)) {
    if (exists(method, mode = "list")) {
      method <- get(method, mode = "list")
    }
    else if (exists(method, mode = "function")) {
      method <- get(method, mode = "function")()
    }
  }
  if (!is.null(method$require)) {
    sapply(method$require, function(x) require(force(x),
                                               character.only = TRUE))
  }
  control <- do.call("SuperLearner.control", control)
  cvControl <- do.call("SuperLearner.CV.control", cvControl)
  varNames <- colnames(X)
  N <- dim(X)[1L]
  p <- dim(X)[2L]
  k <- length(POS.Z)
  Z <- Fitted.SL2$Z[,POS.Z]
  if(is.null(dim(Z))){
    Z <- matrix(Z,N,1)
  }
  libraryNames <- paste(library$library$predAlgorithm, library$screenAlgorithm[library$library$rowScreen],
                        sep = "_")

  fitLibEnv <- new.env()
  assign("fitLibrary", vector("list", length = k),
         envir = fitLibEnv)

  if (is.null(obsWeights)) {
    obsWeights <- rep(1, N)
  }
  if (is.null(newX)) {
    newX <- X
  }
  validRows <- SuperLearner::CVFolds(N = N, id = id, Y = Y, cvControl = cvControl)
  if (is.null(id)) {
    id <- seq(N)
  }
  errorsInCVLibrary <- Fitted.SL2$errorsInCVLibrary[POS.Z]

  getCoef <- method$computeCoef(Z = Z, Y = Y, libraryNames = libraryNames,
                                obsWeights = obsWeights, control = control, verbose = verbose,
                                errorsInLibrary = errorsInCVLibrary)
  coef <- getCoef$coef
  names(coef) <- libraryNames

  if (!("optimizer" %in% names(getCoef))) {
    getCoef["optimizer"] <- NA
  }
  m <- dim(newX)[1L]
  predY <- matrix(NA, nrow = m, ncol = k)
  .screenFun <- function(fun, list) {
    screen_fn = get(fun, envir = env)
    testScreen <- try(do.call(screen_fn, list))
    if (inherits(testScreen, "try-error")) {
      warning(paste("replacing failed screening algorithm,",
                    fun, ", with All() in full data", "\n "))
      out <- rep(TRUE, ncol(list$X))
    }
    else {
      out <- testScreen
    }
    return(out)
  }

  whichScreen <- sapply(library$screenAlgorithm, FUN = .screenFun,
                        list = list(Y = Y, X = X, family = family, id = id, obsWeights = obsWeights),
                        simplify = FALSE)
  whichScreen <- do.call(rbind, whichScreen)
  .predFun <- function(index, lib, Y, dataX, newX, whichScreen,
                       family, id, obsWeights, verbose, control, libraryNames) {
    pred_fn = get(lib$predAlgorithm[index], envir = env)
    testAlg <- try(do.call(pred_fn, list(Y = Y, X = subset(dataX,
                                                           select = whichScreen[lib$rowScreen[index], ], drop = FALSE),
                                         newX = subset(newX, select = whichScreen[lib$rowScreen[index],
                                         ], drop = FALSE), family = family, id = id, obsWeights = obsWeights)))
    if (inherits(testAlg, "try-error")) {
      warning(paste("Error in algorithm", lib$predAlgorithm[index],
                    " on full data", "\n  The Algorithm will be removed from the Super Learner (i.e. given weight 0) \n"))
      out <- rep.int(NA, times = nrow(newX))
    }
    else {
      out <- testAlg$pred
      if (control$saveFitLibrary) {
        eval(bquote(fitLibrary[[.(index)]] <- .(testAlg$fit)),
             envir = fitLibEnv)
      }
    }
    if (verbose) {
      message(paste("full", libraryNames[index]))
    }
    invisible(out)
  }
  predY <- do.call("cbind", lapply(seq(k), FUN = .predFun,
                                   lib = library$library, Y = Y, dataX = X, newX = newX,
                                   whichScreen = whichScreen, family = family, id = id,
                                   obsWeights = obsWeights, verbose = verbose, control = control,
                                   libraryNames = libraryNames))
  errorsInLibrary <- apply(predY, 2, function(algorithm) anyNA(algorithm))
  if (sum(errorsInLibrary) > 0) {
    if (sum(coef[as.logical(errorsInLibrary)]) > 0) {
      warning(paste0("Re-running estimation of coefficients removing failed algorithm(s)\n",
                     "Original coefficients are: \n", paste(coef,
                                                            collapse = ", "), "\n"))
      Z[, as.logical(errorsInLibrary)] <- 0
      if (all(Z == 0)) {
        stop("All algorithms dropped from library")
      }
      getCoef <- method$computeCoef(Z = Z, Y = Y, libraryNames = libraryNames,
                                    obsWeights = obsWeights, control = control, verbose = verbose,
                                    errorsInLibrary = errorsInLibrary)
      coef <- getCoef$coef
      names(coef) <- libraryNames
    }
    else {
      warning("Coefficients already 0 for all failed algorithm(s)")
    }
  }
  getPred <- method$computePred(predY = predY, coef = coef,
                                control = control)

  if (control$saveCVFitLibrary) {
    cvFitLibrary <- lapply(crossValFUN_out, "[[", "model_out")
  } else {
    cvFitLibrary <- NULL
  }
  colnames(predY) <- libraryNames
  if (sum(errorsInCVLibrary) > 0) {
    getCoef$cvRisk[as.logical(errorsInCVLibrary)] <- NA
  }

  out <- list(call = call, libraryNames = libraryNames, SL.library = library,
              SL.predict = getPred, coef = coef, library.predict = predY,
              Z = Z, cvRisk = getCoef$cvRisk, family = family, fitLibrary = get("fitLibrary",
                                                                                envir = fitLibEnv), cvFitLibrary = cvFitLibrary,
              varNames = varNames, validRows = validRows, method = method,
              whichScreen = whichScreen, control = control, cvControl = cvControl,
              errorsInCVLibrary = errorsInCVLibrary, errorsInLibrary = errorsInLibrary,
              metaOptimizer = getCoef$optimizer, env = env, times = times)
  class(out) <- c("SuperLearner")
  return(out)
}
