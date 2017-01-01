#############################
# Set default options for VB.

library(Matrix) # Needed for transpose.  :(


ConditionNumber <- function(mat) {
  ev <- eigen(mat)$values
  max(abs(ev)) / min(abs(ev))
}


GetOptions <- function(n_sim=20, unconstrained=TRUE,
                       calculate_gradient=TRUE, calculate_hessian=FALSE) {
  opt <- list()

  # Simulate the integral with evenly spaced quantiles rather than random draws.
  interval <- 1 / (n_sim + 1)
  opt$std_draws <- qnorm(seq(interval, 1 - interval, length.out=n_sim), mean=0, sd=1)
  opt$unconstrained <- unconstrained
  opt$calculate_hessian <- calculate_hessian
  opt$calculate_gradient <- calculate_gradient

  return(opt)
}


OptimFunctions <- function(y, y_g, x, vp_nat_init, pp, opt, verbose=TRUE) {

  OptimVal <- function(theta) {
    this_vp_nat <- GetNaturalParametersFromVector(vp_nat_init, theta, TRUE)
    opt$calculate_gradient <- FALSE
    derivs <- GetELBODerivatives(y, y_g, x, this_vp_nat, pp, opt)
    if (verbose) cat(sprintf("Value: %0.16f\n", derivs$val))
    return(derivs$val)
  }

  OptimGrad <- function(theta) {
    this_vp_nat <- GetNaturalParametersFromVector(vp_nat_init, theta, TRUE)
    opt$calculate_gradient <- TRUE
    derivs <- GetELBODerivatives(y, y_g, x, this_vp_nat, pp, opt)
    return(derivs$grad)
  }

  OptimHess <- function(theta) {
    this_vp_nat <- GetNaturalParametersFromVector(vp_nat_init, theta, TRUE)
    hess <- GetSparseLogLikHessian(y, y_g, x, this_vp_nat, pp, opt, TRUE) +
            GetSparseEntropyHessian(this_vp_nat, opt)
    return(hess)
  }

  return(list(OptimVal=OptimVal, OptimGrad=OptimGrad, OptimHess=OptimHess))
}


TrustFunction <- function(optim_fns) {
  function(theta) {
    val <- optim_fns$OptimVal(theta)
    cat(val, "\n")
    list(value=val,
         gradient=optim_fns$OptimGrad(theta),
         hessian=as.matrix(optim_fns$OptimHess(theta)))
  }
}

# TODO: pass prob into this!
GetVectorBounds <- function(loc_bound=100, info_bound=1e9, min_bound=1.0000001) {
  # Get sensible extreme bounds, for example for L-BFGS-B
  vp_nat_lower <- prob$vp_nat
  vp_nat_lower$beta_loc[] <- -1 * loc_bound
  vp_nat_lower$beta_info[,] <- -1 * info_bound
  diag(vp_nat_lower$beta_info) <- vp_nat_lower$beta_diag_min * min_bound
  vp_nat_lower$mu_loc <- -1 * loc_bound
  vp_nat_lower$mu_info <- vp_nat_lower$mu_info_min * min_bound
  vp_nat_lower$tau_alpha <- vp_nat_lower$tau_alpha_min * min_bound
  vp_nat_lower$tau_beta <- vp_nat_lower$tau_beta_min * min_bound
  for (g in 1:prob$vp_nat$n_groups) {
    vp_nat_lower$u[[g]]$u_loc <- -1 * loc_bound
    vp_nat_lower$u[[g]]$u_info <- vp_nat_lower$u_info_min * min_bound
  }

  theta_lower <- GetNaturalParameterVector(vp_nat_lower, TRUE)


  vp_nat_upper <- prob$vp_nat
  vp_nat_upper$beta_loc[] <- loc_bound
  vp_nat_upper$beta_info[,] <- info_bound
  vp_nat_upper$mu_loc <- loc_bound
  vp_nat_upper$mu_info <- info_bound
  vp_nat_upper$tau_alpha <- info_bound
  vp_nat_upper$tau_beta <- info_bound
  for (g in 1:prob$vp_nat$n_groups) {
    vp_nat_upper$u[[g]]$u_loc <- loc_bound
    vp_nat_upper$u[[g]]$u_info <- info_bound
  }

  theta_upper <- GetNaturalParameterVector(vp_nat_upper, TRUE)

  return(list(theta_lower=theta_lower, theta_upper=theta_upper))
}




FitModel <- function(y, y_g, x, vp_nat_init, pp, opt=GetOptions(), reltol=1e-16) {
  optim_funs <- OptimFunctions(y, y_g, x, vp_nat_init, pp, opt)
  theta_start <- GetNaturalParameterVector(vp_nat_init, TRUE)
  optim_result <- optim(theta_start, method="BFGS", control=list(fnscale=-1, reltol=reltol),
                        optim_funs$OptimVal, optim_funs$OptimGrad)
  vp_nat_bfgs <- GetNaturalParametersFromVector(vp_nat_init, optim_result$par, TRUE)
  return(vp_nat_bfgs)
}



GetLRVBResults <- function(y, y_g, x, vp_nat, pp, opt=GetOptions()) {
  jac <- Matrix(GetMomentJacobian(vp_nat, opt)$jacobian)

  hess_time <- Sys.time()
  elbo_hess <- GetSparseLogLikHessian(y, y_g, x, vp_nat, pp, opt, TRUE) +
               GetSparseEntropyHessian(vp_nat, opt)
  hess_time <- Sys.time() - hess_time

  # model_hess2 <- GetELBODerivatives(y, y_g, x, vp_nat_bfgs, pp, GetOptions(calculate_hessian = TRUE))
  # max(abs(model_hess2$hess - model_hess$hess))
  lrvb_cov <- -1 * jac %*% Matrix::solve(elbo_hess, Matrix::t(jac))
  return(list(lrvb_cov=lrvb_cov, jac=jac, elbo_hess=elbo_hess, hess_time=hess_time))
}


GetIndices <- function(vp_nat) {
  indices <- GetNaturalParameterVector(vp_nat, FALSE)
  indices <- as.numeric(1:length(indices))
  vp_nat_index <- GetNaturalParametersFromVector(vp_nat, indices, FALSE)

  # The u indices start at the location of the first u index.
  if (length(vp_nat_index$u) > 0) {
    u_ind <- vp_nat_index$u[[1]]$u_loc:length(indices)
  } else {
    u_ind <- c()
  }
  global_ind <- setdiff(1:length(indices), u_ind)

  return(list(u_ind=u_ind, global_ind=global_ind))
}


GetLRVBSubMatrices <- function(vp_nat, elbo_hess, jac) {
  # Get indices by converting to and from a vector.
  indices <- GetIndices(vp_nat)
  u_ind <- indices$u_ind
  global_ind <- indices$global_ind

  q_tt <- elbo_hess[global_ind, global_ind]
  q_tz <- elbo_hess[global_ind, u_ind]
  q_zt <- elbo_hess[u_ind, global_ind]
  q_zz <- elbo_hess[u_ind, u_ind]

  jac_t <- jac[global_ind, global_ind]

  lrvb_inv_term <- q_tt - q_tz %*% Matrix::solve(q_zz, q_zt)
  return(list(lrvb_inv_term=lrvb_inv_term, jac_t=jac_t))
}


PackMCMCSamplesIntoMoments <- function(mcmc_sample, mp_opt, n_draws=dim(mcmc_sample$beta)[1]) {
  mp_draws <- list()
  zero_mp <- GetMomentParametersFromVector(mp_opt, rep(NaN, mp_opt$encoded_size), unconstrained=TRUE)

  draw <- 1
  for (draw in 1:n_draws) {
    if (draw %% 100 == 0) {
      cat("Writing draw ", draw, " of ", n_draws, "(", 100 * draw / n_draws, "%)\n")
    }
    mp_draw <- zero_mp

    beta <- mcmc_sample$beta[draw, ]
    mu <- mcmc_sample$mu[draw]
    tau <- mcmc_sample$tau[draw]
    u <- mcmc_sample$u[draw, ]

    mp_draw$beta_e_vec <- beta
    mp_draw$beta_e_outer <- beta %*% t(beta)

    mp_draw$mu_e <- mu
    mp_draw$mu_e2 <- mu ^ 2

    mp_draw$tau_e <- tau
    mp_draw$tau_e_log <- log(tau)

    for (g in 1:(mp_opt$n_groups)) {
      mp_draw$u[[g]]$u_e <- u[g]
      mp_draw$u[[g]]$u_e2 <- u[g] ^ 2
    }

    GetMomentParameterVector(mp_draw, unconstrained=FALSE)

    mp_draws[[draw]] <- mp_draw
  }

  return(mp_draws)
}


# Keep the formatting standard.
ResultRow <- function(par, component, group, method, metric, val) {
  return(data.frame(par=par, component=component, group=group, method=method, metric=metric, val=val))
}


SummarizeMCMCColumn <- function(draws, par, component=-1, group=-1, method="mcmc") {
  rbind(ResultRow(par, component, group, method=method, metric="mean", val=mean(draws)),
        ResultRow(par, component, group, method=method, metric="sd", val=sd(draws)))
}


SummarizeVBVariableMetric <- function(vp_mom_list, method, metric, Accessor, par, component=-1, group=-1) {
  return(ResultRow(par, component, group, method=method, metric=metric, val=Accessor(vp_mom_list)))
}


# # The Accessor function should take a vb list and return the appropriate component.
# SummarizeVBVariable <- function(vp_mom, mfvb_sd, lrvb_sd, Accessor, par, component=-1, group=-1) {
#   rbind(SummarizeVBVariableMetric(vp_mom, method="mfvb", metric="mean",
#                                   Accessor=Accessor, par=par, component=component, group=group),
#         SummarizeVBVariableMetric(mfvb_sd, method="mfvb", metric="sd",
#                                   Accessor=Accessor, par=par, component=component, group=group),
#         SummarizeVBVariableMetric(lrvb_sd, method="lrvb", metric="sd",
#                                   Accessor=Accessor, par=par, component=component, group=group))
# }


# VB results for a particular method and metric.  vp_mom only needs to be a list with some metric meaninful
# for the VB moment parameters.
SummarizeVBResults <- function(vp_mom, method, metric) {
  k_reg <- vp_mom$k_reg
  n_groups <- vp_mom$n_groups
  
  results_list <- list()
  
  results_list[[length(results_list) + 1]] <-
    SummarizeVBVariableMetric(vp_mom, method=method, metric=metric, function(vp) { vp[["mu_e"]] }, par="mu")
  results_list[[length(results_list) + 1]] <-
    SummarizeVBVariableMetric(vp_mom,method=method, metric=metric, function(vp) { vp[["tau_e"]] }, par="tau")
  for (k in 1:k_reg) {
    results_list[[length(results_list) + 1]] <-
      SummarizeVBVariableMetric(vp_mom, method=method, metric=metric, function(vp) { vp[["beta_e_vec"]][k] },
                                par="beta", component=k)
  }
  for (g in 1:n_groups) {
    results_list[[length(results_list) + 1]] <-
      SummarizeVBVariableMetric(vp_mom, method=method, metric=metric, function(vp) { vp[["u"]][[g]][["u_e"]] },
                                par="u", group=g)
  }

  return(do.call(rbind, results_list))
}



SummarizeResults <- function(mcmc_sample, vp_mom, mfvb_sd, lrvb_sd) {
  k_reg <- vp_mom$k_reg
  n_groups <- vp_mom$n_groups

  results_list <- list()

  results_list[[length(results_list) + 1]] <- SummarizeMCMCColumn(mcmc_sample$mu, par="mu")
  results_list[[length(results_list) + 1]] <- SummarizeMCMCColumn(mcmc_sample$tau, par="tau")
  for (k in 1:k_reg) {
    results_list[[length(results_list) + 1]] <-
      SummarizeMCMCColumn(mcmc_sample$beta[, k], par="beta", component=k)
  }
  for (g in 1:n_groups) {
    results_list[[length(results_list) + 1]] <-
      SummarizeMCMCColumn(mcmc_sample$u[, g], par="u", group=g)
  }

  results_list[[length(results_list) + 1]] <- SummarizeVBResults(vp_mom, method="mfvb", metric="mean")
  results_list[[length(results_list) + 1]] <- SummarizeVBResults(mfvb_sd, method="mfvb", metric="sd")
  results_list[[length(results_list) + 1]] <- SummarizeVBResults(lrvb_sd, method="lrvb", metric="sd")
  
  return(do.call(rbind, results_list))
}
