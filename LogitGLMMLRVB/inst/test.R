library(LogitGLMMLRVB)
library(lme4)

library(dplyr)
library(reshape2)

# Get C++ versions of the data structures to make sure
# we get the field names right.
n_obs <- 1000
k_reg <- 2
n_groups <- 20

prob <- SampleData(n_obs, k_reg, n_groups)

####################
# Check conversions

for (g in 1:prob$vp_nat$n_groups) {
  stopifnot(prob$vp_nat$u_vec[[g]]$u_info > prob$vp_nat$u_info_min)
}

for (unconstrained in c(T, F)) {
  theta <- GetMomentParameterVector(prob$vp_mom, unconstrained)
  vp_new <- GetMomentParametersFromVector(prob$vp_mom, theta, unconstrained)
  stopifnot(all.equal(vp_new, prob$vp_mom))
  
  theta <- GetNaturalParameterVector(prob$vp_nat, unconstrained)
  vp_new <- GetNaturalParametersFromVector(prob$vp_nat, theta, unconstrained)
  stopifnot(all.equal(vp_new, prob$vp_nat))
}


##############################
# Check fit

elbo_derivs <-
  GetELBODerivatives(prob$y, prob$y_g, prob$x, prob$vp_nat, prob$pp, GetOptions(n_sim=100))
stopifnot(!is.nan(elbo_derivs$val))
stopifnot(all(!is.nan(elbo_derivs$grad)))

opt <- GetOptions()
bounds <- GetVectorBounds()
optim_fns <- OptimFunctions(prob$y, prob$y_g, prob$x, prob$vp_nat, prob$pp, opt)
theta_init <- GetNaturalParameterVector(prob$vp_nat, TRUE)
optim_result <- optim(theta_init, optim_fns$OptimVal, optim_fns$OptimGrad,
                      method="L-BFGS-B",
                      lower=bounds$theta_lower, upper=bounds$theta_upper,
                      control=list(fnscale=-1, factr=1))
vp_nat <- GetNaturalParametersFromVector(prob$vp_nat, optim_result$par, TRUE)


################################
# LRVB

mfvb_cov <- GetCovariance(vp_nat)
lrvb_cov <- GetLRVBCovariance(prob$y, prob$y_g, prob$x, vp_nat, prob$pp, opt)

stopifnot(min(diag(lrvb_cov)) > 0)
stopifnot(min(diag(mfvb_cov)) > 0)
stopifnot(max(abs(lrvb_cov - t(lrvb_cov))) < 1e-12)
stopifnot(max(abs(mfvb_cov - t(mfvb_cov))) < 1e-12)


##########################
# Sub-matrix LRVB

indices <- GetIndices(vp_nat)
global_ind <- indices$global_ind

elbo_hess <- GetSparseLogLikHessian(prob$y, prob$y_g, prob$x, vp_nat, prob$pp, opt, TRUE) +
             GetSparseEntropyHessian(vp_nat, opt)
jac <- Matrix(GetMomentJacobian(vp_nat, opt)$jacobian)

sub_mat <- GetLRVBSubMatrices(vp_nat, elbo_hess, jac)
lrvb_v2 <- -1 * sub_mat$jac_t %*% solve(sub_mat$lrvb_inv_term, t(sub_mat$jac_t))

stopifnot(max(abs(lrvb_v2 - lrvb_cov[global_ind, global_ind])) < 1e-12)


##############################
# Test the hessian.  The sparse version scales much better.

n_group_small <- 20
vp_nat_small <- prob$vp_nat
keep_ind <- prob$y_g < n_group_small

vp_nat_small$n_obs <- sum(keep_ind)
vp_nat_small$n_groups <- n_group_small
vp_nat_small$u_vec <- prob$vp_nat$u_vec[1:vp_nat_small$n_groups]

for (unconstrained in c(TRUE, FALSE)) {
  sparse_hess_time <- Sys.time()
  hess <- GetSparseLogLikHessian(prob$y[keep_ind], prob$y_g[keep_ind], prob$x[keep_ind, ],
                                 vp_nat_small, prob$pp, GetOptions(unconstrained=unconstrained), TRUE)
  sparse_hess_time <- Sys.time() - sparse_hess_time
  
  hess_time <- Sys.time()
  derivs <- GetLogLikDerivatives(
    prob$y[keep_ind], prob$y_g[keep_ind], prob$x[keep_ind, ],
    vp_nat_small, GetOptions(unconstrained=unconstrained, calculate_hessian = TRUE))
  prior_derivs <-
    GetLogPriorDerivatives(vp_nat_small, prob$pp,
                           GetOptions(unconstrained=unconstrained, calculate_hessian = TRUE))
  hess_time <- Sys.time() - hess_time
  
  stopifnot(max(abs(hess - (derivs$hess + prior_derivs$hess))) < 1e-10)
  stopifnot(sparse_hess_time < hess_time)
  
  # Entropy
  hess <- GetSparseEntropyHessian(vp_nat_small, GetOptions())
  derivs <- GetEntropyDerivatives( vp_nat_small, GetOptions(calculate_hessian = TRUE))
  stopifnot(max(abs(hess - derivs$hess)) < 1e-12)
}



#####################
# GLMER.  Not a formal test.

prob_data <- data.frame(y=y, y_g=as.character(y_g))
prob_data <- cbind(prob_data, data.frame(x))
glmer_result <- glmer(y ~ 1 + X1 + X2 + (1 | y_g), prob_data, family=binomial)
glmer_summary <- summary(glmer_result)

# Not bad:
glmer_summary
true_params$beta

# This is not actually very good:
if (F) {
  plot(ranef(glmer_result)[[1]][[1]], unlist(true_params$u)); abline(0, 1)
}







