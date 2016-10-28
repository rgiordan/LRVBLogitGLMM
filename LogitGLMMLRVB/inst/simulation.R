library(LogitGLMMLRVB)
library(rstan)
library(lme4)

library(dplyr)
library(reshape2)
library(trust)

project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "LRVBLogitGLMM")
source(file.path(project_directory, "LogitGLMMLRVB/inst/optimize_lib.R"))

analysis_name <- "simulated_data"

data_directory <- file.path(project_directory, "LogitGLMMLRVB/inst/data/")
stan_draws_file <- file.path(data_directory, paste(analysis_name, "_mcmc_draws.Rdata", sep=""))

stan_results <- environment()
load(stan_draws_file, envir=stan_results)

true_params <- stan_results$true_params
x <- stan_results$stan_dat$x
k_reg <- ncol(x)
n_obs <- nrow(x)
y <- stan_results$stan_dat$y
y_g <- stan_results$stan_dat$y_g
n_groups <- max(y_g) + 1

# Get C++ versions of the data structures.
prob <- SampleData(n_obs, k_reg, n_groups)

pp <- stan_results$pp
# Fix an obsolete verison of the prior parameters
pp$encoded_size <- length(GetPriorParametersVector(pp, FALSE))

vp_nat <- prob$vp_nat

vp_nat$beta_loc <- rep(0, k_reg)
vp_nat$mu_loc <- 0
vp_nat$tau_alpha <- 2
vp_nat$tau_beta <- 2
for (g in 1:n_groups) {
  vp_nat$u[[g]]$u_loc <- 0
  vp_nat$u[[g]]$u_info <- 1
}

mcmc_sample <- extract(stan_results$stan_sim)

##############################
# Optimization

opt <- GetOptions(n_sim=2)

# TODO: pass prob into this!
bounds <- GetVectorBounds()

optim_fns <- OptimFunctions(y, y_g, x, vp_nat, pp, opt)
theta_init <- GetNaturalParameterVector(vp_nat, TRUE)

fit_time <- Sys.time()
# Initialize with BFGS and finish with Newton.
cat("BFGS initialization.\n")
optim_result <- optim(theta_init, optim_fns$OptimVal, optim_fns$OptimGrad,
                      method="L-BFGS-B", lower=bounds$theta_lower, upper=bounds$theta_upper,
                      control=list(fnscale=-1, factr=1e7))

cat("Trust region with more samples.\n")
opt <- GetOptions(n_sim=10)
trust_fn <- TrustFunction(OptimFunctions(y, y_g, x, vp_nat, pp, opt))
trust_result <- trust(trust_fn, optim_result$par,
                      rinit=1, rmax=100, minimize=FALSE, blather=TRUE, iterlim=100)
vp_opt <- GetNaturalParametersFromVector(vp_nat, trust_result$argument, TRUE)


# LRVB
mfvb_cov <- GetCovariance(vp_opt)
lrvb_results <- GetLRVBResults(y, y_g, x, vp_opt, pp, opt)
lrvb_cov <- lrvb_results$lrvb_cov
stopifnot(min(diag(lrvb_cov)) > 0)
stopifnot(min(diag(mfvb_cov)) > 0)

fit_time <- Sys.time() - fit_time


#################################
# Sensitivity analysis

comb_indices <- GetPriorsAndNaturalParametersFromVector(
  vp_opt, pp, as.numeric(1:(vp_opt$encoded_size + pp$encoded_size)), FALSE)
comb_prior_ind <- GetPriorParametersVector(comb_indices$pp, FALSE)
comb_vp_ind <- GetNaturalParameterVector(comb_indices$vp, FALSE)

opt$calculate_hessian <- TRUE
log_prior_derivs <- GetFullModelLogPriorDerivatives(vp_opt, pp, opt)
log_prior_param_prior <- Matrix(log_prior_derivs$hess[comb_vp_ind, comb_prior_ind])

prior_sens <- -1 * lrvb_results$jac %*% Matrix::solve(lrvb_results$elbo_hess, log_prior_param_prior)


#####################################
# Draws to moments

mp_opt <- GetMomentParametersFromNaturalParameters(vp_opt)
mp_draws <- PackMCMCSamplesIntoMoments(mcmc_sample, mp_opt)
draws_mat <- do.call(rbind, lapply(mp_draws, function(draw) GetMomentParameterVector(draw, FALSE)))
log_prior_grad_list <- GetMCMCLogPriorDerivatives(mp_draws, pp, opt)
log_prior_grad_mat <- do.call(rbind, log_prior_grad_list)


##################
# Influence functions

obs <- mp_draws[[1]]

# Segfaulting:
GetLogVariationalDensityDerivatives(mp_draws[[1]], vp_opt, opt)


#############################
# Unpack the results.

vp_mom <- GetMomentParametersFromNaturalParameters(vp_opt)
lrvb_sd <- GetMomentParametersFromVector(vp_mom, sqrt(diag(lrvb_cov)), FALSE)
mfvb_sd <- GetMomentParametersFromVector(vp_mom, sqrt(diag(mfvb_cov)), FALSE)

results <- SummarizeResults(mcmc_sample, vp_mom, mfvb_sd, lrvb_sd)

if (FALSE) {
  ggplot(
    filter(results, metric == "mean") %>%
      dcast(par + component + group ~ method, value.var="val") %>%
      mutate(is_u = par == "u")
  ) +
    geom_point(aes(x=mcmc, y=mfvb, color=par), size=3) +
    geom_abline(aes(intercept=0, slope=1)) +
    facet_grid(~ is_u)

  ggplot(
    filter(results, metric == "sd") %>%
      dcast(par + component + group ~ method, value.var="val") %>%
      mutate(is_u = par == "u")) +
    geom_point(aes(x=mcmc, y=mfvb, color="mfvb"), size=3) +
    geom_point(aes(x=mcmc, y=lrvb, color="lrvb"), size=3) +
    geom_abline(aes(intercept=0, slope=1)) +
    facet_grid(~ is_u) +
    ggtitle("Posterior standard deviations")

  ggplot(
    filter(results, metric == "sd", par == "u") %>%
      dcast(par + component + group ~ method, value.var="val")
  ) +
    geom_point(aes(x=mcmc, y=mfvb, color="mfvb"), size=3) +
    geom_point(aes(x=mcmc, y=lrvb, color="lrvb"), size=3) +
    expand_limits(x=0, y=0) +
    xlab("MCMC (ground truth)") + ylab("VB") +
    scale_color_discrete(guide=guide_legend(title="Method")) +
    geom_abline(aes(intercept=0, slope=1))
}
