# First, run the script run_stan.R.  This will get the data and priors
# from that script to ensure that the two analyses are doing everything the same.

library(LogitGLMMLRVB)
library(rstan)
library(lme4)

library(dplyr)
library(reshape2)
library(trust)

library(LRVBUtils)

project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "LRVBLogitGLMM")
source(file.path(project_directory, "LogitGLMMLRVB/inst/optimize_lib.R"))

analysis_name <- "simulated_data"

data_directory <- file.path(project_directory, "LogitGLMMLRVB/inst/data/")
stan_draws_file <- file.path(data_directory, paste(analysis_name, "_mcmc_draws.Rdata", sep=""))

stan_results <- LoadIntoEnvironment(stan_draws_file)

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
lrvb_results <- GetLRVBResults(y, y_g, x, vp_opt, pp, opt)
lrvb_cov <- lrvb_results$lrvb_cov
stopifnot(min(diag(lrvb_cov)) > 0)

fit_time <- Sys.time() - fit_time


#####################################
# Draws to moments

mp_opt <- GetMomentParametersFromNaturalParameters(vp_opt)
mp_draws <- PackMCMCSamplesIntoMoments(mcmc_sample, mp_opt)
draws_mat <- do.call(rbind, lapply(mp_draws, function(draw) GetMomentParameterVector(draw, FALSE)))
log_prior_grad_list <- GetMCMCLogPriorDerivatives(mp_draws, pp, opt)
log_prior_grad_mat <- do.call(rbind, log_prior_grad_list)

####################################
# Save results

vb_results_file <- file.path(data_directory, paste(analysis_name, "_vb_results.Rdata", sep=""))
save(stan_results, vp_opt, mp_opt, lrvb_results, opt, fit_time,
     log_prior_grad_mat, mp_draws, draws_mat, file=vb_results_file)

