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


# Load the STAN model
stan_directory <- file.path(project_directory, "LogitGLMMLRVB/inst/stan")
stan_model_name <- "logit_glmm_lrvb"
model_file <- file.path(stan_directory, paste(stan_model_name, "stan", sep="."))
model_file_rdata <- file.path(stan_directory, paste(stan_model_name, "Rdata", sep="."))
if (file.exists(model_file_rdata)) {
  print("Loading pre-compiled Stan model.")
  load(model_file_rdata)
} else {
  # Run this to force re-compilation of the model.
  print("Compiling Stan model.")
  model_file <- file.path(stan_directory, paste(stan_model_name, "stan", sep="."))
  model <- stan_model(model_file)
  save(model, file=model_file_rdata)
}

pp <- stan_results$pp


# Stan data.
N_sim <- 10
stan_dat <- list(NG = n_groups,
                 N = length(y),
                 K = ncol(x),
                 y_group = y_g,
                 y = y,
                 x = x,
                 # Priors
                 beta_prior_mean = pp$beta_loc,
                 beta_prior_var = solve(pp$beta_info),
                 mu_prior_mean = pp$mu_loc,
                 mu_prior_var = 1 / pp$mu_info,
                 tau_prior_alpha = pp$tau_alpha,
                 tau_prior_beta = pp$tau_beta,
                 N_sim=N_sim,
                 std_normal_draws=qnorm(1:N_sim / (N_sim + 1)))


init_dat <- list(
  e_beta=pp$beta_loc,
  cov_beta=solve(pp$beta_info),
  e_mu=pp$mu_loc,
  var_mu=1/pp$mu_info,
  tau_shape=pp$tau_alpha,
  tau_rate=pp$tau_beta,
  e_u=rep(0, n_groups),
  var_u=rep(1, n_groups)
  )
# Some knobs we can tweak.  Note that we need many iterations to accurately assess
# the prior sensitivity in the MCMC noise.
iters <- 10000
seed <- 42

# Draw the draws and save.
model <- stan_model(model_file)
vb_optimum <- optimizing(model, data=stan_dat, seed=seed, init=init_dat, verbose=TRUE, iter=1, save_iterations=TRUE, method="")


# A fit with no samples
# model_fit <- stan(model_file, algorithm="Fixed_param", data=stan_dat)
model_fit <- sampling(model, data=stan_dat, chains=1, iter=1)
vb_pars_free <- unconstrain_pars(model_fit, init_dat)

# Get beta, and initialize the moment objects.
log_prob(model_fit, vb_pars_free, adjust_transform=FALSE, gradient=TRUE)

library(numDeriv)
library(trust)

LogProbGrad <- function(vb_pars_free) {
  grad_log_prob(model_fit, vb_pars_free, adjust_transform=FALSE)
}

LogProbHess <- function(vb_pars_free) {
  log_prob_hess <- jacobian(LogProbGrad, vb_pars_free)
  0.5 * (log_prob_hess + t(log_prob_hess))
}

TrustWrapper <- function(vb_pars_free) {
  lp_grad <- log_prob(model_fit, vb_pars_free, adjust_transform=FALSE, gradient=TRUE)
  cat("value: ", as.numeric(lp_grad), "\n")
  print(vb_pars_free)
  return(list(value=as.numeric(lp_grad), gradient=attr(lp_grad, "grad"), hessian=LogProbHess(vb_pars_free)))
}

trust_obj <- trust(TrustWrapper, vb_pars_free, 1, rmax=100, minimize=FALSE, iterlim=5)



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

