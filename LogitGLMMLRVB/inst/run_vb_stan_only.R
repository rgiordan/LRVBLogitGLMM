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


# Some knobs we can tweak.  Note that we need many iterations to accurately assess
# the prior sensitivity in the MCMC noise.
iters <- 10000
seed <- 42

# Draw the draws and save.
model <- stan_model(model_file)
vb_optimum <- optimizing(model, data=stan_dat, seed=seed, init=init_dat, verbose=TRUE, save_iterations=TRUE)


# A fit with no samples
# model_fit <- stan(model_file, algorithm="Fixed_param", data=stan_dat)
model_fit <- sampling(model, data=stan_dat, chains=1, iter=1)
vb_pars_free <- unconstrain_pars(model_fit, init_dat)

# Get beta, and initialize the moment objects.
log_prob(model_fit, vb_pars_free, adjust_transform=FALSE, gradient=TRUE)

library(numDeriv)
library(trust)

model <- stan_model(model_file)
model_fit <- sampling(model, data=stan_dat, chains=1, iter=1)

init_dat <- list(
  e_beta=pp$beta_loc + 0.01,
  cov_beta=solve(pp$beta_info) + 0.01,
  e_mu=pp$mu_loc + 0.1,
  var_mu=1/pp$mu_info + 1,
  tau_shape=pp$tau_alpha + 1,
  tau_rate=pp$tau_beta + 1,
  e_u=rep(0, n_groups),
  var_u=rep(1, n_groups)
)
vb_pars_free <- unconstrain_pars(model_fit, init_dat)

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
  print(constrain_pars(model_fit, vb_pars_free))
  return(list(value=as.numeric(lp_grad), gradient=attr(lp_grad, "grad"), hessian=LogProbHess(vb_pars_free)))
}

trust_obj <- trust(TrustWrapper, vb_pars_free, 1, rmax=100, minimize=FALSE, iterlim=5)

model_fit@.MISC$stan_fit_instance$param_names()
unconstrained_opt_pars <-
  data.frame(name=model_fit@.MISC$stan_fit_instance$unconstrained_param_names(FALSE, FALSE),
             val=trust_obj$argument,
             init=vb_pars_free)
max(abs(trust_obj$argument - vb_pars_free))
opt_pars <- constrain_pars(model_fit, trust_obj$argument)
init_pars <- constrain_pars(model_fit, unconstrain_pars(model_fit, init_dat))

stan_results$true_params$beta
opt_pars$e_beta

