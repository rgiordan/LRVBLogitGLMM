library(ggplot2)
library(dplyr)
library(reshape2)
library(rstan)
library(boot) # for inv.logit
library(Matrix)
library(mvtnorm)
library(LogitGLMMLRVB)

project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "LRVBLogitGLMM")

# analysis_name <- "simulated_data_small"
analysis_name <- "simulated_data_large"

set.seed(42)

#############################
# Simualate some data

if (analysis_name == "simulated_data_large") {
  n_obs_per_group <- 20
  k_reg <- 25
  n_groups <- 500
  n_obs <- n_groups * n_obs_per_group
  
  set.seed(42)
  true_params <- list()
  true_params$n_obs <- n_obs
  true_params$k_reg <- k_reg
  true_params$n_groups <- n_groups
  true_params$tau <- 1
  true_params$mu <- -0.5
  true_params$beta <- (1:k_reg) / k_reg

  iters <- 8000
} else if (analysis_name == "simulated_data_small") {
  n_obs_per_group <- 10
  k_reg <- 5
  n_groups <- 100
  n_obs <- n_groups * n_obs_per_group
  
  set.seed(42)
  true_params <- list()
  true_params$n_obs <- n_obs
  true_params$k_reg <- k_reg
  true_params$n_groups <- n_groups
  true_params$tau <- 1
  true_params$mu <- -3.5
  true_params$beta <- 1:k_reg
  
  iters <- 10000
} else {
  stop("Unknown analysis name.")
}

true_params$u <- list()
for (g in 1:n_groups) {
  true_params$u[[g]] <- rnorm(1, true_params$mu, 1 / sqrt(true_params$tau))
}

# x <- matrix(runif(n_obs * k_reg), nrow=n_obs, ncol=k_reg)

# Select correlated regressors to induce posterior correlation in beta.
x_cov <- (matrix(0.5, k_reg, k_reg) + diag(k_reg)) / 2.5
x <- rmvnorm(n_obs, sigma=x_cov)

# y_g is expected to be zero-indexed.
y_g <- as.integer(rep(1:n_groups, each=n_obs_per_group) - 1)
true_offsets <- x %*% true_params$beta
for (n in 1:n_obs) {
  # C++ is zero indexed but R is one indexed
  true_offsets[n] <- true_offsets[n] + true_params$u[[y_g[n] + 1]]
}
true_probs <- inv.logit(true_offsets)
print(summary(true_probs))
y <- rbinom(n=n_obs, size=1, prob=true_probs)

##########################
# Prior parameters

pp <- GetEmptyPriorParameters(k_reg)

pp$beta_loc <- rep(0, k_reg)
pp$beta_info <- 0.01 * diag(k_reg)
pp$mu_loc <- 0
pp$mu_info <- 0.01
pp$tau_alpha <- 3
pp$tau_beta <- 3

######################################
# STAN

# Load the STAN model
stan_directory <- file.path(project_directory, "LogitGLMMLRVB/inst/stan")
stan_model_name <- "logit_glmm"
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

# Stan data.
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
                 mu_prior_mean_c = pp$mu_loc,
                 mu_prior_var_c = 2 / pp$mu_info,
                 mu_prior_t = 1,
                 mu_prior_epsilon = 0,
                 tau_prior_alpha = pp$tau_alpha,
                 tau_prior_beta = pp$tau_beta)

# Some knobs we can tweak.  Note that we need many iterations to accurately assess
# the prior sensitivity in the MCMC noise.
seed <- 42

# Draw the draws and save.
mcmc_time <- Sys.time()
stan_dat$mu_prior_epsilon <- 0
stan_sim <- sampling(model, data=stan_dat, seed=seed, iter=iters, chains=1)
mcmc_time <- Sys.time() - mcmc_time

# Sample with advi
advi_time <- Sys.time()
stan_advi <- vb(model, data=stan_dat,  algorithm="meanfield", output_samples=iters)
advi_time <- Sys.time() - advi_time

# Get a MAP estimate
map_time <- Sys.time()
stan_map <- optimizing(model, data=stan_dat, algorithm="Newton", hessian=TRUE, tol_obj=1e-12, tol_grad=1e-12, tol_param=1e-12)
map_time <- map_time - Sys.time()

bfgs_map_time <- Sys.time()
stan_map_bfgs <- optimizing(model, data=stan_dat, algorithm="BFGS", hessian=TRUE, tol_obj=1e-12, tol_grad=1e-12, tol_param=1e-12)
bfgs_map_time <- bfgs_map_time - Sys.time()

max(eigen(stan_map$hessian)$values)
max(eigen(stan_map_bfgs$hessian)$values)
max(abs(stan_map$par - stan_map_bfgs$par))
stan_map$par

data_directory <- file.path(project_directory, "LogitGLMMLRVB/inst/data/")
stan_draws_file <- file.path(data_directory, paste(analysis_name, "_mcmc_draws.Rdata", sep=""))
save(stan_sim, mcmc_time, stan_dat,
     stan_advi, stan_map,
     advi_time, map_time, bfgs_map_time,
     true_params, pp, file=stan_draws_file)




