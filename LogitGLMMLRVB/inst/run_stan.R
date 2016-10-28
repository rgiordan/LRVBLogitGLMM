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

analysis_name <- "simulated_data"

set.seed(42)

#############################
# Simualate some data

n_obs_per_group <- 100
k_reg <- 2
n_groups <- 100
n_obs <- n_groups * n_obs_per_group

set.seed(42)
true_params <- list()
true_params$n_obs <- n_obs
true_params$k_reg <- k_reg
true_params$n_groups <- n_groups

true_params$beta <- 1:k_reg
true_params$tau <- 1
true_params$mu <- -0.5
true_params$u <- list()
for (g in 1:n_groups) {
  true_params$u[[g]] <- rnorm(1, true_params$mu, 1 / sqrt(true_params$tau))
}

x <- matrix(runif(n_obs * k_reg), nrow=n_obs, ncol=k_reg)

# y_g is expected to be zero-indexed.
y_g <- as.integer(rep(1:n_groups, each=n_obs_per_group) - 1)
true_offsets <- x %*% true_params$beta
for (n in 1:n_obs) {
  # C++ is zero indexed but R is one indexed
  true_offsets[n] <- true_offsets[n] + true_params$u[[y_g[n] + 1]]
}
y <- rbinom(n=n_obs, size=1, prob=inv.logit(true_offsets))

##########################
# Prior parameters

pp <- GetEmptyPriorParameters(k_reg)

pp$beta_loc <- rep(0, k_reg)
pp$beta_info <- diag(k_reg)
pp$mu_loc <- 0
pp$mu_info <- 1
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
                 tau_prior_alpha = pp$tau_alpha,
                 tau_prior_beta = pp$tau_beta)

# Some knobs we can tweak.  Note that we need many iterations to accurately assess
# the prior sensitivity in the MCMC noise.
chains <- 1
iters <- 2000
control <- list(adapt_t0 = 10,       # default = 10
                stepsize = 1,        # default = 1
                max_treedepth = 6)   # default = 10
seed <- 42

# Draw the draws and save.
mcmc_time <- Sys.time()
stan_sim <- sampling(model, data = stan_dat, seed = seed,
                     chains = chains, iter = iters, control = control)
mcmc_time <- Sys.time() - mcmc_time

data_directory <- file.path(project_directory, "LogitGLMMLRVB/inst/data/")
stan_draws_file <- file.path(data_directory, paste(analysis_name, "_mcmc_draws.Rdata", sep=""))
save(stan_sim, mcmc_time, stan_dat, true_params, pp, file=stan_draws_file)

# stan_advi <- vb(model, data = stan_dat,  algorithm="meanfield", output_samples=iters)
# stan_advi_full <- vb(model, data = stan_dat,  algorithm="fullrank", output_samples=iters)
# 
# save(stan_sim, mcmc_time, stan_dat, true_params, pp,
#      stan_advi,stan_advi_full, file=stan_draws_file)
# 

