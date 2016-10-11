library(LogitGLMMLRVB)
library(rstan)
library(lme4)

library(dplyr)
library(reshape2)

project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "variational_bayes/logit_glmm")
source(file.path(project_directory, "LogitGLMMLRVB/inst/optimize_lib.R"))

analysis_name <- "simulated_data"

data_directory <- file.path(project_directory, "LogitGLMMLRVB/inst/data/")
stan_draws_file <- file.path(data_directory, paste(analysis_name, "_mcmc_draws.Rdata", sep=""))

stan_results <- environment()
load(stan_draws_file, envir=stan_results)
mcmc_sample <- extract(stan_results$stan_sim)

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
vp_nat$beta_info <- diag(5, k_reg)
vp_nat$beta_diag_min <- 1
vp_nat$mu_loc <- 0
vp_nat$tau_alpha <- 2
vp_nat$tau_beta <- 2
vp_nat$u_info_min <- 0.1
for (g in 1:n_groups) {
  vp_nat$u_vec[[g]]$u_loc <- 0
  vp_nat$u_vec[[g]]$u_info <- 1
}


##############################
# Initial fit


vp_nat_init <- vp_nat
bounds <- GetVectorBounds()
theta_init <- GetNaturalParameterVector(vp_nat, TRUE)

n_sim_results <- list()
for (n_sim in c(2, 3, 4, 5, 6, 7, 8, 10, 15, 20)) {
  print("----------------------------")
  print(n_sim)
  print("----------------------------\n")
  opt <- GetOptions(n_sim=n_sim)
  optim_fns <- OptimFunctions(y, y_g, x, vp_nat, pp, opt, verbose=FALSE)
  trust_fn <- TrustFunction(optim_fns)
  
  fit_time <- Sys.time()
  # Initialize with BFGS and finish with Newton.
  cat("BFGS initialization.\n")
  optim_result <- optim(theta_init, optim_fns$OptimVal, optim_fns$OptimGrad,
                        method="L-BFGS-B", lower=bounds$theta_lower, upper=bounds$theta_upper,
                        control=list(fnscale=-1, factr=1))

  cat("Trust region.\n")
  trust_result <- trust(trust_fn, optim_result$par,
                        rinit=1, rmax=100, minimize=FALSE, blather=TRUE, iterlim=100)

  vp_nat_fit <- GetNaturalParametersFromVector(vp_nat, trust_result$argument, TRUE)
  mfvb_cov <- GetCovariance(vp_nat_fit)
  lrvb_cov <- GetLRVBCovariance(y, y_g, x, vp_nat_fit, pp, opt)
  stopifnot(min(diag(lrvb_cov)) > 0)

  fit_time <- Sys.time() - fit_time
  
  hess <- optim_fns$OptimHess(newton_result$theta)
  hess_ev <- eigen(hess)
  max_ev <- max(hess_ev$values)
  stopifnot(max_ev < 1e-8)

  vp_mom <- GetMomentParametersFromNaturalParameters(vp_nat_fit)
  lrvb_sd <- GetMomentParametersFromVector(vp_mom, sqrt(diag(lrvb_cov)), FALSE)
  mfvb_sd <- GetMomentParametersFromVector(vp_mom, sqrt(diag(mfvb_cov)), FALSE)

  results <- SummarizeResults(mcmc_sample, vp_mom, mfvb_sd, lrvb_sd)
  results$n_sim <- n_sim
  results$fit_time <- fit_time
  n_sim_results[[as.character(n_sim)]] <- results
}

results <- do.call(rbind, n_sim_results)

fit_df <-
  filter(results, metric == "mean") %>%
  dcast(par + component + group + n_sim ~ method, value.var="val") %>%
  mutate(diff2=(mcmc - mfvb) ^ 2)

# Accuracy

ggplot(filter(fit_df, par != "tau", par != "u")) +
  geom_point(aes(x=n_sim, y=diff2, color=paste(par, component, sep=" "))) +
  expand_limits(x=0, y=0)

ggplot(filter(fit_df, par == "u")) +
  geom_point(aes(x=n_sim, y=diff2, color=paste(par, component, sep=" "))) +
  expand_limits(x=0, y=0)

ggplot(filter(fit_df, par == "tau")) +
  geom_point(aes(x=n_sim, y=diff2, color=paste(par, component, sep=" "))) +
  expand_limits(x=0, y=0)

# Timing
ggplot(unique(results[c("n_sim", "fit_time")])) +
  geom_point(aes(x=n_sim, y=fit_time)) +
  expand_limits(x=0, y=0)
  


