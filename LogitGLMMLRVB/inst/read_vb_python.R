# First, run the script run_stan.R.  This will get the data and priors
# from that script to ensure that the two analyses are doing everything the same.

library(LogitGLMMLRVB)
library(rstan)
library(jsonlite)

# library(dplyr)
# library(reshape2)
# library(trust)

# library(LRVBUtils)

project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "LRVBLogitGLMM")
source(file.path(project_directory, "LogitGLMMLRVB/inst/optimize_lib.R"))

analysis_name <- "simulated_data_small"
# analysis_name <- "simulated_data_large"

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


pp <- stan_results$pp

###########################
# Read python results

# Get C++ versions of the data structures.
# We will copy python results into this list.
prob <- SampleData(n_obs, k_reg, n_groups)
vp_opt <- prob$vp_nat

python_filename <- file.path(data_directory, paste(analysis_name, "_python_vb_results.json", sep=""))
json_file <- file(python_filename, "r")
json_dat <- fromJSON(readLines(json_file))
close(json_file)

vp_opt$beta_loc <- json_dat$glmm_par_opt$beta$mean
vp_opt$beta_info <- solve(json_dat$glmm_par_opt$beta$cov)

vp_opt$mu_loc <- json_dat$glmm_par_opt$mu$mean
vp_opt$mu_info <- 1 / json_dat$glmm_par_opt$mu$var

vp_opt$tau_alpha <- json_dat$glmm_par_opt$tau$shape
vp_opt$tau_beta <- json_dat$glmm_par_opt$tau$rate

for (g in 1:vp_opt$n_groups) {
  vp_opt$u[[g]]$u_loc <- json_dat$glmm_par_opt$u$mean[g]
  vp_opt$u[[g]]$u_info <- 1 / json_dat$glmm_par_opt$u$var[g]
}

lrvb_results <- list()
lrvb_results$lrvb_cov <- json_dat$lrvb_cov
lrvb_results$moment_jac <- json_dat$moment_jac
lrvb_results$elbo_hess <- json_dat$elbo_hess
moment_indices <- json_dat$moment_indices

fit_time <- json_dat$vb_time

#####################################
# Draws to moments

mcmc_sample <- extract(stan_results$stan_sim)
mp_opt <- GetMomentParametersFromNaturalParameters(vp_opt)
mp_draws <- PackMCMCSamplesIntoMoments(mcmc_sample, mp_opt)
draws_mat <- do.call(rbind, lapply(mp_draws, function(draw) GetMomentParameterVector(draw, FALSE)))
opt <- GetOptions(n_sim=10)
log_prior_grad_list <- GetMCMCLogPriorDerivatives(mp_draws, pp, opt)
log_prior_grad_mat <- do.call(rbind, log_prior_grad_list)

####################################
# Save results

vb_results_file <- file.path(data_directory, paste(analysis_name, "_vb_python_results.Rdata", sep=""))
save(stan_results, vp_opt, mp_opt, lrvb_results, fit_time, moment_indices,
     log_prior_grad_mat, mp_draws, draws_mat, file=vb_results_file)

