# First, run the script run_stan.R.  This will get the data and priors
# from that script to ensure that the two analyses are doing everything the same.

library(LogitGLMMLRVB)
library(rstan)
library(jsonlite)
library(Matrix)

project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "LRVBLogitGLMM")
source(file.path(project_directory, "LogitGLMMLRVB/inst/optimize_lib.R"))

analysis_name <- "simulated_data_small"
#analysis_name <- "simulated_data_large"

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

vb_results_file <- file.path(data_directory, paste(analysis_name, "_vb_results.Rdata", sep=""))
vb_results <- LoadIntoEnvironment(vb_results_file)

vp_opt$beta_loc <- json_dat$glmm_par_opt$beta$mean
vp_opt$beta_info <- json_dat$glmm_par_opt$beta$info

vp_opt$mu_loc <- json_dat$glmm_par_opt$mu$mean
vp_opt$mu_info <- json_dat$glmm_par_opt$mu$info

vp_opt$tau_alpha <- json_dat$glmm_par_opt$tau$shape
vp_opt$tau_beta <- json_dat$glmm_par_opt$tau$rate

for (g in 1:vp_opt$n_groups) {
  vp_opt$u[[g]]$u_loc <- json_dat$glmm_par_opt$u$mean[g]
  vp_opt$u[[g]]$u_info <- json_dat$glmm_par_opt$u$info[g]
}

fit_time <- json_dat$vb_time

opt <- GetOptions(n_sim=10)

# Get versions from Python.
py_mp_indices <- json_dat$moment_indices
py_prior_indices <- json_dat$prior_indices
mp_opt <- GetMomentParametersFromNaturalParameters(vp_opt)


# Define a permutation of the moment indices to map from the python indices to the R indices.
r_mp_indices <- GetMomentParametersFromVector(mp_opt, as.numeric(1:mp_opt$encoded_size), FALSE)
stopifnot(r_mp_indices$encoded_size == max(unlist(py_mp_indices)))

mp_permutation <- rep(NaN, r_mp_indices$encoded_size)
mp_permutation[r_mp_indices$beta_e_vec] <- py_mp_indices$e_beta
mp_permutation[r_mp_indices$beta_e_outer] <- py_mp_indices$e_beta_outer
mp_permutation[r_mp_indices$mu_e] <- py_mp_indices$e_mu
mp_permutation[r_mp_indices$mu_e2] <- py_mp_indices$e_mu2
mp_permutation[r_mp_indices$tau_e] <- py_mp_indices$e_tau
mp_permutation[r_mp_indices$tau_e_log] <- py_mp_indices$e_log_tau
for (g in 1:r_mp_indices$n_groups) {
  mp_permutation[r_mp_indices$u[[g]]$u_e] <- py_mp_indices$e_u[g]
  mp_permutation[r_mp_indices$u[[g]]$u_e2] <- py_mp_indices$e_u2[g]
}


# Define a permutation of the variational parameter indices to map from the python indices to the R indices.
r_vp_indices <- GetNaturalParametersFromVector(vp_opt, as.numeric(1:vp_opt$encoded_size), FALSE)
py_vp_indices <- json_dat$vp_indices
stopifnot(r_vp_indices$encoded_size == max(unlist(py_vp_indices)))

vp_permutation <- rep(NaN, r_vp_indices$encoded_size)
vp_permutation[r_vp_indices$beta_loc] <- py_vp_indices$beta$mean
vp_permutation[r_vp_indices$beta_info] <- py_vp_indices$beta$info
vp_permutation[r_vp_indices$mu_loc] <- py_vp_indices$mu$mean
vp_permutation[r_vp_indices$mu_info] <- py_vp_indices$mu$info
vp_permutation[r_vp_indices$tau_alpha] <- py_vp_indices$tau$shape
vp_permutation[r_vp_indices$tau_beta] <- py_vp_indices$tau$rate
for (g in 1:r_vp_indices$n_groups) {
  vp_permutation[r_vp_indices$u[[g]]$u_loc] <- py_vp_indices$u$mean[g]
  vp_permutation[r_vp_indices$u[[g]]$u_info] <- py_vp_indices$u$info[g]
}


# Define a permutation of the prior parameter indices to map from the python indices to the R indices.
r_pp_indices <- GetPriorParametersFromVector(pp, as.numeric(1:pp$encoded_size), FALSE)
py_pp_indices <- json_dat$prior_indices
stopifnot(r_pp_indices$encoded_size == max(unlist(py_pp_indices)))

pp_permutation <- rep(NaN, r_pp_indices$encoded_size)
pp_permutation[r_pp_indices$beta_loc] <- py_pp_indices$beta_prior_mean
pp_permutation[r_pp_indices$beta_info] <- py_pp_indices$beta_prior_info
pp_permutation[r_pp_indices$mu_loc] <- py_pp_indices$mu_prior_mean
pp_permutation[r_pp_indices$mu_info] <- py_pp_indices$mu_prior_info
pp_permutation[r_pp_indices$tau_alpha] <- py_pp_indices$tau_prior_alpha
pp_permutation[r_pp_indices$tau_beta] <- py_pp_indices$tau_prior_beta


###############
# Apply the permutations to get results that are comparable to the R results

lrvb_results <- list()
lrvb_results$lrvb_cov <- json_dat$lrvb_cov[mp_permutation, mp_permutation]
lrvb_results$jac <- json_dat$moment_jac[mp_permutation, vp_permutation]
lrvb_results$elbo_hess <- json_dat$elbo_hess[vp_permutation, vp_permutation]
log_prior_hess <- json_dat$log_prior_hess[pp_permutation, vp_permutation]


#####################################
# Draws to moments

mcmc_sample <- extract(stan_results$stan_sim)
mp_draws <- PackMCMCSamplesIntoMoments(mcmc_sample, mp_opt)
draws_mat <- do.call(rbind, lapply(mp_draws, function(draw) GetMomentParameterVector(draw, FALSE)))
log_prior_grad_list <- GetMCMCLogPriorDerivatives(mp_draws, pp, opt)
log_prior_grad_mat <- do.call(rbind, log_prior_grad_list)

####################################
# Save results

vb_results_file <- file.path(data_directory, paste(analysis_name, "_vb_python_results.Rdata", sep=""))
save(stan_results, vp_opt, mp_opt, lrvb_results, opt, fit_time, log_prior_hess,
     log_prior_grad_mat, mp_draws, draws_mat, file=vb_results_file)



stop("Debugging follows")







###############
# Debugging

GetNaturalParameterVector(r_vp=vp_opt, TRUE)[vp_indices$tau_alpha]
GetNaturalParameterVector(r_vp=vp_opt, TRUE)[vp_indices$tau_beta]

r_lrvb_results <- GetLRVBResults(y, y_g, x, vp_opt, pp, opt)
r_lrvb_results_orig <- GetLRVBResults(y, y_g, x, vb_results$vp_opt, pp, opt)

r_elbo_hess <- r_lrvb_results$elbo_hess
py_elbo_hess <- json_dat$elbo_hess[vp_permutation, vp_permutation]
plot(r_elbo_hess, py_elbo_hess); abline(0, 1)
plot(diag(r_elbo_hess), diag(py_elbo_hess)); abline(0, 1)
plot(solve(r_elbo_hess), solve(py_elbo_hess)); abline(0, 1)
max(abs(r_elbo_hess - py_elbo_hess))

#########
# This was messed up due to the C++ moment parameters defaulting to unconstrained :(
r_jac <- r_lrvb_results$jac
py_jac <- json_dat$moment_jac[mp_permutation, vp_permutation]
# inds <- unlist(lapply(r_mp_indices$u, function(par) par$u_e2))
# inds <- inds[45:length(inds)]
inds <- 1:r_mp_indices$encoded_size
# inds <- as.numeric(r_mp_indices$tau_e)
# r_jac[inds, ]
# py_jac[inds, ]
plot(r_jac[inds, ], py_jac[inds, ]); abline(0, 1)
plot(sort(abs(r_jac[inds, ])), sort(abs(py_jac[inds, ]))); abline(0, 1)


# Calculate the R derivative numerically
free_vec <- GetNaturalParameterVector(r_vp=vp_opt, TRUE)
GetETau <- function(free_vec) {
  vp_local <- GetNaturalParametersFromVector(vp_opt, free_vec, unconstrained=TRUE)
  mp_local <- GetMomentParametersFromNaturalParameters(vp_local)
  return(mp_local$tau_e)
}

# It appears that the R Jacobian is wrong!  (Or I have misinterpreted something.)
epsilon <- rep(0, length(free_vec))
epsilon[vp_indices$tau_alpha] <- 1e-4
(GetETau(free_vec + epsilon) - GetETau(free_vec)) / epsilon[vp_indices$tau_alpha]
r_jac[r_mp_indices$tau_e, vp_indices$tau_alpha]


#################

indices <- matrix(1:prod(dim(r_elbo_hess)), dim(r_elbo_hess)[1])
max_ind <- which.max(as.matrix(abs(r_elbo_hess - py_elbo_hess)))
indices[max_ind]
t(indices)[max_ind]

hess_diff <- r_elbo_hess - py_elbo_hess
hess_diff[abs(hess_diff)< 1e-6] <- 0
hess_diff <- Matrix(hess_diff)
image(hess_diff)

library(ggplot2)
library(dplyr)
library(reshape2)

py_lrvb_cov <- json_dat$lrvb_cov[mp_permutation, mp_permutation]
plot(diag(py_lrvb_cov), diag(r_lrvb_results$lrvb_cov)); abline(0, 1)
plot(py_lrvb_cov, r_lrvb_results$lrvb_cov); abline(0, 1)
# plot(diag(r_lrvb_results_orig$lrvb_cov), diag(r_lrvb_results$lrvb_cov)); abline(0, 1)

cbind(diag(py_lrvb_cov), diag(r_lrvb_results$lrvb_cov))

lrvb_df <- rbind(
  SummarizeVBResults(GetMomentParametersFromVector(mp_opt, diag(py_lrvb_cov), FALSE), method="lrvb", metric="python"),
  SummarizeVBResults(GetMomentParametersFromVector(mp_opt, diag(r_lrvb_results$lrvb_cov), FALSE), method="lrvb", metric="r")) %>%
  dcast(par + component + group + method ~ metric, value.var="val")
ggplot(lrvb_df) + geom_point(aes(x=r, y=python)) + geom_abline(aes(slope=1, intercept=0))


# Define a permutation of the prior indices to map from the python indices to the R indices.
r_pp_indices <- GetPriorParametersFromVector(pp, as.numeric(1:pp$encoded_size), FALSE)


if (FALSE) {
  r_lrvb_results <- GetLRVBResults(y, y_g, x, vp_opt, pp, opt)
  plot(diag(json_dat$lrvb_cov[mp_permutation, mp_permutation]), diag(r_lrvb_results$lrvb_cov)); abline(0, 1)
  plot(json_dat$elbo_hess, r_lrvb_results$elbo_hess); abline(0, 1)
  
  
  ####################
  # Hmm check priors
  
  comb_indices <- GetPriorsAndNaturalParametersFromVector(
    vp_opt, pp, as.numeric(1:(vp_opt$encoded_size + pp$encoded_size)), FALSE)
  comb_prior_ind <- GetPriorParametersVector(comb_indices$pp, FALSE)
  comb_vp_ind <- GetNaturalParameterVector(comb_indices$vp, FALSE)
  
  opt$calculate_hessian <- TRUE
  log_prior_derivs <- GetFullModelLogPriorDerivatives(vp_opt, pp, opt)
  log_prior_hess <- log_prior_derivs$hess[comb_prior_ind, comb_vp_ind]
  
  plot(t(log_prior_hess), json_dat$log_prior_hess)
  dim(log_prior_hess)
  dim(json_dat$log_prior_hess)
}