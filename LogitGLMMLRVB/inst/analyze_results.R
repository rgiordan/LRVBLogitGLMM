library(LogitGLMMLRVB)
library(rstan)
library(lme4)

library(dplyr)
library(reshape2)
library(trust)

library(LRVBUtils)
library(mvtnorm)

library(gridExtra)

library(jsonlite)

project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "LRVBLogitGLMM")
source(file.path(project_directory, "LogitGLMMLRVB/inst/densities_lib.R"))

#analysis_name <- "simulated_data_small"
#analysis_name <- "simulated_data_large"
analysis_name <- "criteo_subsampled"

data_directory <- file.path(project_directory, "LogitGLMMLRVB/inst/data/")

# Read from R or python.  If reading from python, make sure to run read_vb_python.R first.
# vb_results_file <- file.path(data_directory, paste(analysis_name, "_vb_results.Rdata", sep=""))
vb_results_file <- file.path(data_directory, paste(analysis_name, "_vb_python_results.Rdata", sep=""))


# If true, save the results to a file readable by knitr.
save_results <- TRUE
results_file <- file.path(data_directory,
                          paste(analysis_name, "sensitivity.Rdata", sep="_"))

vb_results <- LoadIntoEnvironment(vb_results_file)

vp_opt <- vb_results$vp_opt
mp_opt <- vb_results$mp_opt
opt <- vb_results$opt
pp <- vb_results$stan_results$pp
lrvb_results <- vb_results$lrvb_results

#############################
# Indices

pp_indices <- GetPriorParametersFromVector(pp, as.numeric(1:pp$encoded_size), FALSE)
vp_indices <- GetNaturalParametersFromVector(vp_opt, as.numeric(1:vp_opt$encoded_size), FALSE)
mp_indices <- GetMomentParametersFromVector(mp_opt, as.numeric(1:mp_opt$encoded_size), FALSE)

global_mask <- rep(FALSE, vp_opt$encoded_size)
global_indices <- unique(c(vp_indices$beta_loc, as.numeric(vp_indices$beta_info[]),
                           vp_indices$mu_loc, vp_indices$mu_info,
                           vp_indices$tau_alpha, vp_indices$tau_beta))
global_mask[global_indices] <- TRUE

#################################
# Parametric sensitivity analysis

# It would be better to group the beta_loc and beta_info parameters into single
# prior parameters all at once at the beginning.

log_prior_hess <- t(vb_results$log_prior_hess)

prior_sens <- -1 * lrvb_results$jac %*% Matrix::solve(lrvb_results$elbo_hess, log_prior_hess)

# Re-scaling for normalized sensitivities.
draws_mat <- t(vb_results$draws_mat)
lrvb_sd_scale <- sqrt(diag(vb_results$lrvb_results$lrvb_cov))
mcmc_sd_scale <- sqrt(diag(cov(t(draws_mat)))) 

# Get the MCMC covariance-based results
log_prior_grad_mat <- vb_results$log_prior_grad_mat
log_prior_grad_mat <- log_prior_grad_mat - rep(colMeans(log_prior_grad_mat), each=nrow(log_prior_grad_mat))
draws_mat <- draws_mat - rowMeans(draws_mat)
prior_sens_mcmc <- draws_mat %*% log_prior_grad_mat / nrow(log_prior_grad_mat)
num_mcmc_draws <- nrow(log_prior_grad_mat)

# Keep a subset of the rows to simulate having made fewer MCMC draws.  For this subset, calculate
# standard deviations.  Just set keep_rows large to calculate standard deviations for all draws.
# keep_rows <- min(c(nrow(log_prior_grad_mat), 50000))
# draws_mat_small <- draws_mat[, 1:keep_rows]
# log_prior_grad_mat_small <- log_prior_grad_mat[1:keep_rows, ]

# mcmc_sd_scale_small <- sqrt(diag(cov(t(draws_mat_small)))) 
# prior_sens_mcmc_small <- draws_mat_small  %*% log_prior_grad_mat_small / keep_rows
prior_sens_mcmc_squares <- (draws_mat ^ 2)  %*% (log_prior_grad_mat ^ 2) / num_mcmc_draws
prior_sens_mcmc_sd <- sqrt(prior_sens_mcmc_squares - prior_sens_mcmc ^ 2) / sqrt(num_mcmc_draws)

draws_mat_norm <- draws_mat / mcmc_sd_scale
prior_sens_mcmc_norm <- draws_mat_norm  %*% log_prior_grad_mat / num_mcmc_draws
prior_sens_mcmc_norm_squares <- (draws_mat_norm ^ 2)  %*% (log_prior_grad_mat ^ 2) / num_mcmc_draws
prior_sens_mcmc_norm_sd <- sqrt(prior_sens_mcmc_norm_squares - prior_sens_mcmc_norm ^ 2) / sqrt(num_mcmc_draws)

# Unpack the results into dataframes  Note that all
# the sensitivities have already been calculated at this point, but this is slow due to a lot of
# R munging that I have never bothered to tidy up.
prior_sens_df <- rbind(
  UnpackPriorSensitivityMatrix(prior_sens / lrvb_sd_scale, pp_indices, method="lrvb_norm"),
  UnpackPriorSensitivityMatrix(prior_sens_mcmc / mcmc_sd_scale, pp_indices, method="mcmc_norm"))

prior_sens_sd_df <- UnpackPriorSensitivityMatrix(prior_sens_mcmc_norm_sd, pp_indices, method="mcmc_norm_sd")

# Aggregate across different prior components.  This analysis treats each prior component
# separately, but it's easier to graph and understand if when we change one component of beta_loc or
# beta_info we change all of them.
prior_sens_agg <- prior_sens_df %>%
  filter(k2 == -1 | k1 == k2) %>% # Remove the off-diagonal beta_info sensitivities.
  ungroup() %>% group_by(par, component, group, method, metric, prior_par) %>%
  summarize(val=sum(val))

prior_sens_cast <- dcast(
  prior_sens_agg, par + component + group + prior_par + metric ~ method, value.var="val")




#############################
# Unpack the results.

StanParToMomentParams <- function(par, bracket=TRUE) {
  par_mp <- GetMomentParametersFromVector(mp_opt, rep(NaN, mp_opt$encoded_size), unconstrained=TRUE)
  if (bracket) {
    beta <- par[sprintf("beta[%d]", 1:vp_opt$k_reg)]
    u <- par[sprintf("u[%d]", 1:vp_opt$n_groups)]
  } else {
    beta <- par[sprintf("beta.%d", 1:vp_opt$k_reg)]
    u <- par[sprintf("u.%d", 1:vp_opt$n_groups)]
  }
  mu <- par["mu"]
  tau <- par["tau"]
  par_mp$beta_e_vec <- beta
  par_mp$beta_e_outer <- beta %*% t(beta)
  par_mp$mu_e <- mu
  par_mp$mu_e2 <- mu^2
  par_mp$tau_e <- tau
  par_mp$tau_e_log <- log(tau)
  for (g in 1:(vp_opt$n_groups)) {
    par_mp$u[[g]]$u_e <- u[g]
    par_mp$u[[g]]$u_e2 <- u[g]^2
  }
  return(par_mp)  
}

# The MAP estimate
stan_map <- vb_results$stan_results$stan_map 
map_mp <- StanParToMomentParams(stan_map$par)
inv_hess_diag <- -diag(solve(stan_map$hessian))
map_sd_mp <- StanParToMomentParams(sqrt(inv_hess_diag), bracket=FALSE)
# If we wanted the sds of the squares or log, we'd need a delta method.  Not needed, though.
map_sd_mp$beta_e_outer[] <- NaN
map_sd_mp$tau_e_log <- NaN
map_sd_mp$mu_e2 <- NaN
for (g in 1:(vp_opt$n_groups)) {
  map_sd_mp$u[[g]]$u_e2 <- NaN
}

map_results <- rbind(
  SummarizeVBResults(map_mp, "map", "mean"),
  SummarizeVBResults(map_sd_mp, "map", "sd"))

# The truth
true_params <- vb_results$stan_results$true_params
true_mp <- GetMomentParametersFromVector(mp_opt, rep(NaN, mp_opt$encoded_size), unconstrained=TRUE)
true_mp$beta_e_vec <- true_params$beta
true_mp$beta_e_outer <- true_params$beta %*% t(true_params$beta)
true_mp$mu_e <- true_params$mu
true_mp$mu_e2 <- true_params$mu^2
true_mp$tau_e <- true_params$tau
true_mp$tau_e_log <- log(true_params$tau)
for (g in 1:(vp_opt$n_groups)) {
  true_mp$u[[g]]$u_e <- true_params$u[[g]]
  true_mp$u[[g]]$u_e2 <- true_params$u[[g]]^2
}

# MCMC and VB
mcmc_sample <- extract(vb_results$stan_results$stan_sim)
lrvb_cov <- vb_results$lrvb_results$lrvb_cov
mfvb_cov <- GetCovariance(vp_opt)
vp_mom <- GetMomentParametersFromNaturalParameters(vp_opt)
lrvb_sd <- GetMomentParametersFromVector(vp_mom, sqrt(diag(lrvb_cov)), FALSE)
mfvb_sd <- GetMomentParametersFromVector(vp_mom, sqrt(diag(mfvb_cov)), FALSE)

results <-
  rbind(SummarizeResults(mcmc_sample, vp_mom, mfvb_sd, lrvb_sd),
        SummarizeVBResults(true_mp, "truth", "mean"),
        map_results)

if (save_results) {
  mcmc_time <- as.numeric(vb_results$stan_results$mcmc_time, units="secs")
  vb_time <- as.numeric(vb_results$fit_time, units="secs")
  hess_time <- as.numeric(vb_results$hess_time, units="secs")
  num_mcmc_draws <- nrow(as.matrix(vb_results$stan_results$stan_sim))
  num_logit_sims <- vb_results$num_mc_draws
  num_obs <- vb_results$stan_results$stan_dat$N
  beta_dim <- vb_results$stan_results$stan_dat$K
  elbo_hess_sparsity <- Matrix(abs(lrvb_results$elbo_hess) > 1e-8)
  save(results, prior_sens_cast, mp_opt,
       mcmc_time, vb_time, num_mcmc_draws, num_mc_draws, hess_time,
       pp, num_logit_sims, num_obs, beta_dim,
       elbo_hess_sparsity,
       file=results_file)
}


########################################
# Graphs and analysis

stop("Graphs follow -- not executing.")

# Overall

ggplot(
  filter(results, metric == "mean") %>%
    dcast(par + component + group ~ method, value.var="val") %>%
    mutate(is_u = par == "u")) +
  geom_point(aes(x=mcmc, y=mfvb, color=par), size=3) +
  geom_abline(aes(intercept=0, slope=1)) +
  facet_grid(~ is_u)

ggplot(
  filter(results, metric == "sd") %>%
    dcast(par + component + group ~ method, value.var="val") %>%
    mutate(is_u = par == "u")) +
  geom_point(aes(x=mcmc, y=mfvb, shape=par, color="mfvb"), size=3) +
  geom_point(aes(x=mcmc, y=lrvb, shape=par, color="lrvb"), size=3) +
  geom_abline(aes(intercept=0, slope=1)) +
  facet_grid(~ is_u) +
  ggtitle("Posterior standard deviations")

ggplot(
  filter(results, metric == "sd", par != "u") %>%
    dcast(par + component + group ~ method, value.var="val")
) +
  geom_point(aes(x=mcmc, y=mfvb, color="mfvb", shape=par), size=3) +
  geom_point(aes(x=mcmc, y=lrvb, color="lrvb", shape=par), size=3) +
  expand_limits(x=0, y=0) +
  xlab("MCMC (ground truth)") + ylab("VB") +
  scale_color_discrete(guide=guide_legend(title="Method")) +
  geom_abline(aes(intercept=0, slope=1))

ggplot(
  filter(results, metric == "sd", par == "mu") %>%
    dcast(par + component + group ~ method, value.var="val")
) +
  geom_point(aes(x=mcmc, y=mfvb, color="mfvb", shape=par), size=3) +
  geom_point(aes(x=mcmc, y=lrvb, color="lrvb", shape=par), size=3) +
  expand_limits(x=0, y=0) +
  xlab("MCMC (ground truth)") + ylab("VB") +
  scale_color_discrete(guide=guide_legend(title="Method")) +
  geom_abline(aes(intercept=0, slope=1))


ggplot(
  filter(results, metric == "sd", par != "u") %>%
    dcast(par + component + group ~ method, value.var="val")
) +
  geom_point(aes(x=mcmc, y=map, color="map", shape=par), size=3) +
  expand_limits(x=0, y=0) +
  xlab("MCMC (ground truth)") + ylab("MAP") +
  scale_color_discrete(guide=guide_legend(title="Method")) +
  geom_abline(aes(intercept=0, slope=1))

ggplot(
  filter(results, metric == "mean") %>%
    dcast(par + component + group ~ method, value.var="val") %>%
    mutate(is_u = par == "u")) +
  geom_point(aes(x=truth, y=mcmc, shape=par, color="mcmc"), size=3) +
  geom_point(aes(x=truth, y=mfvb, shape=par, color="mfvb"), size=3) +
  geom_point(aes(x=truth, y=map, shape=par, color="map"), size=3) +
  geom_abline(aes(intercept=0, slope=1)) +
  facet_grid(~ is_u)


# Sensitivity

ggplot(filter(prior_sens_cast, par != "u")) +
  geom_point(aes(x=lrvb_norm, y=mcmc_norm, color=par)) +
  geom_abline(aes(intercept=0, slope=1))


# Note: mcmc_norm_sd is not here because it doens't work with the aggregation.

# Compare LRVB with the MCMC standard deviations
ggplot(filter(prior_sens_cast, par=="u")) +
  geom_point(aes(x=lrvb_norm, y=mcmc_norm, color=prior_par)) +
  geom_errorbar(aes(x=lrvb_norm,
                    ymin=mcmc_norm - 2 * mcmc_norm_sd,
                    ymax=mcmc_norm + 2 * mcmc_norm_sd,
                    color=prior_par)) +
  geom_abline(aes(intercept=0, slope=1))

# Compare MCMC with its own estimated standard deviations.
ggplot(filter(prior_sens_cast, par=="u")) +
  geom_point(aes(x=mcmc_norm, y=mcmc_norm, color=prior_par)) +
  geom_errorbar(aes(x=mcmc_norm,
                    ymin=mcmc_norm - 2 * mcmc_norm_sd,
                    ymax=mcmc_norm + 2 * mcmc_norm_sd,
                    color=prior_par)) +
  geom_abline(aes(intercept=0, slope=1))

ggplot(filter(prior_sens_cast, par=="u")) +
  geom_point(aes(x=mcmc_norm, y=mcmc_norm_small, color=prior_par)) +
  geom_errorbar(aes(x=mcmc_norm,
                    ymin=mcmc_norm_small - 2 * mcmc_norm_small_sd,
                    ymax=mcmc_norm_small + 2 * mcmc_norm_small_sd,
                    color=prior_par)) +
  geom_abline(aes(intercept=0, slope=1))

ggplot(filter(prior_sens_cast, par=="u")) +
  geom_point(aes(x=mcmc, y=mcmc_small, color=prior_par)) +
  geom_errorbar(aes(x=mcmc_norm,
                    ymin=mcmc_small - 2 * mcmc_small_sd,
                    ymax=mcmc_small + 2 * mcmc_small_sd,
                    color=prior_par)) +
  geom_abline(aes(intercept=0, slope=1))
