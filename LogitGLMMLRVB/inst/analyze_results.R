library(LogitGLMMLRVB)
library(rstan)
library(lme4)

library(dplyr)
library(reshape2)
library(trust)

library(LRVBUtils)
library(mvtnorm)

library(gridExtra)

project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "LRVBLogitGLMM")
source(file.path(project_directory, "LogitGLMMLRVB/inst/densities_lib.R"))

analysis_name <- "simulated_data_small"
# analysis_name <- "simulated_data_large"

data_directory <- file.path(project_directory, "LogitGLMMLRVB/inst/data/")
vb_results_file <- file.path(data_directory, paste(analysis_name, "_vb_results.Rdata", sep=""))

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

comb_indices <- GetPriorsAndNaturalParametersFromVector(
  vp_opt, pp, as.numeric(1:(vp_opt$encoded_size + pp$encoded_size)), FALSE)
comb_prior_ind <- GetPriorParametersVector(comb_indices$pp, FALSE)
comb_vp_ind <- GetNaturalParameterVector(comb_indices$vp, FALSE)

opt$calculate_hessian <- TRUE
log_prior_derivs <- GetFullModelLogPriorDerivatives(vp_opt, pp, opt)
log_prior_param_prior <- Matrix(log_prior_derivs$hess[comb_vp_ind, comb_prior_ind])

prior_sens <- -1 * lrvb_results$jac %*% Matrix::solve(lrvb_results$elbo_hess, log_prior_param_prior)

# Re-scaling for normalized sensitivities.
draws_mat <- t(vb_results$draws_mat)
lrvb_sd_scale <- sqrt(diag(vb_results$lrvb_results$lrvb_cov))
mcmc_sd_scale <- sqrt(diag(cov(t(draws_mat)))) 

# Get the MCMC covariance-based results
log_prior_grad_mat <- vb_results$log_prior_grad_mat
draws_mat <- draws_mat - rowMeans(draws_mat)
prior_sens_mcmc <- draws_mat %*% log_prior_grad_mat / nrow(log_prior_grad_mat)

# Keep a subset of the rows to simulate having made fewer MCMC draws.  For this subset, calculate
# standard deviations.  Just set keep_rows large to calculate standard deviations for all draws.
keep_rows <- min(c(nrow(log_prior_grad_mat), 50000))
draws_mat_small <- draws_mat[, 1:keep_rows]
log_prior_grad_mat_small <- log_prior_grad_mat[1:keep_rows, ]

mcmc_sd_scale_small <- sqrt(diag(cov(t(draws_mat_small)))) 
prior_sens_mcmc_small <- draws_mat_small  %*% log_prior_grad_mat_small / keep_rows
prior_sens_mcmc_squares <- (draws_mat_small ^ 2)  %*% (log_prior_grad_mat_small ^ 2) / keep_rows
prior_sens_mcmc_sd <- sqrt(prior_sens_mcmc_squares - prior_sens_mcmc_small ^ 2) / sqrt(keep_rows)

draws_mat_small_norm <- draws_mat_small / mcmc_sd_scale_small
prior_sens_mcmc_norm_small <- draws_mat_small_norm  %*% log_prior_grad_mat_small / keep_rows
prior_sens_mcmc_norm_squares <- (draws_mat_small_norm ^ 2)  %*% (log_prior_grad_mat_small ^ 2) / keep_rows
prior_sens_mcmc_norm_sd <- sqrt(prior_sens_mcmc_norm_squares - prior_sens_mcmc_norm_small ^ 2) / sqrt(keep_rows)

# Combine.
prior_sens_df <- rbind(
  UnpackPriorSensitivityMatrix(prior_sens, pp_indices, method="lrvb"),
  UnpackPriorSensitivityMatrix(prior_sens_mcmc, pp_indices, method="mcmc"),
  UnpackPriorSensitivityMatrix(prior_sens_mcmc_small, pp_indices, method="mcmc_small"),
  UnpackPriorSensitivityMatrix(prior_sens_mcmc_sd, pp_indices, method="mcmc_small_sd"),
  
  UnpackPriorSensitivityMatrix(prior_sens / lrvb_sd_scale, pp_indices, method="lrvb_norm"),
  UnpackPriorSensitivityMatrix(prior_sens_mcmc / mcmc_sd_scale, pp_indices, method="mcmc_norm"),
  UnpackPriorSensitivityMatrix(prior_sens_mcmc_norm_small, pp_indices, method="mcmc_norm_small"),
  UnpackPriorSensitivityMatrix(prior_sens_mcmc_norm_sd, pp_indices, method="mcmc_norm_small_sd"))

# Aggregate across different prior components.  This analysis treats each prior component
# separately, but it's easier to graph and understand if when we change one component of beta_loc or
# beta_info we change all of them.
prior_sens_agg <- prior_sens_df %>%
  filter(k2 == -1 | k1 == k2) %>% # Remove the off-diagonal beta_info sensitivities.
  ungroup() %>% group_by(par, component, group, method, metric, prior_par) %>%
  summarize(val=sum(val))

prior_sens_cast <- dcast(
  prior_sens_agg, par + component + group + prior_par + metric ~ method, value.var="val")



##################
# Influence functions

draws_mat <- vb_results$draws_mat
worst_case_list <- list()
for (beta_comp in 1:vp_opt$k_reg) {
  cat("beta_comp ", beta_comp, "\n")
  beta_funs <- GetBetaImportanceFunctions(beta_comp, vp_opt, pp, lrvb_results)
  num_mc_draws <- 10
  beta_influence_results <- GetVariationalInfluenceResults(
    num_draws = 200,
    DrawImportanceSamples = beta_funs$DrawU,
    GetImportanceLogProb = beta_funs$GetULogDensity,
    GetLogQGradTerms = function(u_draws) { beta_funs$GetLogQGradTerms(u_draws, num_mc_draws, normalize=TRUE) },
    GetLogQ = beta_funs$GetLogVariationalDensity,
    GetLogPrior = beta_funs$GetLogPrior)
  
  
  # Get MCMC worst-case
  param_draws <- draws_mat[, beta_comp]
  mcmc_funs <- GetMCMCInfluenceFunctions(param_draws, beta_funs$GetLogPrior)
  GetMCMCWorstCaseColumn <- function(col) { mcmc_funs$GetMCMCWorstCase(draws_mat[, col]) }
  mcmc_worst_case <- sapply(1:ncol(draws_mat), GetMCMCWorstCaseColumn)
  
  # Compare
  worst_case_list[[length(worst_case_list) + 1]] <- rbind(
    SummarizeVBResults(GetMomentParametersFromVector(mp_opt, mcmc_worst_case, FALSE),
                       method="mcmc", metric=paste("beta", beta_comp, sep="")),
    SummarizeVBResults(GetMomentParametersFromVector(mp_opt, beta_influence_results$worst_case, FALSE),
                       method="lrvb", metric=paste("beta", beta_comp, sep=""))
  )
}

worst_case_df <- do.call(rbind, worst_case_list)



###################################
# Get graphs of influence functions

beta_comp <- 5
beta_funs <- GetBetaImportanceFunctions(beta_comp, vp_opt, pp, lrvb_results)
beta_influence_results <- GetVariationalInfluenceResults(
  num_draws = 200,
  DrawImportanceSamples = beta_funs$DrawU,
  GetImportanceLogProb = beta_funs$GetULogDensity,
  GetLogQGradTerms = function(u_draws) { beta_funs$GetLogQGradTerms(u_draws, num_mc_draws, normalize=TRUE) },
  GetLogQ = beta_funs$GetLogVariationalDensity,
  GetLogPrior = beta_funs$GetLogPrior)


# Get MCMC worst-case
param_draws <- draws_mat[, mp_indices$beta_e_vec[beta_comp]]
mcmc_funs <- GetMCMCInfluenceFunctions(param_draws, beta_funs$GetLogPrior)
GetMCMCWorstCaseColumn <- function(col) { mcmc_funs$GetMCMCWorstCase(draws_mat[, col]) }
mcmc_worst_case <- sapply(1:ncol(draws_mat), GetMCMCWorstCaseColumn)

GetInfluenceDF <- function(u, influence, gbar, log_posterior, log_prior, worst_u, method, metric) {
  data.frame(u=u, influence=influence, gbar=gbar,
             log_posterior=log_posterior, log_prior=log_prior,
             worst_u=worst_u,
             method=method, metric=metric)  
}

# ind <- mp_indices$beta_e_vec[beta_comp]
# ind <- mp_indices$beta_e_vec[4]
ind <- mp_indices$beta_e_vec[beta_comp]
metric <- sprintf("beta%d_on_ind%d", beta_comp, ind)
g_draws <- draws_mat[, ind]

mcmc_worst <- mcmc_funs$GetMCMCWorstCaseResults(g_draws)
mcmc_influence_df <- GetInfluenceDF(
  u=mcmc_funs$param_draws,
  influence=mcmc_worst$mcmc_influence,
  gbar=mcmc_funs$GetConditionalMeanDiff(g_draws),
  log_posterior=log(mcmc_funs$dens_at_draws),
  log_prior=mcmc_funs$log_prior_at_draws,
  worst_u=mcmc_worst$worst_u,
  method="mcmc",
  metric=metric)

vb_influence_df <- GetInfluenceDF(
  u=beta_influence_results$u_draws,
  influence=beta_influence_results$influence_fun[, ind],
  gbar=beta_influence_results$log_q_grad_terms[ind, ],
  log_posterior=beta_influence_results$log_q,
  log_prior=beta_influence_results$log_prior,
  worst_u=beta_influence_results$worst_case_u[, ind],
  method="lrvb",
  metric=metric)

if (FALSE) {
  grid.arrange(
    ggplot() +
      geom_line(data=vb_influence_df, aes(x=u, y=influence, color=method), lwd=2) +
      geom_line(data=mcmc_influence_df, aes(x=u, y=influence, color=method), lwd=2) 
    , 
    ggplot() +
      geom_line(data=vb_influence_df, aes(x=u, y=gbar, color=method), lwd=2) +
      geom_line(data=mcmc_influence_df, aes(x=u, y=gbar, color=method), lwd=2) 
    , 
    ggplot() +
      geom_line(data=vb_influence_df, aes(x=u, y=exp(log_posterior), color=method), lwd=2) +
      geom_line(data=mcmc_influence_df, aes(x=u, y=exp(log_posterior), color=method), lwd=2) +
      geom_line(data=vb_influence_df, aes(x=u, y=exp(log_prior), color="prior"), lwd=2) +
      geom_line(data=mcmc_influence_df, aes(x=u, y=exp(log_prior), color="prior"), lwd=2) 
    , ncol=3
  )
}


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
  num_mcmc_draws <- nrow(as.matrix(vb_results$stan_results$stan_sim))
  num_logit_sims <- length(vb_results$opt$std_draws)
  num_obs <- vb_results$stan_results$stan_dat$N
  beta_dim <- vb_results$stan_results$stan_dat$K
  elbo_hess_sparsity <- Matrix(abs(lrvb_results$elbo_hess) > 1e-8)
  save(results, prior_sens_cast, mp_opt,
       mcmc_time, vb_time, num_mcmc_draws, pp, num_logit_sims, num_obs, beta_dim,
       worst_case_df, vb_influence_df, mcmc_influence_df,
       elbo_hess_sparsity,
       file=results_file)
}


########################################
# Graphs and analysis

stop("Graphs follow -- not executing.")



grid.arrange(
  ggplot() +
    geom_line(data=vb_influence_df, aes(x=u, y=influence, color=method), lwd=2) +
    geom_line(data=mcmc_influence_df, aes(x=u, y=influence, color=method), lwd=2) 
  , 
  ggplot() +
    geom_line(data=vb_influence_df, aes(x=u, y=gbar, color=method), lwd=2) +
    geom_line(data=mcmc_influence_df, aes(x=u, y=gbar, color=method), lwd=2) 
  , 
  ggplot() +
    geom_line(data=vb_influence_df, aes(x=u, y=exp(log_posterior), color=method), lwd=2) +
    geom_line(data=mcmc_influence_df, aes(x=u, y=exp(log_posterior), color=method), lwd=2) 
  , ncol=3
)


# Look at the draws underlying the conditional expectation estimates
ggplot() +
  geom_point(aes(x=param_draws, y=g_draws, color="draws"), alpha=0.3, size=2) +
  geom_line(data=mcmc_influence_df, aes(x=u, y=gbar + mean(g_draws), color=method), lwd=2) 

# Look at the worst-case perturbation
prior_scale <- 1 / sqrt(diag(pp$beta_info))[beta_comp]
prior_loc <- pp$beta_loc[beta_comp]
prior_draws <- seq(-2 * prior_scale + prior_loc, 2 * prior_scale + prior_loc, length.out=2000)
log_prior <- beta_funs$GetLogPrior(prior_draws)
prior_df <- data.frame(u=prior_draws, log_prior=log_prior)
# ggplot() + 
#   # geom_line(aes(x=beta_influence_results$u_draws, y=beta_influence_results$worst_case_u[, ind], color="ustar"), lwd=2) +
#   geom_line(aes(x=beta_influence_results$u_draws, y=beta_influence_results$influence_fun[, ind], color="influence"), lwd=2) +
#   geom_line(data=prior_df, aes(x=u, y=exp(log_prior), color="prior"), lwd=2)

influence_approx_fun <- with(beta_influence_result, approxfun(u_draws, influence_fun[, ind]))
prior_df$influence <- influence_approx_fun(prior_df$u)
prior_df$influence[is.na(prior_df$influence)] <- 0

log_q_approx_fun <- with(beta_influence_results, approxfun(u_draws, log_q))
prior_df$log_q <- log_q_approx_fun(prior_df$u)
prior_df$log_q[is.na(prior_df$log_q)] <- -Inf

ustar_approx_fun <- with(beta_influence_results, approxfun(u_draws, worst_case_u[, ind]))
prior_df$ustar <- ustar_approx_fun(prior_df$u)
prior_df$ustar[is.na(prior_df$ustar)] <- 0

grid.arrange(
  ggplot(prior_df) + 
    geom_line(aes(x=u, y=exp(log_prior), color="prior"), lwd=2)
,
  ggplot(prior_df) + 
    geom_line(aes(x=u, y=exp(log_q), color="log_q"), lwd=2)
,
  ggplot(prior_df) + 
    geom_line(aes(x=u, y=influence, color="influence"), lwd=2)
,
  ggplot(prior_df) + 
    geom_line(aes(x=u, y=ustar, color="ustar"), lwd=2)
, ncol=1
)


worst_case_cast <- dcast(worst_case_df, par + component + group + metric ~ method, value.var="val")
ggplot(worst_case_cast) +
  geom_point(aes(x=mcmc, y=lrvb, color=par, shape=factor(component)), size=2) +
  geom_abline(aes(slope=1, intercept=0)) +
  expand_limits(x=0, y=0) +
  facet_grid( ~ metric)


# Overall
ggplot(
  filter(results, metric == "mean") %>%
    dcast(par + component + group ~ method, value.var="val") %>%
    mutate(is_u = par == "u")) +
  geom_point(aes(x=truth, y=mcmc, shape=par, color="mcmc"), size=3) +
  geom_point(aes(x=truth, y=mfvb, shape=par, color="mfvb"), size=3) +
  geom_point(aes(x=truth, y=map, shape=par, color="map"), size=3) +
  geom_abline(aes(intercept=0, slope=1)) +
  facet_grid(~ is_u)


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


ggplot(
  filter(results, metric == "sd", par != "u") %>%
    dcast(par + component + group ~ method, value.var="val")
) +
  geom_point(aes(x=mcmc, y=map, color="map", shape=par), size=3) +
  expand_limits(x=0, y=0) +
  xlab("MCMC (ground truth)") + ylab("MAP") +
  scale_color_discrete(guide=guide_legend(title="Method")) +
  geom_abline(aes(intercept=0, slope=1))


# Sensitivity

grid.arrange(
ggplot(filter(prior_sens_cast, par != "u")) +
  geom_point(aes(x=lrvb_norm, y=mcmc_norm, color=par)) +
  geom_abline(aes(intercept=0, slope=1))
,
ggplot(filter(prior_sens_cast, par != "u")) +
  geom_point(aes(x=lrvb, y=mcmc, color=par)) +
  geom_abline(aes(intercept=0, slope=1))
, ncol=2)

# Compare LRVB with the MCMC standard deviations
ggplot(filter(prior_sens_cast, par=="u")) +
  geom_point(aes(x=lrvb_norm, y=mcmc_norm_small, color=prior_par)) +
  geom_errorbar(aes(x=lrvb_norm,
                    ymin=mcmc_norm_small - 2 * mcmc_norm_small_sd,
                    ymax=mcmc_norm_small + 2 * mcmc_norm_small_sd,
                    color=prior_par)) +
  geom_abline(aes(intercept=0, slope=1))

# Compare MCMC with its own estimated standard deviations.
ggplot(filter(prior_sens_cast, par=="u")) +
  geom_point(aes(x=mcmc_norm, y=mcmc_norm_small, color=prior_par)) +
  geom_errorbar(aes(x=mcmc_norm,
                    ymin=mcmc_norm_small - 2 * mcmc_norm_small_sd,
                    ymax=mcmc_norm_small + 2 * mcmc_norm_small_sd,
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
