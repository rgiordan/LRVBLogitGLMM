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

prior_sens_cast <- dcast(
  prior_sens_df, par + component + group + prior_par + k1 + k2 + metric ~ method, value.var="val")



##################
# Influence functions

mp_draw <- mp_opt
log_q_grad <- rep(0, vp_indices$encoded_size)
GetLogVariationalDensity <- function(u) {
  beta_q_derivs <- GetBetaLogDensity(u, vp_opt, mp_draw, pp, TRUE, TRUE)
  log_q_grad[global_mask] <- beta_q_derivs$grad
  list(val=beta_q_derivs$val, grad=log_q_grad)
}


# beta_funs <- GetBetaImportanceFunctions(beta_comp, vp_opt, pp, lrvb_results)
# 
# # For debugging
# num_draws = 500
# num_mc_draws = 20
# DrawImportanceSamples = beta_funs$DrawU
# GetImportanceLogProb = beta_funs$GetULogDensity
# GetLogQGradTerms = beta_funs$GetLogQGradTerms
# GetLogQ = beta_funs$GetLogVariationalDensity
# GetLogPrior = beta_funs$GetLogPrior
# 
# u_draws <- DrawImportanceSamples(num_draws)
# 
# log_prior <- sapply(u_draws, GetLogPrior)
# log_q <- sapply(u_draws, GetLogQ)
# importance_lp <- sapply(u_draws, GetImportanceLogProb)
# 
# importance_lp_ratio <- log_prior - importance_lp
# influence_lp_ratio <- log_q - log_prior
# log_q_grad_terms <- GetLogQGradTerms(u_draws, num_mc_draws)
# 

GetVariationalInfluenceResults <- function(
  num_draws,
  num_mc_draws,
  DrawImportanceSamples,
  GetImportanceLogProb,
  GetLogQGradTerms,
  GetLogQ,
  GetLogPrior) {
  
  u_draws <- DrawImportanceSamples(num_draws)
  
  log_prior <- sapply(u_draws, GetLogPrior)
  log_q <- sapply(u_draws, GetLogQ)
  importance_lp <- sapply(u_draws, GetImportanceLogProb)

  importance_lp_ratio <- log_prior - importance_lp
  influence_lp_ratio <- log_q - log_prior
  log_q_grad_terms <- GetLogQGradTerms(u_draws, num_mc_draws)

  influence_fun  <- t(log_q_grad_terms) * exp(influence_lp_ratio)
  u_influence_mat <- (influence_fun ^ 2) * exp(importance_lp_ratio)
  u_influence_mat_pos <- ((influence_fun > 0) * influence_fun ^ 2) * exp(importance_lp_ratio)
  u_influence_mat_neg <- ((influence_fun < 0) * influence_fun ^ 2) * exp(importance_lp_ratio)
  
  worst_case <-
    sapply(1:ncol(influence_fun),
           function(ind) { sqrt(max(mean(u_influence_mat_pos[, ind]),
                                    mean(u_influence_mat_neg[, ind]))) })
  
  return(list(
    u_draws=u_draws,
    influence_fun=influence_fun,
    importance_lp_ratio=importance_lp_ratio,
    influence_lp_ratio=influence_lp_ratio,
    log_prior=log_prior,
    log_q=log_q,
    importance_lp=importance_lp,
    log_q_grad_terms=log_q_grad_terms,
    worst_case=worst_case))
}


beta_comp <- 1
beta_funs <- GetBetaImportanceFunctions(beta_comp, vp_opt, pp, lrvb_results)

beta_influence_results <- GetVariationalInfluenceResults(
  num_draws = 500,
  num_mc_draws = 20,
  DrawImportanceSamples = beta_funs$DrawU,
  GetImportanceLogProb = beta_funs$GetULogDensity,
  GetLogQGradTerms = beta_funs$GetLogQGradTerms,
  GetLogQ = beta_funs$GetLogVariationalDensity,
  GetLogPrior = beta_funs$GetLogPrior)


draws_mat <- vb_results$draws_mat
param_draws <- draws_mat[, beta_comp]
mcmc_funs <- GetMCMCInfluenceFunctions(param_draws, beta_funs$GetLogPriorVec)
mcmc_worst_case <- sapply(1:ncol(draws_mat), function(ind) { mcmc_funs$GetMCMCWorstCase(draws_mat[, ind]) })

worst_case_df <- rbind(
  SummarizeVBResults(GetMomentParametersFromVector(mp_opt, mcmc_worst_case, FALSE),
                     method="mcmc", metric=paste("beta", beta_comp, sep="")),
  SummarizeVBResults(GetMomentParametersFromVector(mp_opt, beta_influence_results$worst_case, FALSE),
                     method="lrvb", metric=paste("beta", beta_comp, sep=""))
)

worst_case_cast <- dcast(worst_case_df, par + component + group + metric ~ method, value.var="val")
ggplot(worst_case_cast) +
  geom_point(aes(x=mcmc, y=lrvb, color=par)) +
  geom_abline(aes(slope=1, intercept=0)) +
  expand_limits(x=0, y=0)

ind <- mp_indices$beta_e_vec[beta_comp]
ind <- mp_indices$beta_e_vec[2]

mcmc_influence <- mcmc_funs$GetMCMCInfluence(g_draws=draws_mat[, ind])
mcmc_conditional_mean_diff <- mcmc_funs$GetConditionalMeanDiff(g_draws=draws_mat[, ind])

grid.arrange(
  ggplot() +
    geom_line(aes(x=beta_influence_results$u_draws, y=beta_influence_results$influence_fun[, ind], color="lrvb")) +
    geom_line(aes(x=param_draws, y=mcmc_influence, color="mcmc"))
,  
  ggplot() +
    geom_line(aes(x=beta_influence_results$u_draws, y=exp(beta_influence_results$log_q), color="lrvb")) +
    geom_line(aes(x=param_draws, y=mcmc_funs$dens_at_draws$y, color="mcmc"))
 , 
  ggplot() +
    geom_line(aes(x=beta_influence_results$u_draws, y=beta_influence_results$log_q_grad_terms[ind, ], color="lrvb")) +
    geom_line(aes(x=param_draws, y=mcmc_conditional_mean_diff, color="mcmc"))
  , ncol=3
)

g_draws <- draws_mat[, ind]
ggplot() +
  geom_point(aes(x=param_draws, y=g_draws, color="draws"), alpha=0.3, size=2) +
  geom_line(aes(x=param_draws, y=mcmc_conditional_mean_diff + mean(g_draws), color="mcmc"), lwd=3) +
  geom_line(aes(x=beta_influence_results$u_draws, y=beta_influence_results$log_q_grad_terms[ind, ]+ mean(g_draws), color="lrvb"), lwd=3)
  
mean(g_draws)
mp_opt$beta_e_vec[2]

mcmc_worst_case[ind]
beta_influence_results$worst_case[ind]


#############################
# Unpack the results.

mcmc_sample <- extract(vb_results$stan_results$stan_sim)
lrvb_cov <- vb_results$lrvb_results$lrvb_cov
mfvb_cov <- GetCovariance(vp_opt)
vp_mom <- GetMomentParametersFromNaturalParameters(vp_opt)
lrvb_sd <- GetMomentParametersFromVector(vp_mom, sqrt(diag(lrvb_cov)), FALSE)
mfvb_sd <- GetMomentParametersFromVector(vp_mom, sqrt(diag(mfvb_cov)), FALSE)

results <- SummarizeResults(mcmc_sample, vp_mom, mfvb_sd, lrvb_sd)

if (save_results) {
  influence_cast_sub <- sample_n(influence_cast, 5000)
  mcmc_time <- as.numeric(vb_results$stan_results$mcmc_time, units="secs")
  vb_time <- as.numeric(vb_results$fit_time, units="secs")
  num_mcmc_draws <- nrow(as.matrix(vb_results$stan_results$stan_sim))
  num_logit_sims <- length(vb_results$opt$std_draws)
  num_obs <- vb_results$stan_results$stan_dat$N
  beta_dim <- vb_results$stan_results$stan_dat$K
  save(results, influence_cast_sub, prior_sens_cast, mp_opt,
       mcmc_time, vb_time, num_mcmc_draws, pp, num_logit_sims, num_obs, beta_dim,
       file=results_file)
}

stop("Graphs follow -- not executing.")



# Overall

ggplot(
  filter(results, metric == "mean") %>%
    dcast(par + component + group ~ method, value.var="val") %>%
    mutate(is_u = par == "u")
) +
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


# Sensitivity

ggplot(filter(prior_sens_cast, par != "u")) +
  geom_point(aes(x=lrvb_norm, y=mcmc_norm, color=par)) +
  geom_abline(aes(intercept=0, slope=1))

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


# Influence functions

p1 <- ggplot(influence_cast) +
  geom_point(aes(x=beta1, y=beta2, color=var_beta1_influence), alpha=0.2) +
  geom_point(aes(x=mp_opt$beta_e_vec[1], y=mp_opt$beta_e_vec[2]), color="red", size=2) +
  ggtitle(paste("Influence of beta prior on beta1 variance")) +
  scale_color_gradient2()
p2 <- ggplot(influence_cast) +
  geom_point(aes(x=beta1, y=beta2, color=E_beta1_beta1_influence), alpha=0.2) +
  geom_point(aes(x=mp_opt$beta_e_vec[1], y=mp_opt$beta_e_vec[2]), color="red", size=2) +
  ggtitle(paste("Influence of beta prior on beta1 beta1")) +
  scale_color_gradient2()
p3 <- ggplot(influence_cast) +
  geom_point(aes(x=beta1, y=beta2, color=E_beta1_influence), alpha=0.2) +
  geom_point(aes(x=mp_opt$beta_e_vec[1], y=mp_opt$beta_e_vec[2]), color="red", size=2) +
  ggtitle(paste("Influence of beta prior on beta1")) +
  scale_color_gradient2()
grid.arrange(p1, p2, p3, nrow=1)


