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
# source(file.path(project_directory, "LogitGLMMLRVB/inst/optimize_lib.R"))

analysis_name <- "simulated_data"
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


# # Monte Carlo samples
# n_samples <- 50000
# 
# # Define functions necessary to compute influence function stuff
# 
# # Just for testing
# draw <- mp_opt
# beta <- c(1.2, 2.0)
# 
# GetBetaLogPrior <- function(beta, pp) {
#   # You can't use the VB priors because they are
#   # (1) a function of the natural parameters whose variance would have to be zero and
#   # (2) not normalized.
#   dmvnorm(beta, mean=pp$beta_loc, sigma=solve(pp$beta_info), log=TRUE)
# }
# 
# 
# GetBetaLogDensity <- function(beta, vp_opt, draw, pp, unconstrained, calculate_gradient) {
#   draw$beta_e_vec <- beta
#   draw$beta_e2_vec <- beta %*% t(beta)
#   opt$calculate_gradient <- calculate_gradient
#   opt$calculate_hessian <- FALSE
#   q_derivs <- GetLogVariationalDensityDerivatives(draw, vp_opt, opt, global_only=TRUE,
#                                                   include_beta=TRUE, include_mu=FALSE, include_tau=FALSE)
#   return(q_derivs)
# }
# 
# 
# # You could also do this more numerically stably with a Cholesky decomposition.
# lrvb_pre_factor <- -1 * lrvb_results$jac %*% solve(lrvb_results$elbo_hess)
# 
# # Proposals based on q
# u_mean <- mp_opt$beta_e_vec
# # Increase the covariance for sampling.  How much is enough?
# u_cov <- (1.5 ^ 2) * solve(vp_opt$beta_info)
# GetULogDensity <- function(beta) {
#   dmvnorm(beta, mean=u_mean, sigma=u_cov, log=TRUE)
# }
# 
# 
# DrawU <- function(n_samples) {
#   rmvnorm(n_samples, mean=u_mean, sigma=u_cov)
# }
# u_draws <- DrawU(n_samples)
# 
# 
# GetLogPrior <- function(u) {
#   GetBetaLogPrior(u, pp)
# }
# 
# 
# mp_draw <- mp_opt
# log_q_grad <- rep(0, vp_indices$encoded_size)
# GetLogVariationalDensity <- function(u) {
#   beta_q_derivs <- GetBetaLogDensity(u, vp_opt, mp_draw, pp, TRUE, TRUE)
#   log_q_grad[global_mask] <- beta_q_derivs$grad
#   list(val=beta_q_derivs$val, grad=log_q_grad)
# }
# 
# GetLogVariationalDensity(beta)
# 
# GetInfluenceFunctionSample <- GetInfluenceFunctionSampleFunction(
#   GetLogVariationalDensity, GetLogPrior, GetULogDensity, lrvb_pre_factor)
# 
# GetInfluenceFunctionSample(u_draws[1, ])
# 
# influence_list <- list()
# pb <- txtProgressBar(min=1, max=nrow(u_draws), style=3)
# for (ind in 1:nrow(u_draws)) {
#   setTxtProgressBar(pb, ind)
#   influence_list[[ind]] <- GetInfluenceFunctionSample(u_draws[ind, ])
# }
# close(pb)
# 
# 
# names(influence_list[[1]])
# influence_vector_list <- lapply(influence_list, function(x) as.numeric(x$influence_function))
# influence_matrix <- do.call(rbind, influence_vector_list)
# 
# mp_opt_vector <- GetMomentParameterVector(mp_opt, FALSE)
# GetIndexRow <- function(ind, param_name) {
#   data.frame(ind=ind, param_name=param_name, val=mp_opt_vector[ind])
# }
# inds  <- data.frame()
# inds <- rbind(inds, GetIndexRow(mp_indices$beta_e_vec[1], param_name <- "E_beta1"))
# inds <- rbind(inds, GetIndexRow(mp_indices$beta_e_vec[2], param_name <- "E_beta2"))
# inds <- rbind(inds, GetIndexRow(mp_indices$beta_e_outer[1, 1], param_name <- "E_beta1_beta1"))
# 
# 
# GetInfluenceDataFrame <- function(ind, param_name, val) {
#   data.frame(draw=1:nrow(u_draws), beta1=u_draws[, 1], beta2=u_draws[, 2],
#              influence=influence_matrix[, ind], param_name=param_name, val=val)
# }
# 
# influence_df <- data.frame()
# for (n in 1:nrow(inds)) {
#   influence_df <- rbind(influence_df, GetInfluenceDataFrame(inds[n, "ind"], inds[n, "param_name"], inds[n, "val"]))
# }
# 
# influence_cast <-
#   melt(influence_df, id.vars=c("draw", "beta1", "beta2", "param_name")) %>%
#   dcast(draw + beta1 + beta2 ~ param_name + variable) %>%
#   mutate(var_beta1_influence = E_beta1_beta1_influence - 2 * E_beta1_val * E_beta1_influence)
# 
# 

######################################
# New influence functions

mp_draw <- mp_opt
log_q_grad <- rep(0, vp_indices$encoded_size)
GetLogVariationalDensity <- function(u) {
  beta_q_derivs <- GetBetaLogDensity(u, vp_opt, mp_draw, pp, TRUE, TRUE)
  log_q_grad[global_mask] <- beta_q_derivs$grad
  list(val=beta_q_derivs$val, grad=log_q_grad)
}

GetBetaLogDensity <- function(beta, vp_opt, draw, pp, unconstrained, calculate_gradient) {
  draw$beta_e_vec <- beta
  draw$beta_e2_vec <- beta %*% t(beta)
  opt$calculate_gradient <- calculate_gradient
  opt$calculate_hessian <- FALSE
  q_derivs <- GetLogVariationalDensityDerivatives(draw, vp_opt, opt, global_only=TRUE,
                                                  include_beta=TRUE, include_mu=FALSE, include_tau=FALSE)
  return(q_derivs)
}


# Get a function that converts a a draw from mu_k and a standard mvn into a draw from (mu_c | mu_k)
# k is the conditioned component, c is the "complement", i.e. the rest
GetConditionalMVNFunction <- function(k_ind, mvn_mean, mvn_info) {
  mvn_sigma <- solve(mvn_info)
  c_ind <- setdiff(1:length(mvn_mean), k_ind)
  
  # The scale in front of the mu_k for the mean of (mu_c | mu_k)
  # mu_cc_sigma <- mu_sigma[c_ind, c_ind, drop=FALSE]
  mu_kk_sigma <- mvn_sigma[k_ind, k_ind, drop=FALSE]
  mu_ck_sigma <- mvn_sigma[c_ind, k_ind, drop=FALSE]
  sig_cc_corr <- mu_ck_sigma %*% solve(mu_kk_sigma)
  
  # What to multiply by to get Cov(mu_c | mu_k)
  mu_c_cov <- solve(mvn_info[c_ind, c_ind])
  mu_c_scale <- t(chol(mu_c_cov))
  
  # Given u and a draws mu_c_std ~ Standard normal, convert mu_c_std to a draw from MVN( . | mu_k).
  # If there are multiple mu_c_std, each draw should be in its own column.
  GetConditionalDraw <- function(mu_k, mu_c_std) {
    mu_c_mean <- mvn_mean[c_ind] + sig_cc_corr %*% (mu_k - mvn_mean[k_ind, drop=FALSE])
    mu_c_scale %*% mu_c_std + matrix(rep(mu_c_mean, ncol(mu_c_std)), ncol=ncol(mu_c_std))
  }
}


# Beta draws:
GetBetaImportanceFunctions <- function(beta_comp, vp_opt, pp, lrvb_results) {
  mp_opt <- GetMomentParametersFromNaturalParameters(vp_opt)

  beta_cov <- solve(vp_opt$beta_info)
  u_mean <- vp_opt$beta_loc[beta_comp]
  # Increase the variance for sampling.  How much is enough?
  u_cov <- (1.5 ^ 2) * beta_cov[beta_comp, beta_comp]
  GetULogDensity <- function(u) {
    dnorm(u, mean=u_mean, sd=sqrt(u_cov), log=TRUE)
  }
  
  DrawU <- function(n_samples) {
    rnorm(n_samples, mean=u_mean, sd=sqrt(u_cov))
  }
  
  prior_cov <- solve(pp$beta_info)
  GetLogPrior <- function(u) {
    dnorm(u, mean=pp$beta_loc[beta_comp], sd=sqrt(prior_cov[beta_comp, beta_comp]), log=TRUE)
  }
  
  GetLogPriorVec <- function(u_vec) {
    GetLogPrior(u_vec)
  }
  
  # This is the marginal density of the beta_comp component.
  GetLogVariationalDensity <- function(u) {
    return(dnorm(u, mean=vp_opt$beta_loc[beta_comp], sd=sqrt(beta_cov[beta_comp, beta_comp])))
  }

  # This is the density and derivatives of the full beta density.  
  mp_draw <- mp_opt
  log_q_grad <- rep(0, vp_indices$encoded_size)
  GetFullBetaLogVariationalDensity <- function(u) {
    beta_q_derivs <- GetBetaLogDensity(u, vp_opt, mp_draw, pp, TRUE, TRUE)
    log_q_grad[global_mask] <- beta_q_derivs$grad
    list(val=beta_q_derivs$val, grad=log_q_grad)
  }
  
  GetFullLogQGradTerm <- function(beta) {
    beta_log_q_derivs <- GetFullBetaLogVariationalDensity(beta)
    return(as.numeric(beta_log_q_derivs$grad))
  }

  lrvb_pre_factor <- -1 * lrvb_results$jac %*% solve(lrvb_results$elbo_hess)
  DrawConditionalBeta <- GetConditionalMVNFunction(beta_comp, vp_opt$beta_loc, vp_opt$beta_info)
  GetLogQGradTerms <- function(u_draws, num_mc_draws) {
    # The dimensions of beta_u_draws are (component, u draw, mc draw)
    beta_u_draws <- array(NaN, dim=c(vp_opt$k_reg, length(u_draws), num_mc_draws))
    c_ind <- setdiff(1:vp_opt$k_reg, beta_comp)
    beta_std_draws <- rmvnorm(num_mc_draws, mean=rep(0, vp_opt$k - 1))
    
    # Draws from the rest of beta (beta "complement") given u_draws.
    beta_cond_draws <- sapply(u_draws, function(u) DrawConditionalBeta(u, t(beta_std_draws)))
    beta_u_draws[c_ind,, ] <- t(beta_cond_draws)
    beta_u_draws[beta_comp,, ] <- rep(u_draws, num_mc_draws)

    # The dimensions of lrvb_term_draws work out to be c(moment index, u draw, conditional beta draw)
    lrvb_term_draws <- apply(beta_u_draws, MARGIN=c(2, 3), FUN=GetFullLogQGradTerm)
    lrvb_term_e <- apply(lrvb_term_draws, MARGIN=c(1, 2), FUN=mean)
    lrvb_terms <- lrvb_pre_factor %*% lrvb_term_e
    
    if (FALSE) {
      u_draws_flat <- rep(u_draws, num_mc_draws)
      ind <- mp_indices$beta_e_vec[beta_comp]
      ggplot() +
        geom_point(aes(x=u_draws_flat, y=as.numeric(lrvb_term_draws[ind, , ]), color="draw")) +
        geom_line(aes(x=u_draws, y=lrvb_term_e[ind, ], color="e"))
      ggplot() +
        geom_line(aes(x=u_draws, y=lrvb_terms[ind, ], color="e")) +
        geom_line(aes(x=u_draws, y=u_draws - mean(u_draws), color="u"))
    }

    return(lrvb_terms)
  }
  
  return(list(GetULogDensity=GetULogDensity,
              DrawU=DrawU,
              GetLogPrior=GetLogPrior,
              GetLogPriorVec=GetLogPriorVec,
              GetLogVariationalDensity=GetLogVariationalDensity,
              GetLogQGradTerms=GetLogQGradTerms))
}


beta_funs <- GetBetaImportanceFunctions(1, vp_opt, pp, lrvb_results)

num_draws <- 500
num_mc_draws <- 20
DrawImportanceSamples <- beta_funs$DrawU
GetImportanceLogProb <- beta_funs$GetULogDensity
GetLogQGradTerms <- beta_funs$GetLogQGradTerms
GetLogQ <- beta_funs$GetLogVariationalDensity
GetLogPrior <- beta_funs$GetLogPrior

stop()
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

  influence_fun  <-
    do.call(rbind, lapply(u_draws, function(u) { GetLogQGradTerm(u) })) * exp(influence_lp_ratio)
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
    worst_case=worst_case))
}





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


