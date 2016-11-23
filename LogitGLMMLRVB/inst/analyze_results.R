library(LogitGLMMLRVB)
library(rstan)
library(lme4)

library(dplyr)
library(reshape2)
library(trust)

library(LRVBUtils)

library(gridExtra)

project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "LRVBLogitGLMM")
# source(file.path(project_directory, "LogitGLMMLRVB/inst/optimize_lib.R"))

analysis_name <- "simulated_data"

data_directory <- file.path(project_directory, "LogitGLMMLRVB/inst/data/")
vb_results_file <- file.path(data_directory, paste(analysis_name, "_vb_results.Rdata", sep=""))

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
# Sensitivity analysis

comb_indices <- GetPriorsAndNaturalParametersFromVector(
  vp_opt, pp, as.numeric(1:(vp_opt$encoded_size + pp$encoded_size)), FALSE)
comb_prior_ind <- GetPriorParametersVector(comb_indices$pp, FALSE)
comb_vp_ind <- GetNaturalParameterVector(comb_indices$vp, FALSE)

opt$calculate_hessian <- TRUE
log_prior_derivs <- GetFullModelLogPriorDerivatives(vp_opt, pp, opt)
log_prior_param_prior <- Matrix(log_prior_derivs$hess[comb_vp_ind, comb_prior_ind])

prior_sens <- -1 * lrvb_results$jac %*% Matrix::solve(lrvb_results$elbo_hess, log_prior_param_prior)#


##################
# Influence functions

library(mvtnorm)

# Monte Carlo samples
n_samples <- 50000

# Define functions necessary to compute influence function stuff

# Just for testing
draw <- mp_opt
beta <- c(1.2, 2.0)

GetBetaLogPrior <- function(beta, pp) {
  # You can't use the VB priors because they are
  # (1) a function of the natural parameters whose variance would have to be zero and
  # (2) not normalized.
  dmvnorm(beta, mean=pp$beta_loc, sigma=solve(pp$beta_info), log=TRUE)
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


# You could also do this more numerically stably with a Cholesky decomposition.
lrvb_pre_factor <- -1 * lrvb_results$jac %*% solve(lrvb_results$elbo_hess)

# Proposals based on q
u_mean <- mp_opt$beta_e_vec
# Increase the covariance for sampling.  How much is enough?
u_cov <- (1.5 ^ 2) * solve(vp_opt$beta_info)
GetULogDensity <- function(beta) {
  dmvnorm(beta, mean=u_mean, sigma=u_cov, log=TRUE)
}


DrawU <- function(n_samples) {
  rmvnorm(n_samples, mean=u_mean, sigma=u_cov)
}
u_draws <- DrawU(n_samples)


GetLogPrior <- function(u) {
  GetBetaLogPrior(u, pp)
}

# GetLogContaminatingPrior <- function(u) {
#   GetMuLogStudentTPrior(u, pp_perturb)
# }

mp_draw <- mp_opt
log_q_grad <- rep(0, vp_indices$encoded_size)
GetLogVariationalDensity <- function(u) {
  beta_q_derivs <- GetBetaLogDensity(u, vp_opt, mp_draw, pp, TRUE, TRUE)
  log_q_grad[global_mask] <- beta_q_derivs$grad
  list(val=beta_q_derivs$val, grad=log_q_grad)
}

GetLogVariationalDensity(beta)


GetInfluenceFunctionSample <- GetInfluenceFunctionSampleFunction(
  GetLogVariationalDensity, GetLogPrior, GetULogDensity, lrvb_pre_factor)

GetInfluenceFunctionSample(u_draws[1, ])

influence_list <- list()
pb <- txtProgressBar(min=1, max=nrow(u_draws), style=3)
for (ind in 1:nrow(u_draws)) {
  setTxtProgressBar(pb, ind)
  influence_list[[ind]] <- GetInfluenceFunctionSample(u_draws[ind, ])
}
close(pb)


names(influence_list[[1]])
influence_vector_list <- lapply(influence_list, function(x) as.numeric(x$influence_function))
influence_matrix <- do.call(rbind, influence_vector_list)

mp_opt_vector <- GetMomentParameterVector(mp_opt, FALSE)
GetIndexRow <- function(ind, param_name) {
  data.frame(ind=ind, param_name=param_name, val=mp_opt_vector[ind])
}
inds  <- data.frame()
inds <- rbind(inds, GetIndexRow(mp_indices$beta_e_vec[1], param_name <- "E_beta1"))
inds <- rbind(inds, GetIndexRow(mp_indices$beta_e_vec[2], param_name <- "E_beta2"))
inds <- rbind(inds, GetIndexRow(mp_indices$beta_e_outer[1, 1], param_name <- "E_beta1_beta1"))


GetInfluenceDataFrame <- function(ind, param_name, val) {
  data.frame(draw=1:nrow(u_draws), beta1=u_draws[, 1], beta2=u_draws[, 2],
             influence=influence_matrix[, ind], param_name=param_name, val=val)
}

influence_df <- data.frame()
for (n in 1:nrow(inds)) {
  influence_df <- rbind(influence_df, GetInfluenceDataFrame(inds[n, "ind"], inds[n, "param_name"], inds[n, "val"]))
}

influence_cast <-
  melt(influence_df, id.vars=c("draw", "beta1", "beta2", "param_name")) %>%
  dcast(draw + beta1 + beta2 ~ param_name + variable) %>%
  mutate(var_beta1_influence = E_beta1_beta1_influence - 2 * E_beta1_val * E_beta1_influence)

if (FALSE) {
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

if (FALSE) {
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
}
