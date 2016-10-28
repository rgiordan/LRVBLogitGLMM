library(LogitGLMMLRVB)
library(rstan)
library(lme4)

library(dplyr)
library(reshape2)
library(trust)

library(LRVBUtils)


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

#################################
# Sensitivity analysis

comb_indices <- GetPriorsAndNaturalParametersFromVector(
  vp_opt, pp, as.numeric(1:(vp_opt$encoded_size + pp$encoded_size)), FALSE)
comb_prior_ind <- GetPriorParametersVector(comb_indices$pp, FALSE)
comb_vp_ind <- GetNaturalParameterVector(comb_indices$vp, FALSE)

opt$calculate_hessian <- TRUE
log_prior_derivs <- GetFullModelLogPriorDerivatives(vp_opt, pp, opt)
log_prior_param_prior <- Matrix(log_prior_derivs$hess[comb_vp_ind, comb_prior_ind])

prior_sens <- -1 * lrvb_results$jac %*% Matrix::solve(lrvb_results$elbo_hess, log_prior_param_prior)


##################
# Influence functions

mp_draws <- vb_results$mp_draws
obs <- mp_draws[[1]]
log_q_derivs <- GetLogVariationalDensityDerivatives(mp_draws[[1]], vp_opt, opt)


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
