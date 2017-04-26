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
analysis_name <- "simulated_data_large"

data_directory <- file.path(project_directory, "LogitGLMMLRVB/inst/data/")

# Read from R or python.  If reading from python, make sure to run read_vb_python.R first.
# vb_results_file <- file.path(data_directory, paste(analysis_name, "_vb_results.Rdata", sep=""))
vb_results_file <- file.path(data_directory, paste(analysis_name, "_vb_python_results.Rdata", sep=""))


# If true, save the results to a file readable by knitr.
save_results <- TRUE
results_file <- file.path(data_directory,
                          paste(analysis_name, "_influence.Rdata", sep="_"))

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


##################
# Influence functions

draws_mat <- t(vb_results$draws_mat)
lrvb_sd_scale <- sqrt(diag(vb_results$lrvb_results$lrvb_cov))
mcmc_sd_scale <- sqrt(diag(cov(t(draws_mat)))) 

# Calculating the influence functions are a little slow, so don't calculate all of them.
influence_k <- min(vp_opt$k_reg, 1)

# The number of draws to use when marginalizing the VB distribution.
num_mc_draws <- 10

# The number of draws when evaluating the vb integral over the influence function.
num_draws <- 50
num_bootstraps <- 10

worst_case_list <- list()
gc()
for (beta_comp in 1:influence_k) {
  cat("beta_comp ", beta_comp, "\n")
  for (sample in 1:num_bootstraps) {
    cat("\n\nSample ", sample, "\n")
    cat("variational...\n")
    timer <- Sys.time()
    beta_funs <- GetBetaImportanceFunctions(beta_comp, vp_opt, pp, lrvb_results)
    beta_influence_results <- GetVariationalInfluenceResults(
      num_draws = num_draws,
      DrawImportanceSamples = beta_funs$DrawU,
      GetImportanceLogProb = beta_funs$GetULogDensity,
      GetLogQGradTerms = function(u_draws) { beta_funs$GetLogQGradTerms(u_draws, num_mc_draws, normalize=TRUE) },
      GetLogQ = beta_funs$GetLogVariationalDensity,
      GetLogPrior = beta_funs$GetLogPrior)
    cat("time (seconds): ", Sys.time() - timer, "\n")
    
    # Get MCMC worst-case
    cat("mcmc...\n")
    subsample_rows <- sample.int(nrow(draws_mat), nrow(draws_mat), replace=TRUE)
    timer <- Sys.time()
    param_draws <- draws_mat[subsample_rows, beta_comp]
    draws_sample <- draws_mat[subsample_rows, ]
    mcmc_funs <- GetMCMCInfluenceFunctions(param_draws, beta_funs$GetLogPrior)
    GetMCMCWorstCaseColumn <- function(col) { mcmc_funs$GetMCMCWorstCase(draws_sample[, col]) }
    mcmc_worst_case <- sapply(1:ncol(draws_mat), GetMCMCWorstCaseColumn)
    cat("time (seconds): ", Sys.time() - timer, "\n")
    
    # Compare
    cat("summarizing...\n")
    worst_case_list[[length(worst_case_list) + 1]] <- rbind(
      SummarizeVBResults(GetMomentParametersFromVector(mp_opt, mcmc_worst_case, FALSE),
                         method="mcmc", metric=paste("beta", beta_comp, sep="")),
      SummarizeVBResults(GetMomentParametersFromVector(mp_opt, beta_influence_results$worst_case, FALSE),
                         method="lrvb", metric=paste("beta", beta_comp, sep=""))
    ) %>% mutate(sample=sample)
  }
}

# *******************
# TODO: this should be normalized by the standard deviations
# *******************
worst_case_df <-
  do.call(rbind, worst_case_list) %>%
  group_by(par, component, group, metric, method) %>%
  summarize(val_mean=mean(val), val_sd=sd(val)) %>%
  melt(id.vars=c("par", "component", "group", "metric", "method")) %>%
  mutate(summary=variable)

# Inspect
worst_case_cast <-
  dcast(worst_case_df, par + component + group + metric ~ method + summary, value.var="value") %>%
  mutate(component=ordered(component))

theta <- seq(0, 2 * pi, length.out=20)
ellipse_dfs <- list()
for (row in 1:nrow(worst_case_cast)) {
  sd_x <- worst_case_cast[row, "mcmc_val_sd"]
  sd_y <- worst_case_cast[row, "lrvb_val_sd"]
  loc_x <- worst_case_cast[row, "mcmc_val_mean"]
  loc_y <- worst_case_cast[row, "lrvb_val_mean"]
  ellipse_dfs[[length(ellipse_dfs) + 1]] <-
    data.frame(x=loc_x + 2 * sd_x * cos(theta), y=loc_y + 2 * sd_y * sin(theta), row=row)
}
worst_case_cast_sds <- do.call(rbind, ellipse_dfs) %>%
  inner_join(mutate(worst_case_cast, row=1:nrow(worst_case_cast)), by="row")

if (FALSE) {
  ggplot(filter(worst_case_cast_sds, par != "u", component != 1)) +
    geom_polygon(aes(x=x, y=y, group=row), alpha=0.1, color=NA) +
    geom_point(aes(x=mcmc_val_mean, y=lrvb_val_mean), size=2) +
    geom_abline(aes(slope=1, intercept=0)) +
    expand_limits(x=0, y=0) +
    xlab("MCMC") + ylab("VB") 
  
  ggplot(filter(worst_case_cast_sds, par == "u")) +
    geom_polygon(aes(x=x, y=y, group=row), alpha=0.1, color=NA) +
    geom_point(aes(x=mcmc_val_mean, y=lrvb_val_mean), size=2) +
    geom_abline(aes(slope=1, intercept=0)) +
    expand_limits(x=0, y=0) +
    xlab("MCMC") + ylab("VB") 
  
  ggplot(filter(worst_case_cast_sds, par == "beta", component == 1)) +
    geom_polygon(aes(x=x, y=y, group=row), alpha=0.1, color=NA) +
    geom_point(aes(x=mcmc_val_mean, y=lrvb_val_mean), size=2) +
    geom_abline(aes(slope=1, intercept=0)) +
    expand_limits(x=0, y=0) +
    xlab("MCMC") + ylab("VB") 
}


###################################
# Get graphs of influence functions for a single component.

beta_comp <- 1
beta_funs <- GetBetaImportanceFunctions(beta_comp, vp_opt, pp, lrvb_results)

Rprof(tmp <- tempfile())
beta_influence_results <- GetVariationalInfluenceResults(
  num_draws = num_draws,
  DrawImportanceSamples = beta_funs$DrawU,
  GetImportanceLogProb = beta_funs$GetULogDensity,
  GetLogQGradTerms = function(u_draws) { beta_funs$GetLogQGradTerms(u_draws, num_mc_draws, normalize=TRUE) },
  GetLogQ = beta_funs$GetLogVariationalDensity,
  GetLogPrior = beta_funs$GetLogPrior)
Rprof()
summaryRprof(tmp)


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

ind <- mp_indices$beta_e_vec[5]
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

# We probably need error bars in the loess.
if (FALSE) {
  ggplot() +
    geom_point(data=mcmc_influence_df, aes(x=u, y=g_draws - mean(g_draws), color=method), alpha=0.1) +
    geom_line(data=vb_influence_df, aes(x=u, y=gbar, color=method), lwd=2) +
    geom_line(data=mcmc_influence_df, aes(x=u, y=gbar, color=method), lwd=2)
  
  u_draws <- mcmc_funs$param_draws
  lug <- loess(g ~ u, data.frame(u=u_draws, g=g_draws), span = 2/3, degree = 1)
  e_g_given_u_pred <- predict(lug, newdata=u_draws, se=TRUE)
  e_g_given_u <- e_g_given_u_pred$fit - mean(g_draws)
  e_g_given_u_se <- e_g_given_u_pred$se.fit
  
  grid.arrange(
    ggplot() +
      # geom_point(data=mcmc_influence_df, aes(x=u, y=g_draws - mean(g_draws), color=method), alpha=0.1) +
      geom_line(data=vb_influence_df, aes(x=u, y=gbar, color=method), lwd=2) +
      geom_line(data=mcmc_influence_df, aes(x=u, y=gbar, color=method), lwd=2) +
      geom_line(aes(x=u_draws, y=e_g_given_u, color="loess"), lwd=2) +
      geom_line(aes(x=u_draws, y=e_g_given_u + 2 * e_g_given_u_se, color="loess_se")) +
      geom_line(aes(x=u_draws, y=e_g_given_u - 2 * e_g_given_u_se, color="loess_se"))
    ,
    ggplot() +
      geom_line(data=mcmc_influence_df, aes(x=u, y=exp(log_posterior), color=method), lwd=2)
    , ncol=1
  )

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

################
# Save the results for the paper.

if (save_results) {
  save(worst_case_cast_sds, vb_influence_df, mcmc_influence_df,
       file=results_file)
}
