library(LogitGLMMLRVB)
library(rstan)
library(lme4)

library(dplyr)
library(reshape2)
library(trust)

library(gridExtra)

project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "LRVBLogitGLMM")
source(file.path(project_directory, "LogitGLMMLRVB/inst/optimize_lib.R"))

project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "LRVBLogitGLMM")

analysis_name <- "simulated_data"

data_directory <- file.path(project_directory, "LogitGLMMLRVB/inst/data/")
stan_draws_file <- file.path(data_directory, paste(analysis_name, "_mcmc_draws.Rdata", sep=""))

stan_results <- LoadIntoEnvironment(stan_draws_file)


SummarizeResults <- function(stan_sim, method) {
  mcmc_sample <- extract(stan_sim)
  res <- list()
  res[[length(res) + 1]] <- SummarizeMCMCColumn(mcmc_sample$mu, "mu")
  res[[length(res) + 1]] <- SummarizeMCMCColumn(mcmc_sample$tau, "tau")
  num_k <- dim(mcmc_sample$beta)[2]
  for (k in 1:num_k) {
    res[[length(res) + 1]] <- SummarizeMCMCColumn(mcmc_sample$beta[, k], "beta", component=k)
  }
  df <- do.call(rbind, res)
  df$method <- method
  
  return(df)
}


result <- rbind(
  SummarizeResults(stan_results$stan_sim, "normal") %>% mutate(epsilon=0),
  SummarizeResults(stan_results$stan_sim_eps0_1, "eps0_1") %>% mutate(epsilon=0.1),
  SummarizeResults(stan_results$stan_sim_eps1, "eps1") %>% mutate(epsilon=1)
)


dcast(result, par + component ~ metric + method, value.var="val")

stan_sim <- extract(stan_results$stan_sim)
plot(stan_sim$mu_normal_lpdf, stan_sim$mu_normal_c_lpdf)

weights <- exp(stan_sim$mu_normal_c_lpdf - stan_sim$mu_normal_lpdf)
weights <- length(weights) * weights / sum(weights)
hist(weights, 100)

ggplot(filter(result, metric=="mean", par=="mu")) +
  geom_point(aes(x=epsilon, y=val, color=par))
