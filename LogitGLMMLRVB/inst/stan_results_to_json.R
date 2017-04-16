# Export data from the stan results into JSON so python can read it.
library(jsonlite)

project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "LRVBLogitGLMM")
source(file.path(project_directory, "LogitGLMMLRVB/inst/optimize_lib.R"))

# analysis_name <- "simulated_data_small"
analysis_name <- "simulated_data_large"

data_directory <- file.path(project_directory, "LogitGLMMLRVB/inst/data/")
stan_draws_file <- file.path(data_directory, paste(analysis_name, "_mcmc_draws.Rdata", sep=""))

stan_results <- LoadIntoEnvironment(stan_draws_file)

stan_results$pp
stan_results$stan_dat
stan_dat_json <- toJSON(stan_results$stan_dat)

prob <- SampleData(n_obs, k_reg, n_groups)
vp_base <- prob$vp_nat
vp_base_json <- toJSON(vp_base)

vp_base$u_info_min <- 1e-3
vp_base$beta_diag_min <- 1e-3
vp_base$tau_alpha_min <- 1e-6
vp_base$tau_beta_min <- 1e-6

# Write the ADVI solution to a json format.
mcmc_sample <- extract(stan_results$stan_advi)
advi_results <- list()
advi_results$mu_mean <- mean(mcmc_sample$mu)
advi_results$mu_var <- var(mcmc_sample$mu)
advi_results$beta_mean <- colMeans((mcmc_sample$beta))
advi_results$beta_info <- solve(cov((mcmc_sample$beta)))
advi_results$u_mean <- apply(mcmc_sample$u, MARGIN=2, mean)
advi_results$u_var <- apply(mcmc_sample$u, MARGIN=2, var)
advi_results$tau_mean <- mean(mcmc_sample$tau)
advi_results$tau_var <- var(mcmc_sample$tau)

json_filename <- file.path(data_directory, paste(analysis_name, "_stan_dat.json", sep=""))
json_file <- file(json_filename, "w")
json_list <- toJSON(list(stan_dat=stan_results$stan_dat, vp_base=vp_base, advi_results=advi_results))
write(json_list, file=json_file)
close(json_file)



