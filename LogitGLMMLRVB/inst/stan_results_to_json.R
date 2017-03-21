# Export data from the stan results into JSON so python can read it.
library(jsonlite)

project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "LRVBLogitGLMM")
source(file.path(project_directory, "LogitGLMMLRVB/inst/optimize_lib.R"))

analysis_name <- "simulated_data_small"
# analysis_name <- "simulated_data_large"

data_directory <- file.path(project_directory, "LogitGLMMLRVB/inst/data/")
stan_draws_file <- file.path(data_directory, paste(analysis_name, "_mcmc_draws.Rdata", sep=""))

stan_results <- LoadIntoEnvironment(stan_draws_file)

stan_results$pp
stan_results$stan_dat
stan_dat_json <- toJSON(stan_results$stan_dat)

json_filename <- file.path(data_directory, paste(analysis_name, "_stan_dat.json", sep=""))
json_file <- file(json_filename, "w")
write(stan_dat_json, file=json_file)
close(json_file)



