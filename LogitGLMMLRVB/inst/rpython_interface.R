
# In order to replace some of read_vb_python.R, try using rPython to convert between parameters
# and vectors using the python machinery.

# https://stackoverflow.com/questions/45319387/using-rpython-import-numply-with-python-3-5
# https://github.com/rstudio/reticulate
library(reticulate)
library(purrr)
library(dplyr)

py_main <- py_run_string(
"
import sys
import os
print(sys.version)
sys.path.append('/home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.py')
sys.path.append('/home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.py/Models')
sys.path.append('/home/rgiordan/Documents/git_repos/autograd')

import VariationalBayes as vb
import LogisticGLMM_lib as logit_glmm
")

glmm_par <- py_main$logit_glmm$get_glmm_parameters(K=5, NG=10)

rand_par <- array(runif(glmm_par$free_size()))
glmm_par$set_free(rand_par)
glmm_par

glmm_indices <- py_main$logit_glmm$get_glmm_parameters(K=5, NG=10)
glmm_indices$set_vector(array(seq(1, glmm_indices$vector_size())))

glmm_par_list <- glmm_par$dictval()
glmm_par$param_dict$mu$mean$get()


moment_par <- py_main$logit_glmm$MomentWrapper(glmm_par)
moment_par$get_moment_vector(glmm_par$get_free())
moment_par
glmm_par

moment_list <- moment_par$moment_par$dictval()


########################
# Convert an MCMC draw to a moment vector

project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "LRVBLogitGLMM")
source(file.path(project_directory, "LogitGLMMLRVB/inst/optimize_lib.R"))

# analysis_name <- "simulated_data_small"
# analysis_name <- "simulated_data_large"
analysis_name <- "criteo_subsampled"

data_directory <- file.path(project_directory, "LogitGLMMLRVB/inst/data/")
stan_draws_file <- file.path(data_directory, paste(analysis_name, "_mcmc_draws.Rdata", sep=""))
stan_results <- LoadIntoEnvironment(stan_draws_file)

mcmc_sample <- extract(stan_results$stan_sim)

# mp_draws <- PackMCMCSamplesIntoMoments(mcmc_sample, mp_opt)
# draws_mat <- do.call(rbind, lapply(mp_draws, function(draw) GetMomentParameterVector(draw, FALSE)))
# log_prior_grad_list <- GetMCMCLogPriorDerivatives(mp_draws, pp, opt)
# log_prior_grad_mat <- do.call(rbind, log_prior_grad_list)


glmm_par <- py_main$logit_glmm$get_glmm_parameters(
  K=stan_results$stan_dat$K, NG=stan_results$stan_dat$NG)


#####################
# Convert python object dictvals to tidy data frames.

RecursiveUnpackParameter <- function(par, level=0, id_name="par") {
  if (is.numeric(par)) {
    if (length(par) == 1) {
      return(tibble(val=par))
    } else {
      return(tibble(val=par, component=1:length(par)))
    }
  } else if (is.list(par)) {
    next_level <- map(par, RecursiveUnpackParameter, level + 1)
    return(bind_rows(next_level, .id=paste(id_name, level + 1, sep="_")))
  }
}

foo <- list(a=runif(3), b=runif(4))
baz <- list(c=runif(2), d=foo)

RecursiveUnpackParameter(foo)
RecursiveUnpackParameter(baz)

glmm_par_df <-
  RecursiveUnpackParameter(glmm_par_list) %>%
  rename(par=par_1)
#View(glmm_par_df)

moment_df <-
  RecursiveUnpackParameter(moment_list) %>%
  rename(par=par_1, component1=par_2)



