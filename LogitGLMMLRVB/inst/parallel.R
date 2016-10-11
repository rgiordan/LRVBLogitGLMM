library(LogitGLMMLRVB)
library(rstan)
library(lme4)

library(dplyr)
library(reshape2)


project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "variational_bayes/logit_glmm")

analysis_name <- "simulated_data"

data_directory <- file.path(project_directory, "LogitGLMMLRVB/inst/data/")
stan_draws_file <- file.path(data_directory, paste(analysis_name, "_mcmc_draws.Rdata", sep=""))

stan_results <- environment()
load(stan_draws_file, envir=stan_results)
mcmc_sample <- extract(stan_results$stan_sim)

true_params <- stan_results$true_params
x <- stan_results$stan_dat$x
k_reg <- ncol(x)
n_obs <- nrow(x)
y <- stan_results$stan_dat$y
y_g <- stan_results$stan_dat$y_g
n_groups <- max(y_g) + 1

# Get C++ versions of the data structures.
prob <- SampleData(n_obs, k_reg, n_groups)

pp <- stan_results$pp

vp_nat <- prob$vp_nat

vp_nat$beta_loc <- rep(0, k_reg)
vp_nat$mu_loc <- 0
vp_nat$tau_alpha <- 2
vp_nat$tau_beta <- 2
for (g in 1:n_groups) {
  vp_nat$u_vec[[g]]$u_loc <- 0
  vp_nat$u_vec[[g]]$u_info <- 1
}


##############################
# BFGS

opt <- GetOptions()
vp_nat_bfgs <- FitModel(y, y_g, x, vp_nat, pp, opt, reltol=1e-16)

################################
# LRVB

mfvb_cov <- GetCovariance(vp_nat_bfgs)
lrvb_cov <- GetLRVBCovariance(y, y_g, x, vp_nat_bfgs, pp, opt)
stopifnot(min(diag(lrvb_cov)) > 0)
stopifnot(min(diag(mfvb_cov)) > 0)

vp_mom <- GetMomentParametersFromNaturalParameters(vp_nat_bfgs)
lrvb_sd <- GetMomentParametersFromVector(n_obs, k_reg, n_groups, sqrt(diag(lrvb_cov)), FALSE)
mfvb_sd <- GetMomentParametersFromVector(n_obs, k_reg, n_groups, sqrt(diag(mfvb_cov)), FALSE)

#############################
# Unpack the results.

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
      mutate(is_u = par == "u")
  ) +
    geom_point(aes(x=mcmc, y=mfvb, color="mfvb"), size=3) +
    geom_point(aes(x=mcmc, y=lrvb, color="lrvb"), size=3) +
    geom_abline(aes(intercept=0, slope=1)) +
    facet_grid(~ is_u)
  
  ggplot(
    filter(results, metric == "sd", par != "u") %>%
      dcast(par + component + group ~ method, value.var="val")
  ) +
    geom_point(aes(x=mcmc, y=lrvb, color=par), size=3) +
    geom_abline(aes(intercept=0, slope=1))
}





##########################
# Sub-matrix LRVB

sub_mat <- GetLRVBSubMatrices(vp_nat_bfgs, elbo_hess, jac)
lrvb_v2 <- -1 * sub_mat$jac_t %*% solve(sub_mat$lrvb_inv_term, t(sub_mat$jac_t))

stopifnot(max(abs(lrvb_v2 - lrvb[global_ind, global_ind])) < 1e-12)





###############################################
# Fit in parallel

shards <- 3
shard_map <- data.frame(g=0:(n_groups - 1))
shard_map$shard <- shard_map$g %% shards + 1
shard_map <-
  group_by(shard_map, shard) %>%
  arrange(shard, g) %>%
  mutate(local_g=order(g) - 1)

print(shard_map, n=Inf)


shard_results <- list()

for (shard_id in 1:shards) {
  shard_rows <- y_g %in% filter(shard_map, shard == shard_id)$g
  
  local_shard_y_g <-
    data.frame(g=y_g[shard_rows]) %>%
    inner_join(select(shard_map, g, local_g), by="g")
  
  shard_n_obs <- sum(shard_rows)
  shard_n_groups <- max(local_shard_y_g$local_g) + 1
  
  shard_prob <- SampleData(shard_n_obs, k_reg, shard_n_groups)
  
  shard_vp_nat <- shard_prob$vp_nat
  shard_vp_nat$beta_loc <- rep(0, k_reg)
  shard_vp_nat$mu_loc <- 0
  shard_vp_nat$tau_alpha <- 2
  shard_vp_nat$tau_beta <- 2
  for (g in 1:shard_n_groups) {
    shard_vp_nat$u_vec[[g]]$u_loc <- 0
    shard_vp_nat$u_vec[[g]]$u_info <- 1
  }
  
  local_y <- y[shard_rows]
  local_x <- x[shard_rows, ]
  local_y_g <- as.integer(local_shard_y_g$local_g)
  shard_vp_nat <- FitModel(local_y, local_y_g, local_x, shard_vp_nat, pp)
  
  mfvb_cov <- GetCovariance(shard_vp_nat)
  elbo_hess <- GetSparseLogLikHessian(local_y, local_y_g, local_x, shard_vp_nat, pp, opt, TRUE) +
    GetSparseEntropyHessian(shard_vp_nat, opt)
  jac <- GetMomentJacobian(vp_nat, opt)$jacobian
  
  lrvb_sub_mats <- GetLRVBSubMatrices(shard_vp_nat, elbo_hess, jac)
  
  shard_results[[as.character(shard_id)]] <-
    list(vp_nat=shard_vp_nat, local_shard_y_g=local_shard_y_g,
         mfvb_cov=mfvb_cov, lrvb_sub_mats=lrvb_sub_mats)
}

AddGlobalParams <- function(vp_nat1, vp_nat2) {
  stopifnot(vp_nat1$k_reg == vp_nat2$k_reg)
  
  indices <- GetIndices(vp_nat1)
  theta1 <- GetNaturalParameterVector(vp_nat1, FALSE)
  theta2 <- GetNaturalParameterVector(vp_nat2, FALSE)
  theta <- theta1[indices$global_ind] + theta2[indices$global_ind]
  vp_nat_comb <- GetNaturalParametersFromVector(0, vp_nat1$k_reg, 0, theta, FALSE)
  return(vp_nat_comb)
}


GetEmptyGlobalParams <- function(base_vp_nat) {
  indices <- GetIndices(base_vp_nat)
  theta <- rep(0, length(indices$global_ind))
  return(GetNaturalParametersFromVector(0, base_vp_nat$k_reg, 0, theta, FALSE))
}

# Note: these are not actually the natural parameters, so you can't just add them!
vp_nat_comb <- GetEmptyGlobalParams(shard_results[["1"]]$vp_nat)
indices <- GetIndices(vp_nat_comb)
lrvb_inv_term <- Matrix(0, length(indices$global_ind), length(indices$global_ind))
for (shard_id in 1:shards) {
  # Hack: convert the normals to actual natural parameters.
  shard_vp_nat <- shard_results[[as.character(shard_id)]]$vp_nat
  shard_vp_nat$beta_loc <- shard_vp_nat$beta_info %*% shard_vp_nat$beta_loc
  shard_vp_nat$mu_loc <- shard_vp_nat$mu_info * shard_vp_nat$mu_loc
  vp_nat_comb <- AddGlobalParams(vp_nat_comb, shard_vp_nat)
  lrvb_inv_term <- lrvb_inv_term + shard_results[[as.character(shard_id)]]$lrvb_sub_mats$lrvb_inv_term
}
# Hack: convert the normals back from actual natural parameters.
vp_nat_comb$beta_loc <- solve(vp_nat_comb$beta_info, vp_nat_comb$beta_loc)
vp_nat_comb$mu_loc <- vp_nat_comb$mu_loc / vp_nat_comb$mu_info

vp_mom_comb <- GetMomentParametersFromNaturalParameters(vp_nat_comb)
jac <- GetMomentJacobian(vp_nat_comb, opt)$jacobian
jac_t <- jac[indices$global_ind, indices$global_ind]
lrvb_comb <- -1 * jac_t %*% solve(lrvb_inv_term, t(jac_t))

vp_mom_bfgs <- GetMomentParametersFromNaturalParameters(vp_nat_bfgs)
vp_mom_bfgs$u_vec <- list()

vp_mom_comb
vp_mom_bfgs

cbind(sqrt(diag(lrvb_comb)),
      sqrt(diag(lrvb)[global_ind]))




##############################
# Simple sanity checks


comp <- 2
result <- data.frame()
for (beta in seq(-50, 30, 1)) {
  cat(beta, "\n")
  vp_nat_local <- vp_nat
  vp_nat_local$beta_loc[comp] <- beta
  vp_mom_local <- GetMomentParametersFromNaturalParameters(vp_nat_local)
  val <- GetLogLikDerivatives(y, y_g, x, vp_mom_local,
                              GetOptions(calculate_gradient=FALSE))
  this_result <- data.frame(beta=beta, e_beta=vp_mom_local$beta_e_vec[comp], val=val$val)
  result <- rbind(result, this_result)
}


result <- data.frame()
for (mu_info in seq(0.1, 300, 10)) {
  cat(mu_info, "\n")
  vp_nat_local <- vp_nat
  vp_nat_local$mu_info <- mu_info
  val <- GetEntropyDerivatives(vp_nat_local,
                               GetOptions(calculate_gradient=FALSE))
  this_result <- data.frame(mu_info=mu_info, val=val$val)
  result <- rbind(result, this_result)
}


if(F) {
  plot(result$mu_info, result$val)
}


#####################
# GLMER

prob_data <- data.frame(y=y, y_g=as.character(y_g))
prob_data <- cbind(prob_data, data.frame(x))
glmer_result <- glmer(y ~ 1 + X1 + X2 + (1 | y_g), prob_data, family=binomial)
glmer_summary <- summary(glmer_result)

# Not bad:
glmer_summary
true_params$beta

# This is not actually very good:
if (F) {
  plot(ranef(glmer_result)[[1]][[1]], unlist(true_params$u)); abline(0, 1)
}




######################
# Timing

system.time(
  derivs <-GetModelDerivatives(y, y_g, x, vp_mom, pp, GetOptions(calculate_gradient=TRUE))
)

system.time(
  derivs <- GetModelDerivatives(y, y_g, x, vp_mom, pp, GetOptions(calculate_gradient=FALSE))
)

# unsurprisingly, times are linear in the number of simulated points.
n_sims <- c(1, 3, 5, 10, 50, 75, 100)
times <- c()
for (n_sim in n_sims) {
  deriv_time <- Sys.time()
  derivs <-
    GetModelDerivatives(y, y_g, x, vp_mom, pp,
                        GetOptions(calculate_gradient=FALSE, n_sim=n_sim))
  times <- c(times, Sys.time() - deriv_time)
}

if (F) {
  plot(n_sims, times)
}



##############################
# Trust region.  Actually surprisingly slow.

theta <- theta_start <- GetNaturalParameterVector(vp_nat, TRUE)
OptimVal(theta)
length(OptimGrad(theta))
dim(OptimHess(theta))

ObjFun <- function(theta) {
  return(list(value=OptimVal(theta), gradient=OptimGrad(theta), hessian=as.matrix(OptimHess(theta))))
}

library(trust)
trust_result <- trust(ObjFun, theta_start, 1, rmax=10000, minimize=FALSE)
vp_nat_trust <- GetNaturalParametersFromVector(n_obs, k_reg, n_groups, trust_result$argument, TRUE)



