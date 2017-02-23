# This didn't work out because loess is too slow, and I'm concerned about how to set the bandwidth correctly.

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
  # GetConditionalDraw <- function(mu_k, mu_c_std) {
  #   mu_c_mean <- mvn_mean[c_ind] + sig_cc_corr %*% (mu_k - mvn_mean[k_ind, drop=FALSE])
  #   mu_c_scale %*% mu_c_std + matrix(rep(mu_c_mean, ncol(mu_c_std)), ncol=ncol(mu_c_std))
  # }
  DrawConditionalMVN <- function(mu_k, num_draws) {
    mu_c_mean <- mvn_mean[c_ind] + sig_cc_corr %*% (mu_k - mvn_mean[k_ind, drop=FALSE])
    rmvnorm(num_draws, mean=mu_c_mean, sigma=mu_c_cov)
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
  
  lrvb_pre_factor <- -1 * lrvb_results$jac %*% solve(lrvb_results$elbo_hess)
  GetFullLogQGradTerm <- function(beta) {
    beta_log_q_derivs <- GetFullBetaLogVariationalDensity(beta)
    return(as.numeric(lrvb_pre_factor %*% beta_log_q_derivs$grad))
  }
  
  DrawConditionalBeta <- GetConditionalMVNFunction(beta_comp, vp_opt$beta_loc, vp_opt$beta_info)
  GetLogQGradTerms <- function(u_draws, num_mc_draws) {
    beta_u_draws <- array(NaN, dim=c(vp_opt$k_reg, length(u_draws), num_mc_draws))
    c_ind <- setdiff(1:vp_opt$k_reg, beta_comp)
    
    # Draws from the rest of beta (beta "complement") given u_draws.
    beta_u_draws[c_ind,, ] <- sapply(u_draws, function(u) DrawConditionalBeta(u, num_mc_draws))
    beta_u_draws[beta_comp,, ] <- u_draws
    
    print("getting log q terms")
    # The dimensions work out to be c(moment index, u draw, conditional beta draw)
    lrvb_term_draws <- apply(beta_u_draws, MARGIN=c(2, 3), FUN=GetFullLogQGradTerm)
    
    # Use loess to get the conditional expectation of row <ind> of the lrvb term.
    u_draws_flat <- rep(u_draws, num_mc_draws)
    GetLRVBTermIndex <- function(ind) {
      cat(".")
      lrvb_term_ind <- as.numeric(lrvb_term_draws[ind,,])
      cat(",")
      loess_fit <- loess(lrvb_term_ind ~ u_draws_flat) # This is too slow
      cat(":")
      predict(loess_fit, u_draws)
      cat("!")
      # ggplot() +
      #   geom_point(aes(x=u_draws_flat, y=lrvb_term_ind, color="draw")) +
      #   geom_line(aes(x=u_draws_flat, y=lrvb_term_e, color="e")) +
      #   geom_line(aes(x=u_draws_flat, y=u_draws_flat - mean(u_draws), color="identity"))
    }
    
    print("running loess")
    log_q_grad_by_u <- sapply(1:(dim(lrvb_term_draws)[1]), GetLRVBTermIndex)
    print("multiplying")
    lrvb_terms <- lrvb_pre_factor %*% t(log_q_grad_by_u)
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

  # Stopped working here --   GetLogQGradTerms is too slow
  # influence_fun  <-
  #   do.call(rbind, lapply(u_draws, function(u) { GetLogQGradTerm(u) })) * exp(influence_lp_ratio)
  # u_influence_mat <- (influence_fun ^ 2) * exp(importance_lp_ratio)
  # u_influence_mat_pos <- ((influence_fun > 0) * influence_fun ^ 2) * exp(importance_lp_ratio)
  # u_influence_mat_neg <- ((influence_fun < 0) * influence_fun ^ 2) * exp(importance_lp_ratio)
  # 
  # worst_case <-
  #   sapply(1:ncol(influence_fun),
  #          function(ind) { sqrt(max(mean(u_influence_mat_pos[, ind]),
  #                                   mean(u_influence_mat_neg[, ind]))) })
  
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

