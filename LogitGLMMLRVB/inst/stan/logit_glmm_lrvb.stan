data {
  // Data
  int <lower=0> NG;  // number of groups
  int <lower=0> N;  // total number of observations
  int <lower=0> K;  // dimensionality of parameter vector which is jointly distributed
  int <lower=0, upper=1> y[N];       // outcome variable of interest
  vector[K] x[N];       // Covariates
  
  // y_group is zero-indexed group indicators
  int y_group[N];
  
  // Prior parameters
  matrix[K,K] beta_prior_var; // Should be beta_prior_var
  vector[K] beta_prior_mean;
  real mu_prior_mean;
  real <lower=0> mu_prior_var;
  real <lower=0> tau_prior_alpha;
  real <lower=0> tau_prior_beta;

  // Standard normal points for MC evaluation of expectations
  int <lower=0> N_sim;
  vector[N_sim] std_normal_draws;
}

parameters {
  // Global regressors.
  vector[K] e_beta;
  cov_matrix[K] cov_beta;
  
  // The mean of the random effect.
  real e_mu;
  real<lower=0.00001> var_mu;

  // The information of the random effect.
  real<lower=0> tau_shape;
  real<lower=0> tau_rate;

  // The actual random effects.
  vector[NG] e_u;
  vector<lower=0>[NG] var_u;
}

model {
  // Some intermediate values
  matrix[K, K] e_beta_outer;
  real e_mu2;
  real e_tau;
  real e_log_tau;
  matrix[K, K] beta_prior_info;

  vector[N] e_log_1mp;
  vector[N] logit_p_mean;
  vector[N] logit_p_sd;
  
  e_beta_outer = e_beta * e_beta' + cov_beta;
  e_mu2 = e_mu^2 + var_mu;
  e_tau = tau_shape / tau_rate;
  e_log_tau = digamma(tau_shape) - log(tau_rate);
  
  // Expected probabilities and functions thereof.
  for (n in 1:N) {
    // y_group is 0-indexed
    logit_p_mean[n] = dot_product(x[n], e_beta) + e_u[y_group[n] + 1];
    logit_p_sd[n] = sqrt(
      dot_product(x[n], e_beta_outer * x[n]) + var_u[y_group[n] + 1]);
  
    // log1m_inv_logit doesn't seem to apply to vectors.
    e_log_1mp[n] = 0;
    for (n_sim in 1:N_sim) {
      e_log_1mp[n] = e_log_1mp[n] +
        log1m_inv_logit(logit_p_sd[n] * std_normal_draws[n_sim] + logit_p_mean[n]);
    }
    e_log_1mp[n] = e_log_1mp[n] / N_sim;
  }  

  //////////////////////
  // Priors

  // mu ~ normal_lpdf(mu_prior_mean, mu_prior_var);
  // tau ~ gamma(tau_prior_alpha, tau_prior_beta);
  // beta ~ multi_normal(beta_prior_mean, beta_prior_var);
  target += -0.5 * (e_mu2 - 2 * e_mu * mu_prior_mean + mu_prior_mean^2) / mu_prior_var;
  target += (tau_prior_alpha - 1) * e_log_tau - tau_prior_beta * e_tau;
  // When taking gradients of quadratic forms they must be symmetric
  target += -0.5 * (
    trace(mdivide_left_spd(beta_prior_var, e_beta_outer)) -
    dot_product(e_beta, mdivide_left_spd(beta_prior_var, beta_prior_mean)) -
    dot_product(beta_prior_mean, mdivide_left_spd(beta_prior_var, e_beta)));

  //////////////////////
  // The model

  for (g in 1:NG) {
    // u[g] ~ normal(mu, 1 / tau);
    target +=
      -0.5 * e_tau * (e_mu2 - 2 * e_mu * e_u[g] + (e_u[g]^2 + var_u[g]))
      - 0.5 * e_log_tau;
  }
  
  for (n in 1:N) {
    // y[n] ~ bernoulli(p[n]);
    // y_group is zero-indexed, but stan is one-indexed
    // y[n] ~ bernoulli(inv_logit(x[n]' * beta + u[y_group[n] + 1]));
    target += y[n] * logit_p_mean[n] + e_log_1mp[n];
  }
  
  
  // The entropy
  // beta
  target += 0.5 * log(determinant(cov_beta));  
  
  // mu
  target += 0.5 * log(var_mu);
  
  // tau
  target += tau_shape - log(tau_rate) + lgamma(tau_shape) +
            (1 - tau_shape) * digamma(tau_shape);
  
  // u
  // for (g in 1:NG) {
  //   target += 0.5 * log(var_u);
  // }
}
