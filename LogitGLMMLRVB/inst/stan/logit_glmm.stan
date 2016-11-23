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
  matrix[K,K] beta_prior_var;
  vector[K] beta_prior_mean;
  real mu_prior_mean;
  real <lower=0> mu_prior_var;
  real <lower=0> tau_prior_alpha;
  real <lower=0> tau_prior_beta;
  
  // An alternative prior for the mu prior distribution.
  real <lower=0, upper=1> mu_prior_epsilon;
  real <lower=0> mu_prior_t;
}

parameters {
  // Global regressors.
  vector[K] beta;
  
  // The mean of the random effect.
  real mu;

  // The information of the random effect.
  real <lower=0> tau;

  // The actual random effects.
  vector[NG] u;

}

transformed parameters {
  // Latent probabilities
  // vector[N] p;
  // vector[N] logit_p;
  // for (n in 1:N) {
  //   // y_group is zero-indexed, but stan is one-indexed
  //   logit_p[n] = x[n]' * beta + u[y_group[n] + 1];
  //   p[n] = inv_logit(logit_p[n]);
  // }
  real mu_normal_lpdf;
  real mu_student_t_lpdf;

  mu_normal_lpdf = normal_lpdf(mu | mu_prior_mean, mu_prior_var);
  mu_student_t_lpdf = student_t_lpdf(mu | mu_prior_t, mu_prior_mean, sqrt(mu_prior_var));
}

model {
  // priors
  tau ~ gamma(tau_prior_alpha, tau_prior_beta);
  beta ~ multi_normal(beta_prior_mean, beta_prior_var);

  // Express the mu prior as a mixture of a normal and t prior.
  if (mu_prior_epsilon == 0) {
    mu ~ normal(mu_prior_mean, mu_prior_var);
  } else if (mu_prior_epsilon == 1) {
    mu ~ student_t(mu_prior_t, mu_prior_mean, sqrt(mu_prior_var));
  } else {
    // It is a mixture.
    // Why doesn't this work?  See https://groups.google.com/forum/#!category-topic/stan-users/general/_gOPDicnDl0
    //target += log_sum_exp(log(1 - mu_prior_epsilon) + mu_normal_lpdf,
    //                      log(mu_prior_epsilon) + mu_student_t_lpdf);
    target += log_sum_exp(log(1 - mu_prior_epsilon) + normal_lpdf(mu | mu_prior_mean, mu_prior_var),
                          log(mu_prior_epsilon) + student_t_lpdf(mu | mu_prior_t, mu_prior_mean, sqrt(mu_prior_var)));
  }
  
  // The model
  for (g in 1:NG) {
    u[g] ~ normal(mu, 1 / tau);
  }

  for (n in 1:N) {
    // y[n] ~ bernoulli(p[n]);
    // y_group is zero-indexed, but stan is one-indexed
    y[n] ~ bernoulli(inv_logit(x[n]' * beta + u[y_group[n] + 1]));
  }
}
