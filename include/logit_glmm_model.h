# ifndef LOGIT_GLMM_MODEL_H
# define LOGIT_GLMM_MODEL_H

# define INSTANTIATE_LOGIT_GLMM_MODEL_H 1

# include <Eigen/Dense>
# include <vector>
# include <boost/math/tools/promotion.hpp>

# include <string>

# include <stan/math.hpp>
// # include <stan/math/mix/mat/functor/hessian.hpp>
// # include <stan/math/fwd/scal.hpp>

# include "monte_carlo_parameters.h"
# include "variational_parameters.h"
# include "exponential_families.h"
# include "logit_glmm_model_parameters.h"
# include "logit_glmm_model.h"
# include "eigen_includes.h"


using std::vector;
using typename boost::math::tools::promote_args;

template <typename T> T ELogOneMinusP(VectorXT<T> draws) {
  T e_log_1mp = 0;
  for (int s = 0; s < draws.size(); s++) {
    e_log_1mp += stan::math::log1m_inv_logit(draws(s));
  }
  return e_log_1mp  / draws.size();

}


////////////////////////////////////////
// Likelihood functions

template <typename T> T
GetObservationLogLikelihood(VariationalMomentParameters<T> const &vp,
                            Data const &data,
                            MonteCarloNormalParameter const &logit_p_mc,
                            int n) {

    if (n < 0 || n > data.n_obs) {
        throw std::runtime_error("n out of bounds");
    }
    if (vp.k_reg != data.k_reg) {
        throw std::runtime_error("k_reg does not match");
    }

    // TODO: Maybe cache this calculation.
    MatrixXT<T> beta_var =
        vp.beta.e_outer.mat - vp.beta.e_vec * vp.beta.e_vec.transpose();
    int g = data.y_g(n);
    if (g >= vp.n_groups || g < 0) {
        throw std::runtime_error("g out of bounds");
    }
    T u_var = (vp.u[g].e2 - pow(vp.u[g].e, 2));
    VectorXT<T> x_row = data.x.row(n).template cast<T>();
    MatrixXT<T> x_outer = x_row * x_row.transpose();
    T logit_p_mean = x_row.dot(vp.beta.e_vec) + vp.u[g].e;
    T logit_p_var = (x_outer * beta_var).trace() + u_var;

    if (logit_p_var < 0) {
        std::ostringstream err_msg;
        err_msg << "Negative p variance: " << logit_p_var <<
            " u_var: " << u_var <<
            " beta variance: " << (x_outer * beta_var).trace() <<
            " observation " << n << " group " << g << "\n";
        throw std::runtime_error(err_msg.str());
    }
    VectorXT<T> draws = logit_p_mc.Evaluate(logit_p_mean, logit_p_var);
    return data.y(n) * logit_p_mean + ELogOneMinusP(draws);
}


template <typename T> T
GetGroupLogLikelihood(VariationalMomentParameters<T> const &vp,
                      Data const &data,
                      MonteCarloNormalParameter const &logit_p_mc,
                      int g) {

    if (vp.k_reg != data.k_reg) {
        throw std::runtime_error("k_reg does not match");
    }
    if (g + 1 > vp.n_groups || g < 0) {
        throw std::runtime_error("g out of bounds");
    }

    T log_lik = 0;
    for (int n = 0; n < data.n_obs; n++ ) {
        if (g == data.y_g(n)) {
            log_lik += GetObservationLogLikelihood(vp, data, logit_p_mc, n);
        }
    }

    log_lik += vp.u[g].ExpectedLogLikelihood(vp.mu, vp.tau);;

    return log_lik;
}


template <typename T> T
GetLogLikelihood(VariationalMomentParameters<T> const &vp,
                 Data const &data,
                 MonteCarloNormalParameter const &logit_p_mc) {
  if (vp.k_reg != data.k_reg) {
    throw std::runtime_error("k_reg does not match");
  }

  T log_lik = 0;
  for (int n=0; n < data.n_obs; n++) {
      log_lik += GetObservationLogLikelihood(vp, data, logit_p_mc, n);
  }

  for (int g = 0; g < vp.n_groups; g++) {
    log_lik += vp.u[g].ExpectedLogLikelihood(vp.mu, vp.tau);;
  }

  return log_lik;
}


template <typename T> T
GetEntropy(VariationalNaturalParameters<T> const &vp) {
    T entropy = 0;
    entropy =
        GetMultivariateNormalEntropy(vp.beta.info.mat) +
        GetUnivariateNormalEntropy(vp.mu.info) +
        GetGammaEntropy(vp.tau.alpha, vp.tau.beta);
    for (int g = 0; g < vp.n_groups; g++) {
        entropy += GetUnivariateNormalEntropy(vp.u[g].info);
    }
    return entropy;
}


template <typename Tlik, typename Tprior>
typename promote_args<Tlik, Tprior>::type  GetExpectedLogPrior(
    VariationalMomentParameters<Tlik> const &vp,
    PriorParameters<Tprior> const &pp) {

  typedef typename promote_args<Tlik, Tprior>::type T;

  // Convert to the promoted type.
  UnivariateNormalMoments<T> mu(vp.mu);
  MultivariateNormalMoments<T> beta(vp.beta);
  GammaMoments<T> tau(vp.tau);

  MultivariateNormalNatural<T> beta_prior(pp.beta);
  GammaNatural<T> tau_prior(pp.tau);
  UnivariateNormalNatural<T> mu_prior(pp.mu);

  T log_prior = 0.0;
  log_prior += mu.ExpectedLogLikelihood(mu_prior.loc, mu_prior.info);
  log_prior += beta.ExpectedLogLikelihood(beta_prior.loc, beta_prior.info.mat);
  log_prior += tau.ExpectedLogLikelihood(tau_prior.alpha, tau_prior.beta);

  return log_prior;
};



///////////////////////////////////////
// Functors.

// Single-group versions

// The expected log likelihood
struct GroupExpectedLogLikelihoodFunctor {
  VariationalNaturalParameters<double> base_vp;
  Data data;
  MonteCarloNormalParameter mc_param;
  int g;

  GroupExpectedLogLikelihoodFunctor(
      VariationalNaturalParameters<double> const &base_vp,
      Data const &data,
      MonteCarloNormalParameter const &mc_param,
      int g):
    base_vp(base_vp), data(data), mc_param(mc_param), g(g) {};

  template <typename T> T operator()(VectorXT<T> const &theta) const {
    VariationalNaturalParameters<T> vp(base_vp);
    SetFromGroupVector(theta, vp, g);
    VariationalMomentParameters<T> vp_mom(vp);
    T e_log_lik = GetGroupLogLikelihood(vp_mom, data, mc_param, g);
    return e_log_lik;
  }
};


// The entropy of a single group
struct GroupEntropyFunctor {
  VariationalNaturalParameters<double> base_vp;
  int g;

  GroupEntropyFunctor(
      VariationalNaturalParameters<double> const &base_vp,
      int g):
    base_vp(base_vp), g(g) {};

  template <typename T> T operator()(VectorXT<T> const &theta) const {
    VariationalNaturalParameters<T> vp(base_vp);
    SetFromGroupVector(theta, vp, g);
    return GetUnivariateNormalEntropy(vp.u[g].info);
  }
};


// The entropy of only the global parameters.
struct GlobalEntropyFunctor {
  VariationalNaturalParameters<double> base_vp;

  GlobalEntropyFunctor(
      VariationalNaturalParameters<double> const &base_vp):
    base_vp(base_vp) {};

  template <typename T> T operator()(VectorXT<T> const &theta) const {
    VariationalNaturalParameters<T> vp(base_vp);
    SetFromGlobalVector(theta, vp);
    T entropy =
        GetMultivariateNormalEntropy(vp.beta.info.mat) +
        GetUnivariateNormalEntropy(vp.mu.info) +
        GetGammaEntropy(vp.tau.alpha, vp.tau.beta);
    return entropy;
  }
};


// The expected log likelihood
struct ExpectedLogLikelihoodFunctor {
  VariationalNaturalParameters<double> base_vp;
  Data data;
  MonteCarloNormalParameter mc_param;

  ExpectedLogLikelihoodFunctor(
      VariationalNaturalParameters<double> const &base_vp,
      Data const &data,
      MonteCarloNormalParameter const &mc_param):
    base_vp(base_vp), data(data), mc_param(mc_param) {};

  template <typename T> T operator()(VectorXT<T> const &theta) const {
    VariationalNaturalParameters<T> vp(base_vp);
    SetFromVector(theta, vp);
    VariationalMomentParameters<T> vp_mom(vp);
    T e_log_lik = GetLogLikelihood(vp_mom, data, mc_param);
    return e_log_lik;
  }
};


// The expected log prior
struct ExpectedLogPriorFunctor {
  VariationalNaturalParameters<double> base_vp;
  PriorParameters<double> base_pp;
  bool global_only;

  ExpectedLogPriorFunctor(
      VariationalNaturalParameters<double> const &base_vp,
      PriorParameters<double> const &base_pp):
    base_vp(base_vp), base_pp(base_pp) {
        global_only = false;
    };

  template <typename T> T operator()(VectorXT<T> const &theta) const {
    VariationalNaturalParameters<T> vp(base_vp);
    if (global_only) {
        SetFromGlobalVector(theta, vp);
    } else {
        SetFromVector(theta, vp);
    }
    VariationalMomentParameters<T> vp_mom(vp);
    return GetExpectedLogPrior(vp_mom, base_pp);
  }
};


// The expected log prior also as a function of the prior parameters
struct FullModelExpectedLogPriorFunctor {
  VariationalNaturalParameters<double> base_vp;
  PriorParameters<double> base_pp;

  FullModelExpectedLogPriorFunctor(
      VariationalNaturalParameters<double> const &_base_vp,
      PriorParameters<double> const &_base_pp):
    base_vp(_base_vp), base_pp(_base_pp) {};

  template <typename T> T operator()(VectorXT<T> const &theta) const {
    VariationalNaturalParameters<T> vp(base_vp);
    PriorParameters<T> pp(base_pp);
    SetFromVector(theta, vp, pp);
    VariationalMomentParameters<T> vp_mom(vp);
    return GetExpectedLogPrior(vp_mom, pp);
  }
};


// The log prior evaluated at a particular draw encoded as
// VariationalMomentParameters.
struct LogPriorDrawFunctor {
  VariationalMomentParameters<double> base_obs;
  PriorParameters<double> base_pp;

  LogPriorDrawFunctor(
      VariationalMomentParameters<double> const &_base_obs,
      PriorParameters<double> const &_base_pp):
    base_obs(_base_obs), base_pp(_base_pp) {};

  template <typename T> T operator()(VectorXT<T> const &theta) const {
    VariationalMomentParameters<T> obs(base_obs);
    PriorParameters<T> pp(base_pp);
    pp.SetFromVector(theta);
    return GetExpectedLogPrior(obs, pp);
  }
};


// The expected log likelihood and the prior
struct ELBOFunctor {
  VariationalNaturalParameters<double> base_vp;
  PriorParameters<double> base_pp;
  Data data;
  MonteCarloNormalParameter mc_param;

  ELBOFunctor(
      VariationalNaturalParameters<double> const &base_vp,
      PriorParameters<double> const &base_pp,
      Data const &data,
      MonteCarloNormalParameter const &mc_param):
    base_vp(base_vp), base_pp(base_pp), data(data), mc_param(mc_param) {};

  template <typename T> T operator()(VectorXT<T> const &theta) const {
    VariationalNaturalParameters<T> vp(base_vp);
    SetFromVector(theta, vp);
    VariationalMomentParameters<T> vp_mom(vp);
    T e_log_lik = GetLogLikelihood(vp_mom, data, mc_param);
    T e_prior = GetExpectedLogPrior(vp_mom, base_pp);
    T entropy = GetEntropy(vp);
    return e_log_lik + e_prior + entropy;
  }
};


// The expected log likelihood and the prior
struct EntropyFunctor {
  VariationalNaturalParameters<double> base_vp;

  EntropyFunctor(
      VariationalNaturalParameters<double> const &base_vp):
    base_vp(base_vp) {};

  template <typename T> T operator()(VectorXT<T> const &theta) const {
    VariationalNaturalParameters<T> vp(base_vp);
    SetFromVector(theta, vp);
    T entropy = GetEntropy(vp);
    return entropy;
  }
};


// The moments as a function of the natural parameters.
struct MomentsFunctor {
  VariationalNaturalParameters<double> base_vp;

  MomentsFunctor(
      VariationalNaturalParameters<double> const &base_vp):
    base_vp(base_vp) {};

  template <typename T> VectorXT<T> operator()(VectorXT<T> const &theta) const {
    VariationalNaturalParameters<T> vp(base_vp);
    SetFromVector(theta, vp);
    VariationalMomentParameters<T> vp_mom(vp);
    VectorXT<T> theta_mom = GetParameterVector(vp_mom);
    return theta_mom;
  }
};


struct VariationalLogDensityFunctor {
    // The observation is encoded in a MomentParameters object.
    VariationalMomentParameters<double> obs;
    VariationalNaturalParameters<double> base_vp;

    bool include_beta;
    bool include_mu;
    bool include_tau;
    VectorXi include_u_groups;

    VariationalLogDensityFunctor(
        VariationalNaturalParameters<double> const &vp,
        VariationalMomentParameters<double> const & _obs) {

        base_vp = VariationalNaturalParameters<double>(vp);
        obs = VariationalMomentParameters<double>(_obs);
        include_mu = include_beta = include_tau = true;
        include_u_groups = VectorXi::Zero(base_vp.n_groups);
        for (int g = 0; g < base_vp.n_groups; g++) {
          include_u_groups(g) = g;
        }
    };

    template <typename T> T operator()(VectorXT<T> const &theta) const {
        VariationalNaturalParameters<T> vp(base_vp);
        SetFromVector(theta, vp);

        T q_log_dens = 0.0;
        if (include_beta) {
            VectorXT<T> beta_obs = obs.beta.e_vec.template cast<T>();
            q_log_dens += vp.beta.log_lik(beta_obs);
        }

        if (include_mu) {
            T mu_obs = obs.mu.e;
            q_log_dens += vp.mu.log_lik(mu_obs);
        }

        if (include_tau) {
            T tau_obs = obs.tau.e;
            q_log_dens += vp.tau.log_lik(tau_obs);
        }

        for (int g_ind = 0; g_ind < include_u_groups.size(); g_ind++) {
            int g = include_u_groups(g_ind);
            if (g < 0 || g >= vp.n_groups) {
                throw std::runtime_error("u_g q log density: g out of bounds.");
            }
            T u_g_obs = obs.u[g].e;
            q_log_dens += vp.u[g].log_lik(u_g_obs);
        }

        return q_log_dens;
    }
};



/////////////////////////////
////// Header definitions

struct Derivatives {
  double val;
  VectorXd grad;
  MatrixXd hess;

  Derivatives(double val, VectorXd grad, MatrixXd hess):
  val(val), grad(grad), hess(hess) {}
};


Derivatives GetLogLikDerivatives(
  Data data,
  VariationalNaturalParameters<double> vp,
  ModelOptions opt);


Derivatives GetLogPriorDerivatives(
  VariationalNaturalParameters<double> vp,
  PriorParameters<double> pp,
  ModelOptions opt);


Derivatives GetFullModelLogPriorDerivatives(
    VariationalNaturalParameters<double> vp,
    PriorParameters<double> pp,
    ModelOptions opt);


Derivatives GetELBODerivatives(
    Data data,
    VariationalNaturalParameters<double> vp,
    PriorParameters<double> pp,
    ModelOptions opt);


Derivatives GetEntropyDerivatives(
    VariationalNaturalParameters<double> vp,
    ModelOptions opt);


struct ProblemInstance {
  VariationalMomentParameters<double> vp_mom;
  VariationalNaturalParameters<double> vp_nat;
  Data data;

  ProblemInstance(VariationalNaturalParameters<double> vp_nat,
                  VariationalMomentParameters<double> vp_mom,
                  Data data):
    vp_nat(vp_nat), vp_mom(vp_mom), data(data) {};
};


ProblemInstance SimulateData(int n_obs, int k_reg, int n_groups);


Derivatives GetLogVariationalDensityDerivatives(
    VariationalMomentParameters<double> const &obs,
    VariationalNaturalParameters<double> &vp,
    ModelOptions const &opt);


Derivatives GetLogPriorDerivativesFromDraw(
    VariationalMomentParameters<double> &obs,
    PriorParameters<double> pp,
    ModelOptions opt);


std::vector<Triplet> GetSparseLogLikHessian(
    Data data,
    VariationalNaturalParameters<double> vp,
    PriorParameters<double> pp,
    ModelOptions opt,
    bool include_prior);



std::vector<Triplet> GetSparseEntropyHessian(
    VariationalNaturalParameters<double> vp,
    ModelOptions opt);

SparseMatrix<double> GetCovariance(
    const VariationalNaturalParameters<double> &vp,
    const Offsets moment_offsets);

Derivatives GetMomentJacobian(
    VariationalNaturalParameters<double> vp,
    ModelOptions opt);


# endif
