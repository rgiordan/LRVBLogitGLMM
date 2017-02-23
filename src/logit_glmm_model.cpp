# include "eigen_includes.h"

# include "logit_glmm_model_parameters.h"
# include "logit_glmm_model.h"

# include "transform_hessian.h"
#include "boost/random.hpp"

// # include <stan/math.hpp>
// # include <stan/math/mix/mat/functor/hessian.hpp>


Derivatives GetLogLikDerivatives(
      Data data,
      VariationalNaturalParameters<double> vp,
      ModelOptions opt) {

  vp.unconstrained = opt.unconstrained;
  MonteCarloNormalParameter mc_param = opt.GetMonteCarloNormalParameter();
  ExpectedLogLikelihoodFunctor ExpectedLogLikelihood(vp, data, mc_param);

  double e_model;
  VectorXd grad = VectorXd::Zero(vp.offsets.encoded_size);
  MatrixXd hess = MatrixXd::Zero(vp.offsets.encoded_size, vp.offsets.encoded_size);
  VectorXd theta = GetParameterVector(vp);

  stan::math::set_zero_all_adjoints();
  if (opt.calculate_hessian) {
    stan::math::hessian(ExpectedLogLikelihood, theta, e_model, grad, hess);
  } else if (opt.calculate_gradient) {
    stan::math::gradient(ExpectedLogLikelihood, theta, e_model, grad);
  } else {
    e_model = ExpectedLogLikelihood(theta);
  }

  return Derivatives(e_model, grad, hess);
}


Derivatives GetLogPriorDerivatives(
  VariationalNaturalParameters<double> vp,
  PriorParameters<double> pp,
  ModelOptions opt) {

  vp.unconstrained = opt.unconstrained;
  ExpectedLogPriorFunctor ExpectedLogPrior(vp, pp);

  double e_model;
  VectorXd grad = VectorXd::Zero(vp.offsets.encoded_size);
  MatrixXd hess = MatrixXd::Zero(vp.offsets.encoded_size, vp.offsets.encoded_size);
  VectorXd theta = GetParameterVector(vp);

  stan::math::set_zero_all_adjoints();
  if (opt.calculate_hessian) {
    stan::math::hessian(ExpectedLogPrior, theta, e_model, grad, hess);
  } else if (opt.calculate_gradient) {
    stan::math::gradient(ExpectedLogPrior, theta, e_model, grad);
  } else {
    e_model = ExpectedLogPrior(theta);
  }

  return Derivatives(e_model, grad, hess);
}


Derivatives GetFullModelLogPriorDerivatives(
    VariationalNaturalParameters<double> vp,
    PriorParameters<double> pp,
    ModelOptions opt) {

    vp.unconstrained = opt.unconstrained;
    FullModelExpectedLogPriorFunctor ExpectedLogPrior(vp, pp);

    double e_model;
    int vec_size = vp.offsets.encoded_size + pp.vec_size;
    VectorXd grad = VectorXd::Zero(vec_size);
    MatrixXd hess = MatrixXd::Zero(vec_size, vec_size);
    VectorXd theta = GetParameterVector(vp, pp);

    stan::math::set_zero_all_adjoints();
    if (opt.calculate_hessian) {
        stan::math::hessian(ExpectedLogPrior, theta, e_model, grad, hess);
    } else if (opt.calculate_gradient) {
        stan::math::gradient(ExpectedLogPrior, theta, e_model, grad);
    } else {
        e_model = ExpectedLogPrior(theta);
    }

    return Derivatives(e_model, grad, hess);
}


// Get the derivatives of the log prior evaluated at a draw encoded
// as VariationalMomentParameters with respect to the prior parameters.
Derivatives GetLogPriorDerivativesFromDraw(
    VariationalMomentParameters<double> &obs,
    PriorParameters<double> pp,
    ModelOptions opt) {

    LogPriorDrawFunctor LogPriorDraw(obs, pp);

    double val;
    VectorXd grad = VectorXd::Zero(pp.vec_size);
    MatrixXd hess = MatrixXd::Zero(pp.vec_size, pp.vec_size);
    VectorXd theta = pp.GetParameterVector();

    stan::math::set_zero_all_adjoints();
    if (opt.calculate_hessian) {
        stan::math::hessian(LogPriorDraw, theta, val, grad, hess);
    } else if (opt.calculate_gradient) {
        stan::math::gradient(LogPriorDraw, theta, val, grad);
    } else {
        val = LogPriorDraw(theta);
    }

    return Derivatives(val, grad, hess);
}


Derivatives GetELBODerivatives(
    Data data,
    VariationalNaturalParameters<double> vp,
    PriorParameters<double> pp,
    ModelOptions opt) {

    vp.unconstrained = opt.unconstrained;
    MonteCarloNormalParameter mc_param = opt.GetMonteCarloNormalParameter();
    ELBOFunctor ELBO(vp, pp, data, mc_param);

    double e_model;
    VectorXd grad = VectorXd::Zero(vp.offsets.encoded_size);
    MatrixXd hess = MatrixXd::Zero(vp.offsets.encoded_size, vp.offsets.encoded_size);
    VectorXd theta = GetParameterVector(vp);

    stan::math::set_zero_all_adjoints();
    if (opt.calculate_hessian) {
      stan::math::hessian(ELBO, theta, e_model, grad, hess);
    } else if (opt.calculate_gradient) {
      stan::math::gradient(ELBO, theta, e_model, grad);
    } else {
      e_model = ELBO(theta);
    }

    return Derivatives(e_model, grad, hess);
}


Derivatives GetEntropyDerivatives(
    VariationalNaturalParameters<double> vp,
    ModelOptions opt) {

    vp.unconstrained = opt.unconstrained;
    EntropyFunctor Entropy(vp);

    double e_model;
    VectorXd grad = VectorXd::Zero(vp.offsets.encoded_size);
    MatrixXd hess = MatrixXd::Zero(vp.offsets.encoded_size, vp.offsets.encoded_size);
    VectorXd theta = GetParameterVector(vp);

    stan::math::set_zero_all_adjoints();
    if (opt.calculate_hessian) {
      stan::math::hessian(Entropy, theta, e_model, grad, hess);
    } else if (opt.calculate_gradient) {
      stan::math::gradient(Entropy, theta, e_model, grad);
    } else {
      e_model = Entropy(theta);
    }

    return Derivatives(e_model, grad, hess);
}


Derivatives GetLogVariationalDensityDerivatives(
    VariationalMomentParameters<double> const &obs,
    VariationalNaturalParameters<double> &vp,
    ModelOptions const &opt,
    bool global_only,
    bool include_beta,
    bool include_mu,
    bool include_tau) {

    vp.unconstrained = opt.unconstrained;
    VariationalLogDensityFunctor VariationalLogDensity(vp, obs);
    VariationalLogDensity.global_only = global_only;
    VariationalLogDensity.include_beta = include_beta;
    VariationalLogDensity.include_mu = include_mu;
    VariationalLogDensity.include_tau = include_tau;

    VectorXd theta;
    if (global_only) {
        theta = GetGlobalParameterVector(vp);
    } else {
        theta = GetParameterVector(vp);
    }

    double val;
    VectorXd grad = VectorXd::Zero(theta.size());
    MatrixXd hess = MatrixXd::Zero(theta.size(), theta.size());

    stan::math::set_zero_all_adjoints();
    if (opt.calculate_hessian) {
      stan::math::hessian(VariationalLogDensity, theta, val, grad, hess);
    } else if (opt.calculate_gradient) {
      stan::math::gradient(VariationalLogDensity, theta, val, grad);
    } else {
      val = VariationalLogDensity(theta);
    }

    return Derivatives(val, grad, hess);
}



std::vector<Triplet> GetSparseLogLikHessian(
    Data data,
    VariationalNaturalParameters<double> vp,
    PriorParameters<double> pp,
    ModelOptions opt,
    bool include_prior) {

    vp.unconstrained = opt.unconstrained;
    MonteCarloNormalParameter mc_param = opt.GetMonteCarloNormalParameter();
    GroupExpectedLogLikelihoodFunctor LogLik(vp, data, mc_param, 0);

    std::vector<Triplet> all_terms = GetSparseHessian(LogLik, vp);

    if (include_prior) {
        ExpectedLogPriorFunctor ExpectedLogPrior(vp, pp);
        ExpectedLogPrior.global_only = true;
        double val;
        VectorXd theta = GetGlobalParameterVector(vp);
        VectorXd grad = VectorXd::Zero(theta.size());
        MatrixXd hess = MatrixXd::Zero(theta.size(), theta.size());

        stan::math::set_zero_all_adjoints();
        stan::math::hessian(ExpectedLogPrior, theta, val, grad, hess);

        for (int i1 = 0; i1 < hess.rows(); i1++) {
            for (int i2 = 0; i2 < hess.cols(); i2++) {
                all_terms.push_back(Triplet(i1, i2, hess(i1, i2)));
            }
        }
    }

    return all_terms;
}


std::vector<Triplet> GetSparseEntropyHessian(
    VariationalNaturalParameters<double> vp,
    ModelOptions opt) {

    vp.unconstrained = opt.unconstrained;
    GroupEntropyFunctor GroupEntropy(vp, 0);

    // The entropy of the group terms.
    std::vector<Triplet> all_terms = GetSparseHessian(GroupEntropy, vp);

    // The entropy of the global terms.
    GlobalEntropyFunctor GlobalEntropy(vp);
    double val;
    VectorXd theta = GetGlobalParameterVector(vp);
    VectorXd grad = VectorXd::Zero(theta.size());
    MatrixXd hess = MatrixXd::Zero(theta.size(), theta.size());

    stan::math::set_zero_all_adjoints();
    stan::math::hessian(GlobalEntropy, theta, val, grad, hess);

    for (int i1 = 0; i1 < hess.rows(); i1++) {
        for (int i2 =0; i2 < hess.cols(); i2++) {
            all_terms.push_back(Triplet(i1, i2, hess(i1, i2)));
        }
    }

    return all_terms;
}


std::vector<Triplet> GetSparseELBOHessianTerms(
    Data data,
    VariationalNaturalParameters<double> vp,
    PriorParameters<double> pp,
    ModelOptions opt,
    bool include_prior) {

    vp.unconstrained = opt.unconstrained;
    MonteCarloNormalParameter mc_param = opt.GetMonteCarloNormalParameter();
    GroupELBOFunctor ELBO(vp, data, mc_param, 0);

    std::vector<Triplet> all_terms = GetSparseHessian(ELBO, vp);

    if (include_prior) {
        ExpectedLogPriorFunctor ExpectedLogPrior(vp, pp);
        ExpectedLogPrior.global_only = true;
        double val;
        VectorXd theta = GetGlobalParameterVector(vp);
        VectorXd grad = VectorXd::Zero(theta.size());
        MatrixXd hess = MatrixXd::Zero(theta.size(), theta.size());

        stan::math::set_zero_all_adjoints();
        stan::math::hessian(ExpectedLogPrior, theta, val, grad, hess);

        for (int i1 = 0; i1 < hess.rows(); i1++) {
            for (int i2 = 0; i2 < hess.cols(); i2++) {
                all_terms.push_back(Triplet(i1, i2, hess(i1, i2)));
            }
        }
    }

    return all_terms;
}


// The return type will not actually be derivatives, but I'll use the
// return type to avoid defining a type just for this.  The "grad" will
// be the moment vector and the "hess" will be the Jacobian.
Derivatives GetMomentJacobian(
    VariationalNaturalParameters<double> vp,
    ModelOptions opt) {

    vp.unconstrained = opt.unconstrained;
    MomentsFunctor Moments(vp);

    VectorXd mom_vec;
    MatrixXd mom_jac = MatrixXd::Zero(vp.offsets.encoded_size,
                                      vp.offsets.encoded_size);
    VectorXd theta = GetParameterVector(vp);

    stan::math::set_zero_all_adjoints();
    stan::math::jacobian(Moments, theta, mom_vec, mom_jac);

    bool jac_correct = CheckJacobianCorrectOrientation();
    if (!jac_correct) {
        MatrixXd mom_jac_t = mom_jac.transpose();
        mom_jac = mom_jac_t;
    }

    // Abuse the Derivatives type.
    return Derivatives(0, mom_vec, mom_jac);
}


ProblemInstance SimulateData(int n_obs, int k_reg, int n_groups) {

  VariationalNaturalParameters<double> vp_nat(k_reg, n_groups, true);
  MatrixXd x = MatrixXd::Random(n_obs, k_reg);
  VectorXi y(n_obs);
  VectorXi y_g(n_obs);

  VectorXd beta = VectorXd::Zero(k_reg);
  MatrixXd beta_var = MatrixXd::Zero(k_reg, k_reg);
  for (int k = 0; k < k_reg; k++) {
    beta(k) = k;
    beta_var(k ,k) = 1;
  }
  MatrixXd beta_info = beta_var.inverse();
  vp_nat.beta.loc = beta;
  vp_nat.beta.info.set(beta_info);
  vp_nat.beta.diag_min = 0.2;

  double row_var = 0.1;
  vp_nat.tau.alpha = 1 / row_var;
  vp_nat.tau.beta = 2;
  vp_nat.tau.alpha_min = 0.5;
  vp_nat.tau.beta_min = 0.5;

  vp_nat.mu.loc = 0.1;
  double mu_var = 1;
  vp_nat.mu.info = 1 / mu_var;
  vp_nat.mu.info_min = 0.3;

  boost::mt19937 rng;
  for (int g = 0; g < vp_nat.n_groups; g++) {
    vp_nat.u[g].loc = stan::math::normal_rng(vp_nat.mu.loc, sqrt(mu_var), rng);
    vp_nat.u[g].info = 1 / (pow(vp_nat.u[g].loc, 2) + 0.1);
    vp_nat.u[g].info_min = vp_nat.u[g].info * 0.1;
  }

  for (int n = 0; n < n_obs; n++) {
    int g = n % n_groups;
    y_g(n) = g;
    double row_rho = x.row(n).dot(beta) + vp_nat.u[g].loc;
    double row_prob = stan::math::inv_logit(row_rho);
    y(n) = stan::math::binomial_rng(1, row_prob, rng);
  }

  VariationalMomentParameters<double> vp_mom(vp_nat);
  vp_mom.beta.diag_min = vp_mom.beta.e_outer.mat(0, 0) * 0.1;
  vp_mom.mu.e2_min = vp_mom.mu.e2 * 0.1;
  vp_mom.tau.e_min = vp_mom.tau.e * 0.1;
  for (int g = 0; g < vp_nat.n_groups; g++) {
      vp_mom.u[g].e2_min = vp_mom.u[g].e2 * 0.1;
  }

  Data data(y, y_g, x);
  return ProblemInstance(vp_nat, vp_mom, data);
}


// Get the covariance of the moment parameters from the natural parameters.
SparseMatrix<double> GetCovariance(
    const VariationalNaturalParameters<double> &vp,
    const Offsets moment_offsets) {

  std::vector<Triplet> all_terms;
  std::vector<Triplet> terms;

  if (vp.offsets.encoded_size != moment_offsets.encoded_size) {
      std::ostringstream err_msg;
      err_msg << "Size mismatch.  Natural parameter encoded size: " <<
          vp.offsets.encoded_size << "  Moment parameter encoded size: " <<
          moment_offsets.encoded_size;
      throw std::runtime_error(err_msg.str());
  }

  terms = GetMomentCovariance(vp.beta, moment_offsets.beta);
  all_terms.insert(all_terms.end(), terms.begin(), terms.end());

  terms = GetMomentCovariance(vp.mu, moment_offsets.mu);
  all_terms.insert(all_terms.end(), terms.begin(), terms.end());

  terms = GetMomentCovariance(vp.tau, moment_offsets.tau);
  all_terms.insert(all_terms.end(), terms.begin(), terms.end());

  for (int g = 0; g < vp.n_groups; g++) {
      terms = GetMomentCovariance(vp.u[g], moment_offsets.u[g]);
      all_terms.insert(all_terms.end(), terms.begin(), terms.end());
  }

  // Construct a sparse matrix.
  SparseMatrix<double>
    theta_cov(moment_offsets.encoded_size, moment_offsets.encoded_size);
  theta_cov.setFromTriplets(all_terms.begin(), all_terms.end());
  theta_cov.makeCompressed();

  return theta_cov;
}
