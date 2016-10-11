# include "logit_glmm_model_parameters.h"
# include "logit_glmm_model.h"
# include "gtest/gtest.h"

# include <stan/math.hpp>

// For RNGs
# include <stan/math/prim/scal.hpp>

# include <string>
# include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

# include "test_eigen.h"

// This is a typedef for a random number generator.
// Try boost::mt19937 or boost::ecuyer1988 instead of boost::minstd_rand
typedef boost::mt19937 RNGType;


void TestVariationalGlobalParametersEqual(
    VariationalMomentParameters<double> vp1,
    VariationalMomentParameters<double> vp2,
    std::string msg) {

  EXPECT_EQ(vp1.k_reg, vp2.k_reg) << msg;

  EXPECT_VECTOR_EQ(vp1.beta.e_vec, vp2.beta.e_vec, msg);
  EXPECT_MATRIX_EQ(vp1.beta.e_outer.mat, vp2.beta.e_outer.mat, msg);

  EXPECT_DOUBLE_EQ(vp1.tau.e, vp2.tau.e) << msg;
  EXPECT_DOUBLE_EQ(vp1.tau.e_log, vp2.tau.e_log) << msg;

  EXPECT_DOUBLE_EQ(vp1.mu.e, vp2.mu.e) << msg;
  EXPECT_DOUBLE_EQ(vp1.mu.e2, vp2.mu.e2) << msg;
}


void TestVariationalGroupParametersEqual(
    VariationalMomentParameters<double> vp1, VariationalMomentParameters<double> vp2,
    int g, std::string msg) {

    EXPECT_DOUBLE_EQ(vp1.u[g].e, vp2.u[g].e) << msg;
    EXPECT_DOUBLE_EQ(vp1.u[g].e2, vp2.u[g].e2) << msg;
}


void TestVariationalParametersEqual(
    VariationalMomentParameters<double> vp1, VariationalMomentParameters<double> vp2,
    std::string msg) {

    TestVariationalGlobalParametersEqual(vp1, vp2, msg);

    EXPECT_EQ(vp1.n_groups, vp2.n_groups) << msg;
    for (int g = 0; g < vp1.n_groups; g++) {
        TestVariationalGroupParametersEqual(vp1, vp2, g, msg);
    }
}

void TestPriorParametersEqual(
    PriorParameters<double> pp1, PriorParameters<double> pp2,
    std::string msg) {

    EXPECT_EQ(pp1.k_reg, pp2.k_reg) << msg;

    EXPECT_VECTOR_EQ(pp1.beta.loc, pp2.beta.loc, msg);
    EXPECT_MATRIX_EQ(pp1.beta.info.mat, pp2.beta.info.mat, msg);

    EXPECT_DOUBLE_EQ(pp1.tau.alpha, pp2.tau.alpha) << msg;
    EXPECT_DOUBLE_EQ(pp1.tau.beta, pp2.tau.beta) << msg;

    EXPECT_DOUBLE_EQ(pp1.mu.loc, pp2.mu.loc) << msg;
    EXPECT_DOUBLE_EQ(pp1.mu.info, pp2.mu.info) << msg;
}


PriorParameters<double> SamplePrior(VariationalMomentParameters<double> vp) {
  PriorParameters<double> pp(vp.k_reg);
  pp.beta.loc << 1, 2, 3;
  pp.beta.info.mat = vp.beta.e_outer.mat.inverse();
  pp.mu.loc = 1;
  pp.mu.info = 2;
  pp.tau.alpha = 3;
  pp.tau.beta = 4;
  return pp;
}


TEST(SetFromVector, basic) {
  ProblemInstance test_case = SimulateData(6, 3, 2);
  VariationalMomentParameters<double> vp = test_case.vp_mom;
  VariationalMomentParameters<double> vp_copy(vp);

  for (int ind = 0; ind < 2; ind++) {
    bool unconstrained = (ind == 0 ? true: false);
    std::string unconstrained_str =
      (unconstrained ? "unconstrained": "constrained");
    vp.unconstrained = unconstrained;
    vp_copy.unconstrained = unconstrained;
    VectorXd theta = GetParameterVector(vp);
    vp_copy.clear();
    SetFromVector(theta, vp_copy);
    TestVariationalParametersEqual(vp, vp_copy, unconstrained_str);
  }
}


TEST(SetFromVector, global) {
    ProblemInstance test_case = SimulateData(6, 3, 2);
    VariationalMomentParameters<double> vp = test_case.vp_mom;
    VariationalMomentParameters<double> vp_copy(vp);

    // Presumably the unconstrained is tested adequately elsewhere.
    vp.unconstrained = false;
    vp_copy.unconstrained = false;

    VectorXd theta = GetGlobalParameterVector(vp);
    vp_copy.clear();
    SetFromGlobalVector(theta, vp_copy);
    TestVariationalGlobalParametersEqual(vp, vp_copy, "Global vector");
}


TEST(SetFromVector, group) {
    ProblemInstance test_case = SimulateData(6, 3, 2);
    VariationalMomentParameters<double> vp = test_case.vp_mom;
    VariationalMomentParameters<double> vp_copy(vp);

    // Presumably the unconstrained is tested adequately elsewhere.
    vp.unconstrained = false;
    vp_copy.unconstrained = false;

    int g = 1;

    VectorXd theta = GetGroupParameterVector(vp, g);
    vp_copy.clear();
    SetFromGroupVector(theta, vp_copy, g);
    TestVariationalGlobalParametersEqual(vp, vp_copy, "Global vector");
    TestVariationalGroupParametersEqual(vp, vp_copy, g, "Global vector");
}


TEST(SetFromVector, combined) {
    ProblemInstance test_case = SimulateData(6, 3, 2);
    VariationalNaturalParameters<double> vp = test_case.vp_nat;
    PriorParameters<double> pp = SamplePrior(test_case.vp_mom);

    VariationalNaturalParameters<double> vp_copy(vp);
    PriorParameters<double> pp_copy(pp.k_reg);

    // Presumably the unconstrained is tested adequately elsewhere.
    vp.unconstrained = vp_copy.unconstrained = true;

    VectorXd theta = GetParameterVector(vp, pp);
    vp_copy.clear();
    SetFromVector(theta, vp_copy, pp_copy);
    TestVariationalParametersEqual(vp, vp_copy, "Global vector");

}


TEST(Model, basic) {
  ProblemInstance test_case = SimulateData(10, 3, 2);
  VariationalMomentParameters<double> vp_mom = test_case.vp_mom;
  VariationalNaturalParameters<double> vp_nat = test_case.vp_nat;
  PriorParameters<double> pp = SamplePrior(vp_mom);
  Data data = test_case.data;
  ModelOptions opt(10);
  MonteCarloNormalParameter mc_param = opt.GetMonteCarloNormalParameter();

  double e_log_lik = GetLogLikelihood(vp_mom, data, mc_param);
  double e_log_prior = GetExpectedLogPrior(vp_mom, pp);

  ExpectedLogLikelihoodFunctor ExpectedLogLikelihood(vp_nat, data, mc_param);
  VectorXd theta_nat = GetParameterVector(vp_nat);
  ExpectedLogLikelihood(theta_nat);

  ExpectedLogPriorFunctor ExpectedLogPrior(vp_nat, pp);
  ExpectedLogPrior(theta_nat);


  VariationalMomentParameters<stan::math::var> vp_stan_mom(vp_mom);

  // std::cout << "Stan log lik\n";
  stan::math::var stan_e_log_lik = GetLogLikelihood(vp_stan_mom, data, mc_param);
  // std::cout << "Stan log prior\n";
  stan::math::var stan_e_log_prior = GetExpectedLogPrior(vp_stan_mom, pp);

  // Just test that these run without error.
  VectorXT<stan::math::var> theta_stan = theta_nat.template cast<stan::math::var>();
  VariationalNaturalParameters<stan::math::var> vp_stan_nat(vp_nat);
  SetFromVector(theta_stan, vp_stan_nat);
  // std::cout << "Stan log lik functor\n";
  ExpectedLogLikelihood(theta_stan);
  // std::cout << "Stan log prior functor\n";
  ExpectedLogPrior(theta_stan);
  // opt.unconstrained = true;
  // opt.calculate_hessian = true;
  // Derivatives derivs = GetModelDerivatives(data, vp, pp, opt);
  // opt.calculate_hessian = false;
  // derivs = GetModelDerivatives(data, vp, pp, opt);
  //
  // opt.unconstrained = false;
  // opt.calculate_hessian = true;
  // derivs = GetModelDerivatives(data, vp, pp, opt);
  // opt.calculate_hessian = false;
  // derivs = GetModelDerivatives(data, vp, pp, opt);
}


TEST(GroupIndexing, basic) {
    ProblemInstance test_case = SimulateData(10, 3, 3);

    for (int g = 0; g < test_case.vp_nat.n_groups; g++) {
        VectorXd theta = GetParameterVector(test_case.vp_nat);
        VectorXd theta_g = GetGroupParameterVector(test_case.vp_nat, g);

        for (int ind_g = 0; ind_g < theta_g.size(); ind_g++) {
            int ind = GlobalIndex(ind_g, g, test_case.vp_nat.offsets);
            EXPECT_DOUBLE_EQ(theta(ind), theta_g(ind_g));
        }
    }
}


TEST(GroupFunctor, basic) {
    // With only one group they should be the same.
    ProblemInstance test_case = SimulateData(10, 3, 1);

    ModelOptions opt(10);
    MonteCarloNormalParameter mc_param = opt.GetMonteCarloNormalParameter();

    VectorXd theta = GetParameterVector(test_case.vp_nat);
    ExpectedLogLikelihoodFunctor
        LogLik(test_case.vp_nat, test_case.data, mc_param);
    GroupExpectedLogLikelihoodFunctor
        GroupLogLik(test_case.vp_nat, test_case.data, mc_param, 0);

    EXPECT_DOUBLE_EQ(LogLik(theta), GroupLogLik(theta));

    // With multiple groups the sum of the group likelihood should be the
    // overall likelihood
    test_case = SimulateData(10, 3, 3);

    LogLik =
        ExpectedLogLikelihoodFunctor(test_case.vp_nat, test_case.data, mc_param);
    GroupLogLik =
        GroupExpectedLogLikelihoodFunctor(test_case.vp_nat, test_case.data, mc_param, 0);

    theta = GetParameterVector(test_case.vp_nat);
    double log_lik = LogLik(theta);

    PriorParameters<double> pp = SamplePrior(test_case.vp_mom);
    double group_log_lik = 0;
    for (int g=0; g < test_case.vp_nat.n_groups; g++) {
        VectorXd theta_g = GetGroupParameterVector(test_case.vp_nat, g);
        GroupLogLik.g = g;
        group_log_lik += GroupLogLik(theta_g);
    }

    EXPECT_DOUBLE_EQ(log_lik, group_log_lik);
}


TEST(LogPrior, global) {
    ProblemInstance test_case = SimulateData(4, 3, 3);
    PriorParameters<double> pp = SamplePrior(test_case.vp_mom);

    ExpectedLogPriorFunctor GlobalExpectedLogPrior(test_case.vp_nat, pp);
    GlobalExpectedLogPrior.global_only = true;

    ExpectedLogPriorFunctor ExpectedLogPrior(test_case.vp_nat, pp);
    ExpectedLogPrior.global_only = false;

    VectorXd theta_global = GetGlobalParameterVector(test_case.vp_nat);
    VectorXd theta = GetParameterVector(test_case.vp_nat);

    EXPECT_DOUBLE_EQ(ExpectedLogPrior(theta), GlobalExpectedLogPrior(theta_global));
}


TEST(LogLikelihood, hessian) {
    ProblemInstance test_case = SimulateData(4, 2, 3);
    ModelOptions opt(10);
    opt.unconstrained = true;

    PriorParameters<double> pp = SamplePrior(test_case.vp_mom);

    std::vector<Triplet> terms =
        GetSparseLogLikHessian(test_case.data, test_case.vp_nat, pp, opt, true);

    opt.calculate_hessian = true;
    Derivatives derivs = GetLogLikDerivatives(test_case.data, test_case.vp_nat, opt);
    Derivatives prior_derivs = GetLogPriorDerivatives(test_case.vp_nat, pp, opt);

    int dim = test_case.vp_nat.offsets.encoded_size;
    SparseMatrix<double> hess(dim, dim);
    hess.setFromTriplets(terms.begin(), terms.end());
    hess.makeCompressed();
    MatrixXd hess_dense(hess);

    MatrixXd combined_hess = derivs.hess + prior_derivs.hess;
    EXPECT_MATRIX_NEAR(combined_hess, hess_dense, 1e-12, "With prior");

    // Test without the prior.
    terms = GetSparseLogLikHessian(test_case.data, test_case.vp_nat, pp, opt, false);
    SparseMatrix<double> hess_noprior(dim, dim);
    hess_noprior.setFromTriplets(terms.begin(), terms.end());
    hess_noprior.makeCompressed();
    hess_dense = MatrixXd(hess_noprior);
    EXPECT_MATRIX_NEAR(derivs.hess, hess_dense, 1e-12, "Without prior");
}


TEST(Entropy, hessian) {
    ProblemInstance test_case = SimulateData(4, 3, 3);
    ModelOptions opt(10);

    PriorParameters<double> pp = SamplePrior(test_case.vp_mom);

    std::vector<Triplet> terms = GetSparseEntropyHessian(test_case.vp_nat, opt);

    opt.calculate_hessian = true;
    Derivatives derivs = GetEntropyDerivatives(test_case.vp_nat, opt);

    int dim = test_case.vp_nat.offsets.encoded_size;
    SparseMatrix<double> hess(dim, dim);
    hess.setFromTriplets(terms.begin(), terms.end());
    hess.makeCompressed();
    MatrixXd hess_dense(hess);

    EXPECT_MATRIX_NEAR(hess_dense, derivs.hess, 1e-12, "Hessian");
}
