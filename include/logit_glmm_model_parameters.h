# ifndef LOGIT_GLMM_MODEL_PARAMETERS_H
# define LOGIT_GLMM_MODEL_PARAMETERS_H

# define INSTANTIATE_LOGIT_GLMM_MODEL_PARAMETERS_H 0

# include <Eigen/Dense>
# include <vector>
# include <iostream>
# include <string>

# include <stan/math.hpp>
# include <stan/math/fwd/mat.hpp>
# include <stan/math/mix/mat/functor/hessian.hpp>

# include "variational_parameters.h"
# include "monte_carlo_parameters.h"
# include "eigen_includes.h"
# include "boost/random.hpp"

using std::vector;


//////////////////////////////////////
// Model options
//////////////////////////////////////
class ModelOptions {
private:
    void Init() {
        unconstrained = false;
        calculate_gradient = false;
        calculate_hessian = false;
    }

    void SetDraws(int n_sim) {
        boost::mt19937 rng;
        std_draws = VectorXd::Zero(n_sim);
        for (int n = 0; n < n_sim; n++) {
            std_draws[n] = stan::math::normal_rng(0,1, rng);
        }
    }

    VectorXd std_draws;

public:
    bool unconstrained;
    bool calculate_gradient;
    bool calculate_hessian;

    ModelOptions(int n_sim) {
        Init();
        SetDraws(n_sim);
    }

    ModelOptions(VectorXd _std_draws): std_draws(_std_draws) {
        Init();
    }

    MonteCarloNormalParameter GetMonteCarloNormalParameter() {
        return MonteCarloNormalParameter(std_draws);
    }
};


/////////////////////////////////////
// Offsets

struct Offsets {
    int beta;
    int mu;
    int tau;
    vector<int> u;

    int encoded_size;

    Offsets() {
        beta = 0;
        mu = 0;
        tau = 0;
        u.resize(0);
        encoded_size = 0;
    }
};


template <typename T, template<typename> class VPType>
Offsets GetOffsets(VPType<T> vp) {
    Offsets offsets;
    int encoded_size = 0;

    offsets.beta = encoded_size;
    encoded_size += vp.beta.encoded_size;

    offsets.mu = encoded_size;
    encoded_size += vp.mu.encoded_size;

    offsets.tau = encoded_size;
    encoded_size += vp.tau.encoded_size;

    offsets.u.resize(vp.n_groups);
    for (int g = 0; g < vp.n_groups; g++) {
        offsets.u[g] = encoded_size;
        encoded_size += vp.u[g].encoded_size;
    }

    offsets.encoded_size = encoded_size;

    return offsets;
};


//////////////////////////////////////
// VariationalParameters
//////////////////////////////////////

template <class T>
class VariationalMomentParameters {
private:
    void Initialize(int _k_reg, int _n_groups, bool _unconstrained) {
        k_reg = _k_reg;
        n_groups = _n_groups;
        unconstrained = _unconstrained;

        beta = MultivariateNormalMoments<T>(k_reg);
        u.resize(n_groups);
        for (int g=0; g < n_groups; g++) {
          // For some reason C++ doesn't like assigning directly.
          UnivariateNormalMoments<T> this_u;
          u[g] = this_u;
        }
        mu = UnivariateNormalMoments<T>();
        tau = GammaMoments<T>();

        offsets = GetOffsets(*this);
    }

public:
  // For each n in n_obs, we observe a binomial
  // y_n ~ Binomial(p)
  // p = InvLogit(x_n' beta + u_{y_g{n}})
  // where beta and x_n are kx1 vectors and u_n is a random scalar
  // u_n ~ N(mu, 1 / tau)

  int n_groups;  // The number of groups
  int k_reg;  // The dimension of the fixed effects regressor
  bool unconstrained;       // Whether or not to encode unconstrained in a vector

  MultivariateNormalMoments<T> beta;
  vector<UnivariateNormalMoments<T>> u;
  UnivariateNormalMoments<T> mu;
  GammaMoments<T> tau;

  Offsets offsets;

  // Methods:
  VariationalMomentParameters(int _k_reg, int _n_groups, bool _unconstrained) {
      Initialize(_k_reg, _n_groups, _unconstrained);
  };

  VariationalMomentParameters() {
      Initialize(1, 1, true);
  };

  template <typename Tnew> operator VariationalMomentParameters<Tnew>() const {
    VariationalMomentParameters<Tnew> vp(k_reg, n_groups, unconstrained);

    vp.beta = beta;
    vp.tau = tau;
    vp.mu = mu;

    vp.u.resize(n_groups);
    for (int g=0; g < n_groups; g++) {
      vp.u[g] = u[g];
    }

    return vp;
  };
  
  
  // This is mostly useful for testing.
  void clear() {
    beta.e_vec = VectorXT<T>::Zero(k_reg);
    beta.e_outer.mat = MatrixXT<T>::Zero(k_reg, k_reg);
    tau.e = 0;
    tau.e_log = 0;
    mu.e = 0;
    mu.e2 = 0;

    for (int g = 0; g < n_groups; g++) {
      u[g].e = 0;
      u[g].e2 = 0;
    }
  }
};


template <class T>
class VariationalNaturalParameters {
private:
    void Initialize(int _k_reg, int _n_groups, bool _unconstrained) {
        k_reg = _k_reg;
        n_groups = _n_groups;
        unconstrained = _unconstrained;

        beta = MultivariateNormalNatural<T>(k_reg);
        mu = UnivariateNormalNatural<T>();
        tau = GammaNatural<T>();

        u.resize(n_groups);
        for (int g=0; g < n_groups; g++) {
          // For some reason C++ doesn't like assigning directly.
          UnivariateNormalNatural<T> this_u;
          u[g] = this_u;
        }

        offsets = GetOffsets(*this);
    }
public:
  // For each n in n_obs, we observe a binomial
  // y_n ~ Binomial(p)
  // p = InvLogit(x_n' beta + u_{y_g{n}})
  // where beta and x_n are kx1 vectors and u_n is a random scalar
  // u_n ~ N(mu, 1 / tau)

  int n_groups;  // The number of groups
  int k_reg;  // The dimension of the fixed effects regressor
  bool unconstrained;       // Whether or not to encode unconstrained in a vector

  Offsets offsets;

  MultivariateNormalNatural<T> beta;
  vector<UnivariateNormalNatural<T>> u;
  UnivariateNormalNatural<T> mu;
  GammaNatural<T> tau;

  // Methods:
  VariationalNaturalParameters(int _k_reg, int _n_groups, bool _unconstrained) {
      Initialize(_k_reg, _n_groups, _unconstrained);
  };
  VariationalNaturalParameters() { Initialize(1, 1, true); };


  template <typename Tnew> operator VariationalNaturalParameters<Tnew>() const {
    VariationalNaturalParameters<Tnew> vp(k_reg, n_groups, unconstrained);

    vp.beta = beta;
    vp.tau = tau;
    vp.mu = mu;

    vp.u.resize(n_groups);
    for (int g=0; g < n_groups; g++) {
      vp.u[g] = u[g];
    }

    return vp;
  };


  operator VariationalMomentParameters<T>() const {
    VariationalMomentParameters<T> vp_moments(k_reg, n_groups, unconstrained);
    vp_moments.beta = MultivariateNormalMoments<T>(beta);
    vp_moments.tau = GammaMoments<T>(tau);
    vp_moments.mu = UnivariateNormalMoments<T>(mu);

    vp_moments.u.resize(n_groups);
    for (int g = 0; g < n_groups; g++) {
        vp_moments.u[g] = UnivariateNormalMoments<T>(u[g]);
    }

    return vp_moments;
  }

  // This is mostly useful for testing.
  void clear() {
    beta.loc = VectorXT<T>::Zero(k_reg);
    beta.info.mat = MatrixXT<T>::Zero(k_reg, k_reg);
    tau.alpha = 0;
    tau.beta = 0;
    mu.loc = 0;
    mu.info = 0;

    for (int g = 0; g < n_groups; g++) {
      u[g].loc = 0;
      u[g].info = 0;
    }
  }
};


//////////////////////////////
// Priors

template <class T> class PriorParameters {
private:
    void Initialize(int _k_reg) {
        k_reg = _k_reg;
        beta = MultivariateNormalNatural<T>(k_reg);
        tau = GammaNatural<T>();
        mu = UnivariateNormalNatural<T>();

        vec_size = beta.encoded_size + tau.encoded_size + mu.encoded_size;
    }
public:
  // Parameters:
  int k_reg;
  int vec_size;

  MultivariateNormalNatural<T> beta;
  GammaNatural<T> tau;
  UnivariateNormalNatural<T> mu;

  // Methods:
  PriorParameters(int _k_reg) {
      Initialize(_k_reg);
  };

  PriorParameters() {
    Initialize(1);
  };

  template <typename Tnew> operator PriorParameters<Tnew>() const {
    PriorParameters<Tnew> pp(k_reg);
    pp.beta = beta;
    pp.tau = tau;
    pp.mu = mu;

    return pp;
  };

  void SetFromVector(VectorXT<T> const &theta) {
    if (theta.size() != vec_size) {
        throw std::runtime_error("Vector is the wrong size.");
    }

    VectorXT<T> theta_sub;
    int ind = 0;

    theta_sub = theta.segment(ind, beta.encoded_size);
    beta.decode_vector(theta_sub, false);
    ind += beta.encoded_size;

    theta_sub = theta.segment(ind, mu.encoded_size);
    mu.decode_vector(theta_sub, false);
    ind += mu.encoded_size;

    theta_sub = theta.segment(ind, tau.encoded_size);
    tau.decode_vector(theta_sub, false);
  }


  VectorXT<T> GetParameterVector() const {
    VectorXT<T> theta(vec_size);
    int ind = 0;

    theta.segment(ind, beta.encoded_size) = beta.encode_vector(false);
    ind += beta.encoded_size;

    theta.segment(ind, mu.encoded_size) = mu.encode_vector(false);
    ind += mu.encoded_size;

    theta.segment(ind, tau.encoded_size) = tau.encode_vector(false);
    ind += tau.encoded_size;

    return theta;
  }
};


// Set a single group's moment parameters from variational parameters.
template <class T> void SetMomentsGroup(
    VariationalMomentParameters<T> &mp,
    VariationalNaturalParameters<T> const vp, int g) {

    // Set the global parameters
    mp.beta = MultivariateNormalMoments<T>(vp.beta);
    mp.tau = GammaMoments<T>(vp.tau);
    mp.mu = UnivariateNormalMoments<T>(vp.mu);

    mp.u[g] = UnivariateNormalMoments<T>(vp.u[g]);
};


//////////////////////////////
// Model data

struct Data {
  int n_obs;    // The number of individuals
  int n_groups; // The number of groups
  int k_reg;    // The dimension of the regressors.

  VectorXi y;   // Binary outcomes.
  VectorXi y_g; // The group of observation n.
  MatrixXd x;   // Regressors.

  void Initialize(MatrixXi const & _y, MatrixXi const & _y_g, MatrixXd const & _x) {
      y = _y;
      y_g = _y_g;
      x = _x;
      n_obs = x.rows();
      k_reg = x.cols();

      // int min_g_index, max_g_index;
      // int min_g = y_g.minCoeff(&min_g_index);
      // int max_g = y_g.maxCoeff(&max_g_index);
      int min_g = y_g.minCoeff();
      int max_g = y_g.maxCoeff();
      if (min_g < 0) {
        throw std::runtime_error("Error -- y_g must have integers between 0 and n_g - 1");
      }
      n_groups = max_g;

      int min_y = y.minCoeff();
      int max_y = y.maxCoeff();
      if (min_y < 0 || max_y > 1) {
        throw std::runtime_error("Error -- y must be binary.");
      }

      if (y.rows() != x.rows()) {
        throw std::runtime_error("Y rows do not match X rows.");
      }
  }

  // Constructor
  Data(MatrixXi _y, MatrixXi _y_g, MatrixXd _x) {
      Initialize(_y, _y_g, _x);
  };

  Data() {
    VectorXi _y(1);
    _y << 0;
    VectorXi _y_g(1);
    _y_g << 1;
    MatrixXd _x(1, 1);
    _x << 1;
    Initialize(_y, _y_g, _x);
  }
};


# include "logit_glmm_model_encoders.h"


//////////////////////////////////////
// Sparse hessians.
// Func should be a functor that evaluates some objective for group g
// when its member functor.g = g
template <typename Func>
std::vector<Triplet> GetSparseHessian(
    Func functor, VariationalNaturalParameters<double> vp) {

    std::cout << "Starting sparse Hessian\n";
    std::vector<Triplet> all_terms;

    // Pre-allocate memory.
    VectorXd theta = GetGroupParameterVector(vp, 0);
    double val;
    VectorXd grad = VectorXd::Zero(theta.size());
    MatrixXd hess = MatrixXd::Zero(theta.size(), theta.size());

    for (int g = 0; g < vp.n_groups; g++) {
        functor.g = g;
        theta = GetGroupParameterVector(vp, g);

        stan::math::hessian(functor, theta, val, grad, hess);

        // The size of the beta, mu, and tau parameters
        for (int i1=0; i1 < theta.size(); i1++) {
            int gi1 = GlobalIndex(i1, g, vp.offsets);
            for (int i2=0; i2 < theta.size(); i2++) {
                int gi2 = GlobalIndex(i2, g, vp.offsets);
                all_terms.push_back(Triplet(gi1, gi2, hess(i1, i2)));
            }
        }
    }
    std::cout << "\nDone with sparse Hessian\n";
    return all_terms;
}


# if INSTANTIATE_LOGIT_GLMM_MODEL_PARAMETERS_H
    // For instantiation:
    // # include <stan/math.hpp>
    // # include "stan/math/fwd/scal.hpp"

    using var = stan::math::var;
    using fvar = stan::math::fvar<var>;

    extern template class VariationalMomentParameters<double>;
    extern template class VariationalMomentParameters<var>;
    extern template class VariationalMomentParameters<fvar>;

    extern template class VariationalNaturalParameters<double>;
    extern template class VariationalNaturalParameters<var>;
    extern template class VariationalNaturalParameters<fvar>;

    extern template class PriorParameters<double>;
    extern template class PriorParameters<var>;
    extern template class PriorParameters<fvar>;
# endif

# endif
