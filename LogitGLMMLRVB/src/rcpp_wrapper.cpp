# include <Rcpp.h>
# include <RcppEigen.h>

# include <vector>
# include <ctime>

# include "logit_glmm_model.h"
# include "logit_glmm_model_parameters.h"

# include <Eigen/Dense>


ModelOptions ConvertListToOption(Rcpp::List r_opt) {
    VectorXd std_draws = Rcpp::as<VectorXd>(r_opt["std_draws"]);
    ModelOptions opt(std_draws);

    opt.unconstrained = Rcpp::as<bool>(r_opt["unconstrained"]);
    opt.calculate_gradient = Rcpp::as<bool>(r_opt["calculate_gradient"]);
    opt.calculate_hessian = Rcpp::as<bool>(r_opt["calculate_hessian"]);

    return opt;
}


Rcpp::List
ConvertParametersToList(VariationalMomentParameters<double> const &vp) {

    Rcpp::List r_list;
    r_list["encoded_size"] = vp.offsets.encoded_size;
    r_list["n_groups"] = vp.n_groups;
    r_list["k_reg"] = vp.k_reg;

    r_list["beta_e_vec"] = vp.beta.e_vec;
    r_list["beta_e_outer"] = vp.beta.e_outer.mat;

    r_list["mu_e"] = vp.mu.e;
    r_list["mu_e2"] = vp.mu.e2;

    r_list["tau_e"] = vp.tau.e;
    r_list["tau_e_log"] = vp.tau.e_log;

    Rcpp::List u(vp.n_groups);
    for (int g = 0; g < vp.n_groups; g++) {
        Rcpp::List this_u;
        this_u["u_e"] = vp.u[g].e;
        this_u["u_e2"] = vp.u[g].e2;
        u[g] = this_u;
    }
    r_list["u"] = u;

    return r_list;
};


void ConvertParametersFromList(
    VariationalMomentParameters<double> &vp, Rcpp::List r_list) {

    vp.n_groups = r_list["n_groups"];
    vp.k_reg = r_list["k_reg"];

    vp.beta.e_vec = Rcpp::as<VectorXd>(r_list["beta_e_vec"]);
    vp.beta.e_outer.mat = Rcpp::as<MatrixXd>(r_list["beta_e_outer"]);

    vp.mu.e = Rcpp::as<double>(r_list["mu_e"]);
    vp.mu.e2 = Rcpp::as<double>(r_list["mu_e2"]);

    vp.tau.e = Rcpp::as<double>(r_list["tau_e"]);
    vp.tau.e_log = Rcpp::as<double>(r_list["tau_e_log"]);

    Rcpp::List u_list = r_list["u"];
    if (vp.n_groups != u_list.size()) {
        throw std::runtime_error("u size does not match");
    }

    for (int g = 0; g < vp.n_groups; g++) {
        Rcpp::List this_u = u_list[g];
        vp.u[g].e = Rcpp::as<double>(this_u["u_e"]);
        vp.u[g].e2 = Rcpp::as<double>(this_u["u_e2"]);
    }
}


Rcpp::List ConvertParametersToList(VariationalNaturalParameters<double> const &vp) {

    Rcpp::List r_list;
    r_list["encoded_size"] = vp.offsets.encoded_size;
    r_list["n_groups"] = vp.n_groups;
    r_list["k_reg"] = vp.k_reg;

    r_list["beta_loc"] = vp.beta.loc;
    r_list["beta_info"] = vp.beta.info.mat;
    r_list["beta_diag_min"] = vp.beta.diag_min;

    r_list["mu_loc"] = vp.mu.loc;
    r_list["mu_info"] = vp.mu.info;
    r_list["mu_info_min"] = vp.mu.info_min;

    r_list["tau_alpha"] = vp.tau.alpha;
    r_list["tau_beta"] = vp.tau.beta;
    r_list["tau_alpha_min"] = vp.tau.alpha_min;
    r_list["tau_beta_min"] = vp.tau.beta_min;

    Rcpp::List u(vp.n_groups);
    double u_info_min = 0;
    for (int g = 0; g < vp.n_groups; g++) {
        Rcpp::List this_u;
        this_u["u_loc"] = vp.u[g].loc;
        this_u["u_info"] = vp.u[g].info;
        u[g] = this_u;
        if (g == 0 || u_info_min > vp.u[g].info_min) {
            u_info_min = vp.u[g].info_min;
        }
    }

    r_list["u"] = u;

    // Set the info min to the smallest lower bound.
    r_list["u_info_min"] = u_info_min;

    return r_list;
};


void ConvertParametersFromList(
    VariationalNaturalParameters<double> &vp, Rcpp::List r_list) {

    vp.n_groups = r_list["n_groups"];
    vp.k_reg = r_list["k_reg"];

    vp.beta.loc = Rcpp::as<VectorXd>(r_list["beta_loc"]);
    vp.beta.info.mat = Rcpp::as<MatrixXd>(r_list["beta_info"]);
    vp.beta.diag_min = Rcpp::as<double>(r_list["beta_diag_min"]);

    vp.mu.loc = Rcpp::as<double>(r_list["mu_loc"]);
    vp.mu.info = Rcpp::as<double>(r_list["mu_info"]);
    vp.mu.info_min = Rcpp::as<double>(r_list["mu_info_min"]);

    vp.tau.alpha = Rcpp::as<double>(r_list["tau_alpha"]);
    vp.tau.beta = Rcpp::as<double>(r_list["tau_beta"]);
    vp.tau.alpha_min = Rcpp::as<double>(r_list["tau_alpha_min"]);
    vp.tau.beta_min = Rcpp::as<double>(r_list["tau_beta_min"]);

    Rcpp::List u_list = r_list["u"];
    if (vp.n_groups != u_list.size()) {
        throw std::runtime_error("u size does not match");
    }

    double u_info_min = Rcpp::as<double>(r_list["u_info_min"]);
    for (int g = 0; g < vp.n_groups; g++) {
        Rcpp::List this_u = u_list[g];
        vp.u[g].loc = Rcpp::as<double>(this_u["u_loc"]);
        vp.u[g].info = Rcpp::as<double>(this_u["u_info"]);
        vp.u[g].info_min = u_info_min;
    }
}


VariationalMomentParameters<double>
ConvertMomentParametersFromList(Rcpp::List r_list) {
    int n_groups = r_list["n_groups"];
    int k_reg = r_list["k_reg"];

    VariationalMomentParameters<double> vp(k_reg, n_groups, true);
    ConvertParametersFromList(vp, r_list);
    return vp;
}


VariationalNaturalParameters<double>
ConvertNaturalParametersFromList(Rcpp::List r_list) {
    int n_groups = r_list["n_groups"];
    int k_reg = r_list["k_reg"];

    VariationalNaturalParameters<double> vp(k_reg, n_groups, true);
    ConvertParametersFromList(vp, r_list);
    return vp;
}


Rcpp::List ConvertPriorParametersToList(PriorParameters<double> const &pp) {
    Rcpp::List r_list;
    r_list["encoded_size"] = pp.vec_size;
    r_list["k_reg"] = pp.k_reg;

    r_list["beta_loc"] = pp.beta.loc;
    r_list["beta_info"] = pp.beta.info.mat;

    r_list["mu_loc"] = pp.mu.loc;
    r_list["mu_info"] = pp.mu.info;

    r_list["tau_alpha"] = pp.tau.alpha;
    r_list["tau_beta"] = pp.tau.beta;

    return r_list;
};


PriorParameters<double> ConvertPriorParametersFromList(Rcpp::List r_list) {
    int k_reg = r_list["k_reg"];
    PriorParameters<double> pp(k_reg);

    pp.beta.loc = Rcpp::as<Eigen::VectorXd>(r_list["beta_loc"]);
    pp.beta.info.mat = Rcpp::as<Eigen::MatrixXd>(r_list["beta_info"]);

    pp.mu.loc = Rcpp::as<double>(r_list["mu_loc"]);
    pp.mu.info = Rcpp::as<double>(r_list["mu_info"]);

    pp.tau.alpha = Rcpp::as<double>(r_list["tau_alpha"]);
    pp.tau.beta = Rcpp::as<double>(r_list["tau_beta"]);

    return pp;
}


Data ConvertDataFromR(
    const Eigen::Map<Eigen::VectorXi> r_y,
    const Eigen::Map<Eigen::VectorXi> r_y_g,
    const Eigen::Map<Eigen::MatrixXd> r_x) {

    VectorXi y = r_y;
    VectorXi y_g = r_y_g;
    MatrixXd x = r_x;
    return Data(y, y_g, x);
}


// [[Rcpp::export]]
Rcpp::List SampleData(int n_obs, int k_reg, int n_groups) {
    ProblemInstance prob = SimulateData(n_obs, k_reg, n_groups);
    Rcpp::List prob_list;
    prob_list["y"] = prob.data.y;
    prob_list["y_g"] = prob.data.y_g;
    prob_list["x"] = prob.data.x;

    Rcpp::List vp_nat_list = ConvertParametersToList(prob.vp_nat);
    prob_list["vp_nat"] = vp_nat_list;

    Rcpp::List vp_mom_list = ConvertParametersToList(prob.vp_mom);
    prob_list["vp_mom"] = vp_mom_list;

    PriorParameters<double> pp(k_reg);
    for (int k=0; k < k_reg; k++) {
        pp.beta.info.mat(k, k) = 1;
    }
    pp.mu.info = 1;
    pp.tau.alpha = 1;
    pp.tau.beta = 1;
    Rcpp::List pp_list = ConvertPriorParametersToList(pp);
    prob_list["pp"] = pp_list;

    return prob_list;
}


// [[Rcpp::export]]
Eigen::VectorXd GetMomentParameterVector(const Rcpp::List r_vp, bool unconstrained) {
    VariationalMomentParameters<double> vp =
        ConvertMomentParametersFromList(r_vp);
    vp.unconstrained = unconstrained;
    return GetParameterVector(vp);
}


// [[Rcpp::export]]
Eigen::VectorXd GetNaturalParameterVector(const Rcpp::List r_vp, bool unconstrained) {
    VariationalNaturalParameters<double> vp =
        ConvertNaturalParametersFromList(r_vp);
    vp.unconstrained = unconstrained;
    return GetParameterVector(vp);
}


// [[Rcpp::export]]
Rcpp::List GetMomentParametersFromNaturalParameters(Rcpp::List r_vp_nat) {
    VariationalNaturalParameters<double> vp_nat =
            ConvertNaturalParametersFromList(r_vp_nat);
    VariationalMomentParameters<double> vp_mom(vp_nat);
    return ConvertParametersToList(vp_mom);
}


// [[Rcpp::export]]
Rcpp::List GetMomentParametersFromVector(
    const Rcpp::List r_vp_base,
    const Eigen::Map<Eigen::VectorXd> r_theta,
    bool unconstrained) {

    VectorXd theta = r_theta;
    VariationalMomentParameters<double> vp = ConvertMomentParametersFromList(r_vp_base);
    if (theta.size() != vp.offsets.encoded_size) {
        throw std::runtime_error("Theta is the wrong size");
    }
    vp.unconstrained = unconstrained;
    SetFromVector(theta, vp);
    Rcpp::List vp_list = ConvertParametersToList(vp);
    return vp_list;
}


// [[Rcpp::export]]
Rcpp::List GetNaturalParametersFromVector(
    const Rcpp::List r_vp_base,
    const Eigen::Map<Eigen::VectorXd> r_theta,
    bool unconstrained) {

    Eigen::VectorXd theta = r_theta;
    VariationalNaturalParameters<double> vp =
        ConvertNaturalParametersFromList(r_vp_base);
    vp.unconstrained = unconstrained;
    SetFromVector(theta, vp);
    Rcpp::List vp_list = ConvertParametersToList(vp);
    return vp_list;
}

// [[Rcpp::export]]
Rcpp::List GetEmptyPriorParameters(int k_reg) {
    PriorParameters<double> pp(k_reg);
    return ConvertPriorParametersToList(pp);
}

// [[Rcpp::export]]
Rcpp::List GetPriorParametersFromVector(
    const Rcpp::List r_pp,
    const Eigen::Map<Eigen::VectorXd> r_theta,
    bool unconstrained) {

    Eigen::VectorXd theta = r_theta;
    PriorParameters<double> pp = ConvertPriorParametersFromList(r_pp);
    pp.SetFromVector(theta);
    Rcpp::List pp_list = ConvertPriorParametersToList(pp);
    return pp_list;
}


// [[Rcpp::export]]
Eigen::VectorXd GetPriorParametersVector(
    const Rcpp::List r_pp,
    bool unconstrained) {

    PriorParameters<double> pp = ConvertPriorParametersFromList(r_pp);
    Eigen::VectorXd theta = pp.GetParameterVector();
    return theta;
}


// [[Rcpp::export]]
Rcpp::List GetPriorsAndNaturalParametersFromVector(
    const Rcpp::List r_vp_base,
    const Rcpp::List r_pp_base,
    const Eigen::Map<Eigen::VectorXd> r_theta,
    bool unconstrained) {

    VectorXd theta = r_theta;
    VariationalNaturalParameters<double> vp =
        ConvertNaturalParametersFromList(r_vp_base);
    PriorParameters<double> pp =
        ConvertPriorParametersFromList(r_pp_base);
    if (theta.size() != vp.offsets.encoded_size + pp.vec_size) {
        throw std::runtime_error("Theta is the wrong size");
    }
    vp.unconstrained = unconstrained;
    SetFromVector(theta, vp, pp);
    Rcpp::List vp_list = ConvertParametersToList(vp);
    Rcpp::List pp_list = ConvertPriorParametersToList(pp);
    Rcpp::List ret;
    ret["vp"] = vp_list;
    ret["pp"] = pp_list;

    return ret;
}


// [[Rcpp::export]]
Eigen::VectorXd GetPriorsAndNaturalParametersVector(
    const Rcpp::List r_vp_base,
    const Rcpp::List r_pp_base,
    bool unconstrained) {

    VariationalNaturalParameters<double> vp =
        ConvertNaturalParametersFromList(r_vp_base);
    PriorParameters<double> pp =
        ConvertPriorParametersFromList(r_pp_base);
    vp.unconstrained = unconstrained;
    Eigen::VectorXd theta = GetParameterVector(vp, pp);

    return theta;
}


Rcpp::List ConvertDerivativesToList(Derivatives derivs, ModelOptions opt) {
    Rcpp::List derivs_list;
    derivs_list["val"] = derivs.val;
    if (opt.calculate_gradient) {
        derivs_list["grad"] = derivs.grad;
    }
    if (opt.calculate_hessian) {
        derivs_list["hess"] = derivs.hess;
    }

    return derivs_list;
}


// [[Rcpp::export]]
Rcpp::List GetLogLikDerivatives(
    const Eigen::Map<Eigen::VectorXi> r_y,
    const Eigen::Map<Eigen::VectorXi> r_y_g,
    const Eigen::Map<Eigen::MatrixXd> r_x,
    const Rcpp::List r_vp,
    const Rcpp::List r_opt) {

    ModelOptions opt = ConvertListToOption(r_opt);
    VariationalNaturalParameters<double> vp = ConvertNaturalParametersFromList(r_vp);
    Data data = ConvertDataFromR(r_y, r_y_g, r_x);
    Derivatives derivs = GetLogLikDerivatives(data, vp, opt);
    return ConvertDerivativesToList(derivs, opt);
}


// [[Rcpp::export]]
Rcpp::List GetLogPriorDerivatives(
    const Rcpp::List r_vp,
    const Rcpp::List r_pp,
    const Rcpp::List r_opt) {

    ModelOptions opt = ConvertListToOption(r_opt);
    VariationalNaturalParameters<double> vp = ConvertNaturalParametersFromList(r_vp);
    PriorParameters<double> pp = ConvertPriorParametersFromList(r_pp);
    Derivatives derivs = GetLogPriorDerivatives(vp, pp, opt);
    return ConvertDerivativesToList(derivs, opt);
};


// [[Rcpp::export]]
Rcpp::List GetFullModelLogPriorDerivatives(
    const Rcpp::List r_vp,
    const Rcpp::List r_pp,
    const Rcpp::List r_opt) {

    ModelOptions opt = ConvertListToOption(r_opt);
    VariationalNaturalParameters<double> vp =
        ConvertNaturalParametersFromList(r_vp);
    PriorParameters<double> pp = ConvertPriorParametersFromList(r_pp);
    Derivatives derivs = GetFullModelLogPriorDerivatives(vp, pp, opt);
    return ConvertDerivativesToList(derivs, opt);
};


// [[Rcpp::export]]
Rcpp::List GetLogVariationalDensityDerivatives(
    const Rcpp::List r_obs,
    const Rcpp::List r_vp,
    const Rcpp::List r_opt) {

    ModelOptions opt = ConvertListToOption(r_opt);
    VariationalNaturalParameters<double> vp =
        ConvertNaturalParametersFromList(r_vp);
    VariationalMomentParameters<double> obs =
        ConvertMomentParametersFromList(r_obs);
    Derivatives derivs = GetLogVariationalDensityDerivatives(obs, vp, opt);
    return ConvertDerivativesToList(derivs, opt);
};


// [[Rcpp::export]]
Rcpp::List GetMCMCLogPriorDerivatives(
    const Rcpp::List draw_list,
    const Rcpp::List r_pp,
    const Rcpp::List r_opt) {

    ModelOptions opt = ConvertListToOption(r_opt);
    int n_draws = draw_list.size();
    PriorParameters<double> pp = ConvertPriorParametersFromList(r_pp);
    Rcpp::Rcout << "Got " << n_draws << " draws.\n";
    Rcpp::List log_prior_gradients(n_draws);
    for (int draw = 0; draw < n_draws; draw++) {
        Rcpp::List this_draw_list = draw_list[draw];
        VariationalMomentParameters<double> mp_draw =
            ConvertMomentParametersFromList(this_draw_list);
        Derivatives derivs =
            GetLogPriorDerivativesFromDraw(mp_draw, pp, opt);
        log_prior_gradients[draw] = derivs.grad;
    }
    return log_prior_gradients;
}


// [[Rcpp::export]]
Rcpp::List GetELBODerivatives(
    const Eigen::Map<Eigen::VectorXi> r_y,
    const Eigen::Map<Eigen::VectorXi> r_y_g,
    const Eigen::Map<Eigen::MatrixXd> r_x,
    const Rcpp::List r_vp,
    const Rcpp::List r_pp,
    const Rcpp::List r_opt) {

    ModelOptions opt = ConvertListToOption(r_opt);
    VariationalNaturalParameters<double> vp = ConvertNaturalParametersFromList(r_vp);
    PriorParameters<double> pp = ConvertPriorParametersFromList(r_pp);
    Data data = ConvertDataFromR(r_y, r_y_g, r_x);
    Derivatives derivs = GetELBODerivatives(data, vp, pp, opt);

    return ConvertDerivativesToList(derivs, opt);
}


// [[Rcpp::export]]
Rcpp::List GetEntropyDerivatives(
    const Rcpp::List r_vp,
    const Rcpp::List r_opt) {

    ModelOptions opt = ConvertListToOption(r_opt);
    VariationalNaturalParameters<double> vp = ConvertNaturalParametersFromList(r_vp);
    Derivatives derivs = GetEntropyDerivatives(vp, opt);

    return ConvertDerivativesToList(derivs, opt);
}


// [[Rcpp::export]]
Eigen::SparseMatrix<double> GetCovariance(const Rcpp::List r_vp) {
    VariationalNaturalParameters<double> vp_nat =
            ConvertNaturalParametersFromList(r_vp);
    VariationalMomentParameters<double> vp_mom(vp_nat);
    return GetCovariance(vp_nat, vp_mom.offsets);
}


// [[Rcpp::export]]
Rcpp::List GetMomentJacobian(
    const Rcpp::List r_vp,
    const Rcpp::List r_opt) {

    ModelOptions opt = ConvertListToOption(r_opt);
    VariationalNaturalParameters<double> vp = ConvertNaturalParametersFromList(r_vp);

    Derivatives moment_jac = GetMomentJacobian(vp, opt);

    Rcpp::List result;
    result["moments"] = moment_jac.grad;
    result["jacobian"] = moment_jac.hess;
    return result;
}


// [[Rcpp::export]]
Eigen::SparseMatrix<double> GetSparseLogLikHessian(
    const Eigen::Map<Eigen::VectorXi> r_y,
    const Eigen::Map<Eigen::VectorXi> r_y_g,
    const Eigen::Map<Eigen::MatrixXd> r_x,
    const Rcpp::List r_vp,
    const Rcpp::List r_pp,
    const Rcpp::List r_opt,
    const bool include_prior) {

    ModelOptions opt = ConvertListToOption(r_opt);
    VariationalNaturalParameters<double> vp =
        ConvertNaturalParametersFromList(r_vp);
    PriorParameters<double> pp = ConvertPriorParametersFromList(r_pp);
    Data data = ConvertDataFromR(r_y, r_y_g, r_x);
    std::vector<Triplet> terms = GetSparseLogLikHessian(data, vp, pp, opt, include_prior);

    Eigen::SparseMatrix<double>
        hess(vp.offsets.encoded_size, vp.offsets.encoded_size);
    hess.setFromTriplets(terms.begin(), terms.end());
    hess.makeCompressed();

    return hess;
}


// [[Rcpp::export]]
Eigen::SparseMatrix<double> GetSparseEntropyHessian(
    const Rcpp::List r_vp,
    const Rcpp::List r_opt) {

    ModelOptions opt = ConvertListToOption(r_opt);
    VariationalNaturalParameters<double> vp =
        ConvertNaturalParametersFromList(r_vp);
    std::vector<Triplet> terms = GetSparseEntropyHessian(vp, opt);

    Eigen::SparseMatrix<double>
        hess(vp.offsets.encoded_size, vp.offsets.encoded_size);
    hess.setFromTriplets(terms.begin(), terms.end());
    hess.makeCompressed();

    return hess;
}
