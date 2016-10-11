# include "logit_glmm_model_parameters.h"


// Convert index from a "local" vector containing first the global variables
// and then a single local variable to indices into a "global" vector containing
// first the global variables and then all the local variables.
int GlobalIndex(int ind, int g, Offsets offsets) {
    // The end of the global indices and the start of the u indices.
    int global_param_size = offsets.u[0];
    int u_offset = offsets.u[g];
    if (ind < global_param_size) {
        // It is a "global" index: here, mu, beta, or tau.
        return ind;
    } else {
        // It is a u-index.
        return ind - global_param_size + offsets.u[g];
    }
};


# if INSTANTIATE_LOGIT_GLMM_MODEL_PARAMETERS_H
    template class VariationalMomentParameters<double>;
    template class VariationalMomentParameters<var>;
    template class VariationalMomentParameters<fvar>;

    template class VariationalNaturalParameters<double>;
    template class VariationalNaturalParameters<var>;
    template class VariationalNaturalParameters<fvar>;

    template class PriorParameters<double>;
    template class PriorParameters<var>;
    template class PriorParameters<fvar>;
# endif
