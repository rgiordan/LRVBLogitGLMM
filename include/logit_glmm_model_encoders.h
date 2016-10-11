// This file is separate for organization only.  It is meant to
// be includd in logit_glmm_model_parameters.h.

int GlobalIndex(int ind, int g, Offsets offsets);

template <typename T, template<typename> class VPType>
VectorXT<T> GetParameterVector(VPType<T> vp) {
  VectorXT<T> theta(vp.offsets.encoded_size);

  theta.segment(vp.offsets.beta, vp.beta.encoded_size) =
    vp.beta.encode_vector(vp.unconstrained);

  theta.segment(vp.offsets.mu, vp.mu.encoded_size) =
    vp.mu.encode_vector(vp.unconstrained);

  theta.segment(vp.offsets.tau, vp.tau.encoded_size) =
    vp.tau.encode_vector(vp.unconstrained);

  for (int g = 0; g < vp.u.size(); g++) {
    theta.segment(vp.offsets.u[g], vp.u[g].encoded_size) =
        vp.u[g].encode_vector(vp.unconstrained);
  }

  return theta;
}


template <typename T, template<typename> class VPType>
void SetFromVector(VectorXT<T> const &theta, VPType<T> &vp) {
  if (theta.size() != vp.offsets.encoded_size) {
      std::ostringstream error_msg;
      error_msg << "SetFromVector vp and pp: Vector is wrong size. "
          << "Expected " << vp.offsets.encoded_size <<
          ", got " << theta.size() << "\n";
      throw std::runtime_error(error_msg.str());
  }

  VectorXT<T> theta_sub;

  theta_sub = theta.segment(vp.offsets.beta, vp.beta.encoded_size);
  vp.beta.decode_vector(theta_sub, vp.unconstrained);

  theta_sub = theta.segment(vp.offsets.mu, vp.mu.encoded_size);
  vp.mu.decode_vector(theta_sub, vp.unconstrained);

  theta_sub = theta.segment(vp.offsets.tau, vp.tau.encoded_size);
  vp.tau.decode_vector(theta_sub, vp.unconstrained);

  for (int g=0; g < vp.u.size(); g++) {
    theta_sub = theta.segment(vp.offsets.u[g], vp.u[g].encoded_size);
    vp.u[g].decode_vector(theta_sub, vp.unconstrained);
  }
}


// Get a parameter vector for a single group.
template <typename T, template<typename> class VPType>
VectorXT<T> GetGroupParameterVector(VPType<T> vp, int const g) {

    if (g >= vp.n_groups || g < 0) {
        throw std::runtime_error("g out of bounds.");
    }
    int vec_size =
        vp.beta.encoded_size +
        vp.mu.encoded_size +
        vp.tau.encoded_size +
        vp.u[g].encoded_size;

    VectorXT<T> theta(vec_size);

    int ind = 0;
    theta.segment(ind, vp.beta.encoded_size) =
        vp.beta.encode_vector(vp.unconstrained);
    ind += vp.beta.encoded_size;

    theta.segment(ind, vp.mu.encoded_size) =
        vp.mu.encode_vector(vp.unconstrained);
    ind += vp.mu.encoded_size;

    theta.segment(ind, vp.tau.encoded_size) =
        vp.tau.encode_vector(vp.unconstrained);
    ind += vp.tau.encoded_size;

    theta.segment(ind, vp.u[g].encoded_size) =
        vp.u[g].encode_vector(vp.unconstrained);

    return theta;
}


template <typename T, template<typename> class VPType>
void SetFromGroupVector(VectorXT<T> const &theta, VPType<T> &vp, int const g) {

    int vec_size =
        vp.beta.encoded_size +
        vp.mu.encoded_size +
        vp.tau.encoded_size +
        vp.u[g].encoded_size;

    if (theta.size() != vec_size) {
        std::ostringstream error_msg;
        error_msg << "SetFromGroupVector: Vector is wrong size. "
            << "Expected " << vec_size <<
            ", got " << theta.size() << "\n";
        throw std::runtime_error(error_msg.str());
    }

    VectorXT<T> theta_sub;
    int ind = 0;

    theta_sub = theta.segment(ind, vp.beta.encoded_size);
    vp.beta.decode_vector(theta_sub, vp.unconstrained);
    ind += vp.beta.encoded_size;

    theta_sub = theta.segment(ind, vp.mu.encoded_size);
    vp.mu.decode_vector(theta_sub, vp.unconstrained);
    ind += vp.mu.encoded_size;

    theta_sub = theta.segment(ind, vp.tau.encoded_size);
    vp.tau.decode_vector(theta_sub, vp.unconstrained);
    ind += vp.tau.encoded_size;

    theta_sub = theta.segment(ind, vp.u[g].encoded_size);
    vp.u[g].decode_vector(theta_sub, vp.unconstrained);
}


// Get a parameter vector for a single group.
template <typename T, template<typename> class VPType>
VectorXT<T> GetGlobalParameterVector(VPType<T> vp) {

    int vec_size =
        vp.beta.encoded_size +
        vp.mu.encoded_size +
        vp.tau.encoded_size;

    VectorXT<T> theta(vec_size);

    int ind = 0;
    theta.segment(ind, vp.beta.encoded_size) =
        vp.beta.encode_vector(vp.unconstrained);
    ind += vp.beta.encoded_size;

    theta.segment(ind, vp.mu.encoded_size) =
        vp.mu.encode_vector(vp.unconstrained);
    ind += vp.mu.encoded_size;

    theta.segment(ind, vp.tau.encoded_size) =
        vp.tau.encode_vector(vp.unconstrained);
    ind += vp.tau.encoded_size;

    return theta;
}


template <typename T, template<typename> class VPType>
void SetFromGlobalVector(VectorXT<T> const &theta, VPType<T> &vp) {

    int vec_size =
        vp.beta.encoded_size +
        vp.mu.encoded_size +
        vp.tau.encoded_size;

    if (theta.size() != vec_size) {
        std::ostringstream error_msg;
        error_msg << "SetFromGlobalVector: Vector is wrong size. "
            << "Expected " << vec_size <<
            ", got " << theta.size() << "\n";
        throw std::runtime_error(error_msg.str());
    }

    VectorXT<T> theta_sub;
    int ind = 0;

    theta_sub = theta.segment(ind, vp.beta.encoded_size);
    vp.beta.decode_vector(theta_sub, vp.unconstrained);
    ind += vp.beta.encoded_size;

    theta_sub = theta.segment(ind, vp.mu.encoded_size);
    vp.mu.decode_vector(theta_sub, vp.unconstrained);
    ind += vp.mu.encoded_size;

    theta_sub = theta.segment(ind, vp.tau.encoded_size);
    vp.tau.decode_vector(theta_sub, vp.unconstrained);
    ind += vp.tau.encoded_size;
}


/////////////////////////////////////////////////////
// Both priors and parameters in a single vector.

template <typename T>
VectorXT<T> GetParameterVector(VariationalNaturalParameters<T> &vp,
                               PriorParameters<T> &pp) {

    int encoded_size = vp.offsets.encoded_size + pp.vec_size;
    VectorXT<T> theta(encoded_size);
    theta.segment(0, vp.offsets.encoded_size) = GetParameterVector(vp);
    theta.segment(vp.offsets.encoded_size, pp.vec_size) =
        pp.GetParameterVector();

    return theta;
}


template <typename T>
void SetFromVector(VectorXT<T> const &theta,
                   VariationalNaturalParameters<T> &vp,
                   PriorParameters<T> &pp) {

    int encoded_size = vp.offsets.encoded_size + pp.vec_size;
    if (theta.size() != encoded_size) {
        std::ostringstream error_msg;
        error_msg
            << "SetFromVector vp and pp: Vector is wrong size. "
            << "Expected " << encoded_size << ", got " << theta.size()
            << " pp.vec_size = " << pp.vec_size
            << " vp.offsets.encoded_size = " << vp.offsets.encoded_size << "\n";
        throw std::runtime_error(error_msg.str());
    }

    VectorXT<T> theta_sub;

    theta_sub = theta.segment(0, vp.offsets.encoded_size);
    SetFromVector(theta_sub, vp);

    theta_sub = theta.segment(vp.offsets.encoded_size, pp.vec_size);
    pp.SetFromVector(theta_sub);
}
