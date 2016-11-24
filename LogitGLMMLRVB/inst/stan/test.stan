data {
    real x;
}
parameters {
    real mu;
}
transformed parameters {
    real mu_lpdf_var;
    mu_lpdf_var = 2.0;
}
model {
    target += mu_lpdf_var;
}