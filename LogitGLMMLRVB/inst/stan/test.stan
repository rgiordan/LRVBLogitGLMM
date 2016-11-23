data {
    real x;
}
parameters {
    real mu;
}
transformed parameters {
    real mu_lpdf;
    mu_lpdf = 2.0;
}
model {
    target += mu_lpdf;
}