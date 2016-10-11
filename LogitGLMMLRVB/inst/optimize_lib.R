library(LogitGLMMLRVB)
library(LRVBUtils)

# # Minimizes a function with line search starting at x in direction v
# LineSearch <- function(EvalFun, EvalGrad, x, v,
#                        initial_step=1, step_scale=0.8, max_iters=5000,
#                        step_min=0, step_max=Inf,
#                        fn_decrease=1e-14, grad_decrease=0.5,
#                        fn_scale=1, verbose=FALSE) {
#   step_size <- initial_step
#
#   f0 <- EvalFun(x) * fn_scale
#   grad0 <- EvalGrad(x) * fn_scale
#   slope0 <- sum(grad0 * v)
#
#   iter <- 1
#   done <- FALSE
#   while (iter <= max_iters && !done) {
#     x_new <- x + step_size * v
#     new_f <- EvalFun(x_new) * fn_scale
#     new_grad <- EvalGrad(x_new) * fn_scale
#     new_slope <- sum(new_grad * v)
#
#     # Strong Wolfe conditions
#     f_condition <- (new_f <= f0 + fn_decrease * step_size * slope0)
#     grad_condition <-
#       (abs(new_slope) <= grad_decrease * abs(slope0)) || (abs(new_slope) < 1e-7)
#
#     if (verbose) {
#       cat(" iter: ", iter,
#           " step_min: ", step_min,
#           " step_max: ", step_max,
#           " step_size: ", step_size,
#           " f diff: ", f0 - new_f,
#           " grad diff: ", abs(slope0) - abs(new_slope),
#           "\n")
#     }
#
#     if (f_condition && grad_condition) {
#       done <- TRUE
#       if (verbose) cat("Done.\n")
#     } else if (f_condition) {
#       # f decrease but not slope decrease.
#       # Increase step size to get a slope decrease
#       step_min <- step_size
#       if (is.infinite(step_max)) {
#         step_size <- step_size/ step_scale
#       } else {
#         step_size <- step_max - (step_max - step_size) * step_scale
#       }
#       if (verbose) cat("Increasing step size to ", step_size, "\n")
#     } else {
#       # Decrease step size to get a function decrease
#       step_max <- step_size
#       step_size <- step_min + (step_size - step_min) * step_scale
#       if (verbose) cat("Decreasing step size to ", step_size, "\n")
#     }
#
#     if (abs(step_max - step_min) < 1e-8) {
#       iter <- max_iters
#     }
#
#     iter <- iter + 1
#   }
#
#   return(list(x=x_new, f=new_f, grad=new_grad, step_size=step_size, done=done))
# }
#
#
# # Minimizes a function with Newton's method, following negative eigenvalues
# # using linesearch if available.
# NewtonsMethod <- function(EvalFun, EvalGrad, EvalHess, theta_init,
#                           max_iters=20, tol=1e-12, fn_scale=1, verbose=FALSE) {
#   iter <- 1
#   done <- FALSE
#   f0 <- fn_scale * EvalFun(theta_init)
#   theta <- theta_init
#   while(iter <= max_iters && !done) {
#     hess <- fn_scale * EvalHess(theta)
#     hess_ev <- eigen(hess)
#     pos_def <- FALSE
#     step_max <- Inf
#     initial_step <- 1
#     grad <- fn_scale * EvalGrad(theta)
#     if (min(hess_ev$values) < -1e-8) {
#       # Minimizes a function with line search starting at x in direction v
#       if (verbose) cat("\n\n\nSearching along negative eigenvector.  ev = ",
#                        min(hess_ev$values), "\n\n")
#       min_ind <- which.min(hess_ev$values)
#       ev <- hess_ev$vectors[, min_ind]
#       step_direction <- sign(sum(-1 * grad * ev)) * ev
#       initial_step <- 1
#     } else {
#       pos_def <- TRUE
#       step_direction <- as.numeric(-1 * solve(hess, grad))
#       step_max <- 1
#     }
#     ls_result <- LineSearch(EvalFun, EvalGrad, theta, step_direction,
#                             step_scale=0.5, max_iters=5000,
#                             step_max=step_max, initial_step=initial_step,
#                             fn_scale=fn_scale, verbose=FALSE)
#     f1 <- fn_scale * EvalFun(ls_result$x)
#     diff <- f1 - f0
#     theta <- ls_result$x
#     f0 <- f1
#     if (!is.numeric(diff)) {
#       warn("Non-numeric function evaluation.")
#       iter <- max_iters
#     }
#     if (verbose) cat(" iter: ", iter,
#                      " diff: ", diff,
#                      " f: ", f1,
#                      " step: ", ls_result$step_size,
#                      "\n")
#     if (pos_def && abs(diff) < tol) {
#       if (verbose) cat("Done.\n")
#       done <- TRUE
#     }
#     iter <- iter + 1
#   }
#
#   return(list(theta=theta, done=done))
# }
