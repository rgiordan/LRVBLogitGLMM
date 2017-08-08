
# In order to replace some of read_vb_python.R, try using rPython to convert between parameters
# and vectors using the python machinery.

# https://stackoverflow.com/questions/45319387/using-rpython-import-numply-with-python-3-5
# https://github.com/rstudio/reticulate
library(reticulate)
library(tidyr)

py_main <- py_run_string(
"
import sys
import os
print(sys.version)
sys.path.append('/home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.py')
sys.path.append('/home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.py/Models')
sys.path.append('/home/rgiordan/Documents/git_repos/autograd')

import VariationalBayes as vb
import LogisticGLMM_lib as logit_glmm
")

np <- import("numpy")


glmm_par <- py_main$logit_glmm$get_glmm_parameters(K=5, NG=10)

rand_par <- array(runif(glmm_par$free_size()))
glmm_par$set_free(rand_par)
glmm_par


glmm_indices <- py_main$logit_glmm$get_glmm_parameters(K=5, NG=10)
glmm_indices$set_vector(array(seq(1, glmm_indices$vector_size())))

glmm_par_list <- glmm_par$dictval()
glmm_par$param_dict$mu$mean$get()

# No
# unnest(tibble(glmm_par_list))

library(purrr)
library(dplyr)

# Doesn't preserve names
flatten(glmm_par_list)
flatten(glmm_par_list, .id="par")

lmap(glmm_par_list, flatten)

# I don't know why, but these don't work.
# flatten_dfr(glmm_par_list)
# flatten_dfc(glmm_par_list)


foo <- list(a=1, b=list(aa=2, bb=3), c=c(10., 11))
flatten_dfc(foo)

foo <- data.frame(a=runif(5), b=runif(5))
bar <- data.frame(b=runif(5), c=runif(5))
bind_rows(foo, bar)

foo <- list(a=runif(4), b=runif(5))
bar <- map(foo, function(x) { tibble(val=x) })
flatten(bar)
bind_rows(bar, .id="id") # This

unpack_parameter <- function(par, level=0) {
  if (is.numeric(par)) {
    if (length(par) == 1) {
      return(tibble(val=par))
    } else {
      return(tibble(val=par, component=1:length(par)))
    }
  } else if (is.list(par)) {
    next_level <- map(par, unpack_parameter, level + 1)
    id_name <- paste("par", level + 1, sep="_")
    bind_rows(next_level, .id=id_name)
  }
}

unpack_parameter(foo)

baz = list(c=runif(2), d=foo)
unpack_parameter(baz)

glmm_par_df <- unpack_parameter(glmm_par_list)
View(glmm_par_df)

#########
# I believe this is broken:

quit()
remove.packages("rPython")
Sys.setenv(RPYTHON_PYTHON_VERSION=3.5)
Sys.getenv("PATH")
install.packages("rPython")


library(rPython)
python.exec("import numpy")


python.exec("import sys")
python.exec("import os")
python.exec("print('hello')")
python.exec("z = sys.path")
python.get("sys.path")
python.get("os.path")
python.exec("import numpy")


python.exec(
"
import sys
import os
print(sys.version)
sys.path.append('/home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.py')
sys.path.append('/home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.py/Models')
sys.path.append('/home/rgiordan/Documents/git_repos/autograd')
")



python.get("sys.path")


# python.exec(
# "
# import numpy as np
# #import VariationalBayes as vb
# #import LogisticGLMM_lib as logit_glmm
# ")



# Pretty slick:
python.exec("
def foo(a, b):
  return a + b

class Bar(object):
  def __init__(self, k):
    self.k = k

  def get(self, a):
    return a + self.k
"
)

python.call("foo", 2, 10)

`%.%` <- function(a, b) { return(paste(a, b, sep="")) }

k <- 6
python.exec("my_bar = Bar(" %.% k %.% ")")
python.method.call("my_bar", "get", 11)

python.exec("my_bar = Bar(" %.% k %.% ")")

k  <- 3
ng <- 100
python.exec(
"
glmm_par = logit_glmm.get_glmm_parameters(
        K=" %.% k %.% ", NG=" %.% ng %.% ", 
        mu_info_min=" %.% 0.0001 %.% ",
        tau_alpha_min=" %.% 0.0001 %.% ",
        tau_beta_min=" %.% 0.0001 %.% ",
        beta_diag_min=" %.% 0.0001 %.% ",
        u_info_min=" %.% 0.0001 %.% ")
")
