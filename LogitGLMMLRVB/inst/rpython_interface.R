
# In order to replace some of read_vb_python.R, try using rPython to convert between parameters
# and vectors using the python machinery.

# https://stackoverflow.com/questions/45319387/using-rpython-import-numply-with-python-3-5
# https://github.com/rstudio/reticulate
install.packages("reticulate")



# I believe this is broken:
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
