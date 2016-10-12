# LRVBLogitGLMM

This is intended as a relatively simple example of how to calculate LRVB estimates
using Stan math. Stay tuned for increasingly detailed documentation.

This package contains a C++ library and an R package.  The R
package is mostly a thin wrapper around functionality implemented in C++.
In order to install it, you'll need to run ```compile_r_package.sh```,
but first you'll have to set some machine-specific variables and
install other packages.  Sorry, this is all kind of home-grown right now.

First, set an environment variable ```GIT_REPO_LOC``` to the location of
this and the other packages below:

```
export GIT_REPO_LOC=/home/username/your/git/directory
cd $GIT_REPO_LOC
git clone https://github.com/rgiordan/LRVBLogitGLMM
```

Next, clone the other required packages.

```
cd $GIT_REPO_LOC
git clone https://github.com/rgiordan/LinearResponseVariationalBayes.cpp
git clone https://github.com/stan-dev/math
git clone https://github.com/stan-dev/stan
```

Follow the directions in LinearResponseVariationalBayes.cpp to build and
install the libraries.  Boost, Eigen, and Stan are header-only and don't
require any compilation.

Next, make sure that ```CMakeLists.txt``` in the root of this repo
is pointing to the appropriate
versions of the Eigen and Boost libraries included in Stan (they
change periodically).  That may mean changing the numbers in these lines:


```
# In $GIT_REPO_LOC/LRVBLogitGLMM/CmakeLists.txt:

set(EIGEN_LIB ${GIT_REPO_LOC}/math/lib/eigen_3.2.8/)
set(BOOST_LIB ${GIT_REPO_LOC}/math/lib/boost_1.60.0/)
```

You'll need the following R packages:

```
# In R:
install.packages(devtools)
install.packages(Rcpp)
```

Finally, as is paradigmatic with cmake, create a directory called ```build```
in this repo for cmake to use:

```
mkdir $GIT_REPO_LOC/LRVBLogitGLMM/build
cd $GIT_REPO_LOC/LRVBLogitGLMM/build
cmake ..
```

At this point, you can try to run ```compile_r_package.sh```.  If everything
is successful, you can try to run the examples in
```LogitGLMMLRVB/inst/```.
