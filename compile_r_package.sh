#!/bin/bash

echo -e "\n\n\n"
echo $(date)
pushd .
BUILD_LOC=$GIT_REPO_LOC"/LRVBLogitGLMM/build"
R_PACKAGE_LOC=$GIT_REPO_LOC"/LRVBLogitGLMM/LogitGLMMLRVB"
cd $BUILD_LOC
sudo make install
if [ $? -ne 0 ]; then
   echo "Exiting."
   popd
   exit 1
fi
cd $R_PACKAGE_LOC
echo 'library(devtools); library(Rcpp); compileAttributes("'$R_PACKAGE_LOC'"); install_local("'$R_PACKAGE_LOC'", force=TRUE)' | R --vanilla
popd

echo $(date)
echo -e "\n\n\n"
