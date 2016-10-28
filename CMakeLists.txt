cmake_minimum_required(VERSION 2.8)
cmake_policy(VERSION 2.8.0)
project(LogitGLMMLRVB)
set(CMAKE_BUILD_TYPE Release)

set(GIT_REPO_LOC $ENV{GIT_REPO_LOC})
set(EIGEN_LIB ${GIT_REPO_LOC}/math/lib/eigen_3.2.9/)
set(BOOST_LIB ${GIT_REPO_LOC}/math/lib/boost_1.60.0/)

add_subdirectory(src)
