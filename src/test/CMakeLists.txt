cmake_minimum_required(VERSION 2.8)
cmake_policy(VERSION 2.8.0)

set(GIT_REPO_LOC $ENV{GIT_REPO_LOC})

# TODO: Unset when done debugging
# set(CMAKE_BUILD_TYPE Debug)

add_definitions(-std=c++11)
add_definitions(-Wno-deprecated)

include_directories(${GIT_REPO_LOC}/math/lib/eigen_3.2.4/)
include_directories(${GIT_REPO_LOC}/googletest/googletest/include)
include_directories(${PROJECT_SOURCE_DIR}/include)

# Only testing requires Stan (for autodiff)
include_directories(${GIT_REPO_LOC}/stan/)
include_directories(${GIT_REPO_LOC}/math/)
include_directories(${EIGEN_LIB})
include_directories(${BOOST_LIB})
include_directories(${GIT_REPO_LOC}/LinearResponseVariationalBayes.cpp/include)
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(logit_glmm_test logit_glmm_test.cpp)
target_link_libraries(logit_glmm_test gtest gtest_main)
target_link_libraries(logit_glmm_test LogitGLMMLRVB)
