add_definitions(-std=c++11 -fPIC)
add_definitions(-Wno-deprecated)

add_subdirectory(test)

include_directories(${GIT_REPO_LOC}/stan/)
include_directories(${GIT_REPO_LOC}/math/)
include_directories(${EIGEN_LIB})
include_directories(${BOOST_LIB})
include_directories(${GIT_REPO_LOC}/LinearResponseVariationalBayes.cpp/include)
include_directories(${PROJECT_SOURCE_DIR}/include)

set(CC_SOURCES logit_glmm_model.cpp logit_glmm_model_parameters.cpp)
add_library(LogitGLMMLRVB SHARED ${CC_SOURCES})
target_link_libraries(LogitGLMMLRVB LinearResponseVariationalBayes)

# Run `sudo make install` to install.
install(TARGETS LogitGLMMLRVB DESTINATION /usr/lib)
