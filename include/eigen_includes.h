
# include <Eigen/Dense>
# include <Eigen/Sparse>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::MatrixXi;
using Eigen::VectorXi;

using Eigen::Matrix;
using Eigen::Dynamic;

using Eigen::SparseMatrix;
typedef Eigen::Triplet<double> Triplet; // For populating sparse matrices

template <typename T> using VectorXT = Eigen::Matrix<T, Dynamic, 1>;
template <typename T> using MatrixXT = Eigen::Matrix<T, Dynamic, Dynamic>;
