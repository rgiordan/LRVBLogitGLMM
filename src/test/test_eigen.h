# ifndef TEST_EIGEN_H
# define TEST_EIGEN_H

// Tests for Eigen types.
#define EXPECT_VECTOR_EQ(x, y, msg) \
  ASSERT_EQ(x.size(), y.size()); \
  for (int i = 0; i < x.size(); i ++) { \
    EXPECT_DOUBLE_EQ(x(i), y(i)) << " at index " << i << " " << msg; \
  }


#define EXPECT_MATRIX_EQ(x, y, msg) \
  ASSERT_EQ(x.rows(), y.rows()); \
  ASSERT_EQ(x.cols(), y.cols()); \
  for (int i = 0; i < x.rows(); i ++) { \
    for (int j = 0; j < x.rows(); j ++) { \
      EXPECT_DOUBLE_EQ(x(i, j), y(i, j)) << " at index " << i << ", " << j \
      << " " << msg; \
    } \
  }


#define EXPECT_VECTOR_NEAR(x, y, abs_error, msg) \
    ASSERT_EQ(x.size(), y.size()); \
    for (int i = 0; i < x.size(); i ++) { \
        EXPECT_NEAR(x(i), y(i), abs_error) << \
        " at index " << i << " " << msg; \
    }


#define EXPECT_MATRIX_NEAR(x, y, abs_error, msg) \
    ASSERT_EQ(x.rows(), y.rows()); \
    ASSERT_EQ(x.cols(), y.cols()); \
    for (int i = 0; i < x.rows(); i ++) { \
        for (int j = 0; j < x.rows(); j ++) { \
            EXPECT_NEAR(x(i, j), y(i, j), abs_error) << \
            " at index " << i << ", " << j  << " " << msg; \
        } \
    }



// Tests for Eigen types.
#define EXPECT_FLOAT_VECTOR_EQ(x, y, msg) \
  ASSERT_EQ(x.size(), y.size()); \
  for (int i = 0; i < x.size(); i ++) { \
    EXPECT_FLOAT_EQ(x(i), y(i)) << " at index " << i << " " << msg; \
  }


#define EXPECT_FLOAT_MATRIX_EQ(x, y, msg) \
  ASSERT_EQ(x.rows(), y.rows()); \
  ASSERT_EQ(x.cols(), y.cols()); \
  for (int i = 0; i < x.rows(); i ++) { \
    for (int j = 0; j < x.rows(); j ++) { \
      EXPECT_FLOAT_EQ(x(i, j), y(i, j)) << " at index " << i << ", " << j << \
      " " << msg; \
    } \
  }


# endif
