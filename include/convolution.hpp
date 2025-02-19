
#ifndef CONVOLUTION_HPP
# define CONVOLUTION_HPP

# include "Eigen/Dense"
# include "unsupported/Eigen/CXX11/Tensor"

typedef Eigen::Tensor<float, 4>	Tensor4D;

# include "Matrix.hpp"
# include "math.hpp"

Tensor4D convolve(const Tensor4D &input, const Tensor4D &kernel, int stride, int padding);
Tensor4D rev_convolve(const Tensor4D &input, const Tensor4D &output, int stride, int padding);
Matrix	gaussian_blur(int size, double sigma);

#endif
