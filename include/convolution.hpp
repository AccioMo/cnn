
#ifndef CONVOLUTION_HPP
# define CONVOLUTION_HPP

# include "Eigen/Dense"
# include "unsupported/Eigen/CXX11/Tensor"

typedef Eigen::Tensor<float, 4>	Tensor4D;
typedef Eigen::Tensor<float, 1>	Tensor1D;

# include "Matrix.hpp"
# include "math.hpp"

Tensor4D	convolve(const Tensor4D &input, const Tensor4D &kernel, int stride, int padding);
Tensor4D	rev_convolve(const Tensor4D &input, const Tensor4D &output, int stride, int padding);
Tensor4D	gradient_convolve(const Tensor4D &input, const Tensor4D &error, int stride, int padding);
Tensor4D	max_pooling(const Tensor4D &input, int pool_size);
Tensor4D	upsample(const Tensor4D &pooled, const Tensor4D &input, int pool_size);

#endif
