
#ifndef CONVOLUTION_HPP
# define CONVOLUTION_HPP

# include "ConvLayer.hpp"
# include "Matrix.hpp"
# include "math.hpp"

Tensor4D convolve(const Tensor4D &input, const Tensor4D &kernel, int stride = 1, int padding = 0);
Matrix	gaussian_blur(int size, double sigma);

#endif
