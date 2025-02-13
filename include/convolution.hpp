
#ifndef CONVOLUTION_HPP
# define CONVOLUTION_HPP

# include "Matrix.hpp"
# include "math.hpp"

Matrix	convolve(const Matrix &input, const Matrix &kernel);
Matrix	gaussian_blur(int size, double sigma);

#endif
