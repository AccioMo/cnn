
#ifndef MATH_HPP
# define MATH_HPP

# include "convolution.hpp"
# include <cmath>
# include <iomanip>
# include <fstream>
# include <cstring>

Tensor4D	ReLU(const Tensor4D &input, float alpha = 0);
Tensor4D	ReLU_derivative(const Tensor4D &input, float alpha = 0);
Tensor4D	flip(const Tensor4D& tensor);
Tensor4D	transpose(const Tensor4D& tensor);
Tensor4D	unflatten(const Matrix &flat, int d1, int d2, int d3, int d4);
Tensor4D	normalize(const Tensor4D &tensor, float min, float max);
Matrix		flatten(const Tensor4D& tensor);
Matrix		ReLU( const Matrix &input );
Matrix		ReLU_derivative( const Matrix &input );
Matrix		sigmoid( const Matrix &input );
Matrix		sigmoid_derivative( const Matrix &input );
Matrix		exp( const Matrix &input );
Matrix		log( const Matrix &input );
Matrix		abs( const Matrix &input );
float		sum( const Matrix &input );
float		xavier_glorot_init( int fan_in, int fan_out );
float		he_init(int fan_in);

#endif
