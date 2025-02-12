
#ifndef MATH_HPP
# define MATH_HPP

# include "Matrix.hpp"
# include <cmath>
# include <iomanip>
# include <fstream>
# include <cstring>

Matrix	ReLU( const Matrix &input );
Matrix	ReLU_derivative( const Matrix &input );
Matrix	sigmoid( const Matrix &input );
Matrix	sigmoid_derivative( const Matrix &input );
Matrix	exp( const Matrix &input );
Matrix	log( const Matrix &input );
Matrix	abs( const Matrix &input );
double	sum( const Matrix &input );
double	xavier_glorot_init( int fan_in, int fan_out );

#endif
