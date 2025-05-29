
#ifndef MATRIX_HPP
# define MATRIX_HPP

# include <iostream>
# include <vector>
# include <random>

typedef std::vector<std::vector<double>> t_vec;

class Matrix {
	
	private:

		int	_rows;
		int	_cols;

	public:

		t_vec	m;

		Matrix( void );
		Matrix( t_vec array );
		Matrix( int rows, int cols );
		Matrix( int rows, int cols, double rand_range );
		~Matrix();

		int	rows( void ) const;
		int	cols( void ) const;

		/* default matrix addition and subtraction. if the matrices 
		are of different sizes it attempts a broadcast */
		Matrix	operator+( const Matrix &to_add ) const;
		Matrix	operator-( const Matrix &to_subtract ) const;

		/* default matrix multiplication. if the matrices 
		are not aligned it applies hadamard product */
		Matrix	operator*( const Matrix &mult ) const;

		/* uses hadamard division */
		Matrix	operator/( const Matrix &divide ) const;

		/* comparison operator overload */
		Matrix	operator==( const Matrix &m2 ) const;

		/* default matrix scalar operations */
		Matrix	operator*( const double scalar ) const;
		Matrix	operator/( const double scalar ) const;
		Matrix	operator-( const double scalar ) const;
		Matrix	operator+( const double scalar ) const;

		/* sums the matrix along its cols (collapsing rows) */
		Matrix	sum_cols( void ) const;

		/* sums the matrix along its rows (collapsing cols) */
		Matrix	sum_rows( void ) const;

		/* matrix transpose */
		Matrix	transpose( void ) const;

		/* repeats cols/rows `n` times. used in broadcast */
		Matrix	repeat_cols( int n ) const;
		Matrix	repeat_rows( int n ) const;

		/* dot product. used in multiplication overload */
		Matrix	dot( const Matrix &mult ) const;

		/* elementwize (hadamard) operations */
		Matrix	hadamard_product( const Matrix &mult ) const;
		Matrix	hadamard_division( const Matrix &divide ) const;

		/* elementwise square */
		Matrix	square( void ) const;

		/* elementwise square root */
		Matrix	sqrt( void ) const;

		Matrix	argmax( void ) const;

		/* calculates the arithmetic mean of the matrix */
		double	mean( void ) const;

		/* normalizes the matrix using min-max scaling */
		Matrix	normalize( double min, double max ) const;

		/* reverse normalizes the matrix using min-max scaling */
		Matrix	denormalize( double min, double max ) const;

		/* returns minimum value in the matrix */
		double	min() const;

		/* returns maximum value in the matrix */
		double	max() const;
		
		/* returns maximum value on each row */
		Matrix	row_max() const;

		/* returns the a flipped version the matrix */
		Matrix	flip( void ) const;
};

std::ostream& operator<<(std::ostream& os, const Matrix& matrix);

#endif
