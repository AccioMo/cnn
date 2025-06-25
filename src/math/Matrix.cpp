
#include "Matrix.hpp"

Matrix::Matrix( void )
	: _rows(0),
	_cols(0)
{ }

Matrix::Matrix( t_vec array ) : m(array) {
	this->_rows = array.size();
	if (this->_rows == 0)
			throw std::invalid_argument("empty array");
	this->_cols = array[0].size();
	for (int i = 0; i < this->_rows; i++) {
		if (this->_cols != (int)array[i].size())
			throw std::invalid_argument("inconsistent size");
	}
}

Matrix::Matrix( int rows, int cols ) {
	this->_rows = rows;
	this->_cols = cols;
	this->m = t_vec(rows, std::vector<float>(cols));
}

Matrix::Matrix( int rows, int cols, float std_deviation ) {
	this->_rows = rows;
	this->_cols = cols;
	this->m = t_vec(rows, std::vector<float>(cols));
	if (std_deviation != 0) {
		std::random_device					rd;
		std::mt19937						gen(rd());
		std::normal_distribution<>	dis(0.0, std_deviation);
		for ( auto &row : this->m ) {
			for (auto &value : row) {
				value = dis(gen);
			}
		}
	}
}

Matrix::~Matrix() { }


int	Matrix::rows( void ) const {
	return this->_rows;
}

int	Matrix::cols( void ) const {
	return this->_cols;
}

Matrix	Matrix::repeat_cols( int cols ) const {
	Matrix	matrix(this->_rows, cols);
	for (int i = 0; i < matrix.rows(); i++) {
		for (int j = 0; j < matrix.cols(); j++) {
			matrix.m[i][j] = this->m[i][0];
		}
	}
	return matrix;
}

Matrix	Matrix::repeat_rows( int rows ) const {
	Matrix	matrix(rows, this->_cols);
	for (int i = 0; i < matrix.rows(); i++) {
		for (int j = 0; j < matrix.cols(); j++) {
			matrix.m[i][j] = this->m[0][j];
		}
	}
	return matrix;
}

Matrix	Matrix::operator+( const Matrix &to_add ) const {
	Matrix result;
	Matrix	m1;
	if (to_add.rows() != this->rows() || to_add.cols() != this->cols()) {
		if ((to_add.cols() == 1 && to_add.rows() == this->rows())) {
			m1 = to_add.repeat_cols(this->cols());
			result = *this;
		} else if (to_add.rows() == 1 && to_add.cols() == this->cols()) {
			m1 = to_add.repeat_rows(this->rows());
			result = *this;
		} else if ((this->cols() == 1 && this->rows() == to_add.rows())) {
			m1 = this->repeat_cols(to_add.cols());
			result = to_add;
		} else if (this->rows() == 1 && this->cols() == to_add.cols()) {
			m1 = this->repeat_rows(to_add.rows());
			result = to_add;
		} else
			throw std::invalid_argument("addition not possible: " + std::to_string(this->rows()) \
				+ "x" + std::to_string(this->cols()) + " and " + std::to_string(to_add.rows()) \
				+ "x" + std::to_string(to_add.cols()));
	} else {
		result = *this;
		m1 = to_add;
	}
	for (int i = 0; i < m1.rows(); i++) {
		for (int j = 0; j < m1.cols(); j++) {
			result.m[i][j] += m1.m[i][j];
		}
	}
	return (result);
}

Matrix	Matrix::operator*( const float scalar ) const {
	Matrix result(*this);
	for (int i = 0; i < result.rows(); i++) {
		for (int j = 0; j < result.cols(); j++) {
			result.m[i][j] *= scalar;
		}
	}
	return (result);
}

Matrix	Matrix::operator/( const float scalar ) const {
	Matrix result(*this);
	for (int i = 0; i < result.rows(); i++) {
		for (int j = 0; j < result.cols(); j++) {
			result.m[i][j] /= scalar;
		}
	}
	return (result);
}

Matrix	Matrix::operator-( const float scalar ) const {
	Matrix result(*this);
	for (int i = 0; i < result.rows(); i++) {
		for (int j = 0; j < result.cols(); j++) {
			result.m[i][j] -= scalar;
		}
	}
	return (result);
}

Matrix	Matrix::operator+( const float scalar ) const {
	Matrix result(*this);
	for (int i = 0; i < result.rows(); i++) {
		for (int j = 0; j < result.cols(); j++) {
			result.m[i][j] += scalar;
		}
	}
	return (result);
}

Matrix	Matrix::operator-( const Matrix &to_subtract ) const {
	Matrix result;
	Matrix	m1;
	if (to_subtract.rows() != this->rows() || to_subtract.cols() != this->cols()) {
		if ((to_subtract.cols() == 1 && to_subtract.rows() == this->rows())) {
			m1 = to_subtract.repeat_cols(this->cols());
			result = *this;
		} else if (to_subtract.rows() == 1 && to_subtract.cols() == this->cols()) {
			m1 = to_subtract.repeat_rows(this->rows());
			result = *this;
		} else if ((this->cols() == 1 && this->rows() == to_subtract.rows())) {
			m1 = this->repeat_cols(to_subtract.cols());
			result = to_subtract;
		} else if (this->rows() == 1 && this->cols() == to_subtract.cols()) {
			m1 = this->repeat_rows(to_subtract.rows());
			result = to_subtract;
		} else
			throw std::invalid_argument("subtraction not possible: " + std::to_string(this->rows()) \
				+ "x" + std::to_string(this->cols()) + " and " + std::to_string(to_subtract.rows()) \
				+ "x" + std::to_string(to_subtract.cols()));
	} else {
		result = *this;
		m1 = to_subtract;
	}
	for (int i = 0; i < m1.rows(); i++) {
		for (int j = 0; j < m1.cols(); j++) {
			result.m[i][j] -= m1.m[i][j];
		}
	}
	return (result);
}

Matrix	Matrix::dot( const Matrix &mult ) const {
	if (this->cols() != mult.rows()) {
		throw std::invalid_argument("multiplication not possible: " + std::to_string(this->rows()) \
			+ "x" + std::to_string(this->cols()) + " and " + std::to_string(mult.rows()) \
			+ "x" + std::to_string(mult.cols()));
	}
	Matrix result(this->rows(), mult.cols());
	for (int i = 0; i < this->rows(); i++) {
		for (int j = 0; j < mult.cols(); j++) {
			for (int k = 0; k < this->cols(); k++) {
				result.m[i][j] += this->m[i][k] * mult.m[k][j];
			}
		}
	}
	return (result);
}

Matrix	Matrix::operator*( const Matrix &mult ) const {
	return (this->dot(mult));
}

Matrix	Matrix::operator/( const Matrix &divide ) const {
	return (this->hadamard_division(divide));
}

Matrix	Matrix::operator==( const Matrix &m2 ) const {
	if (m2.rows() != this->rows() || m2.cols() != this->cols())
		throw std::invalid_argument("comparison not possible: " + std::to_string(this->rows()) \
			+ "x" + std::to_string(this->cols()) + " and " + std::to_string(m2.rows()) \
			+ "x" + std::to_string(m2.cols()));
	Matrix result(this->rows(), 1);
	for (int i = 0; i < this->rows(); i++) {
		result.m[i][0] = 1;
		for (int j = 0; j < m2.cols(); j++) {
			result.m[i][0] *= this->m[i][j] == m2.m[i][j];
		}
	}
	return (result);
}

Matrix	Matrix::hadamard_product( const Matrix &mult ) const {
	if (mult.rows() != this->rows() || mult.cols() != this->cols())
		throw std::invalid_argument("hadamard multiplication not possible: " + std::to_string(this->rows()) \
			+ "x" + std::to_string(this->cols()) + " and " + std::to_string(mult.rows()) \
			+ "x" + std::to_string(mult.cols()));
	Matrix result(*this);
	for (int i = 0; i < mult.rows(); i++) {
		for (int j = 0; j < mult.cols(); j++) {
			result.m[i][j] *= mult.m[i][j];
		}
	}
	return (result);
}

Matrix	Matrix::hadamard_division( const Matrix &divide ) const {
	if (divide.rows() != this->rows() || divide.cols() != this->cols())
		throw std::invalid_argument("hadamard division not possible: " + std::to_string(this->rows()) \
			+ "x" + std::to_string(this->cols()) + " and " + std::to_string(divide.rows()) \
			+ "x" + std::to_string(divide.cols()));
	Matrix result(*this);
	for (int i = 0; i < divide.rows(); i++) {
		for (int j = 0; j < divide.cols(); j++) {
			result.m[i][j] /= divide.m[i][j];
		}
	}
	return (result);
}

Matrix	Matrix::square( void ) const {
	return (this->hadamard_product(*this));
}

Matrix	Matrix::sum_cols( void ) const {
	Matrix result(1, this->cols());
	for (int i = 0; i < this->rows(); i++) {
		for (int j = 0; j < this->cols(); j++) {
			result.m[0][j] += this->m[i][j];
		}
	}
	return (result);
}

Matrix	Matrix::sum_rows( void ) const {
	Matrix result(this->rows(), 1);
	for (int i = 0; i < this->rows(); i++) {
		for (int j = 0; j < this->cols(); j++) {
			result.m[i][0] += this->m[i][j];
		}
	}
	return (result);
}

Matrix	Matrix::transpose( void ) const {
	Matrix	T(this->cols(), this->rows());
	for (int i = 0; i < this->rows(); i++) {
		for (int j = 0; j < this->cols(); j++) {
			T.m[j][i] = this->m[i][j];
		}
	}
	return (T);
}

Matrix	Matrix::flip( void ) const {
	return (this->transpose().transpose());
}

Matrix	Matrix::sqrt( void ) const {
	Matrix	result(this->rows(), this->cols());
	for (int i = 0; i < this->rows(); i++) {
		for (int j = 0; j < this->cols(); j++) {
			result.m[i][j] = std::sqrt(this->m[i][j]);
		}
	}
	return (result);
}

float	Matrix::mean( void ) const {
	float	sum_value = 0.0;
	for (int i = 0; i < this->rows(); i++) {
		for (int j = 0; j < this->cols(); j++) {
			sum_value += this->m[i][j];
		}
	}
	return (sum_value / (this->rows() * this->cols()));
}

Matrix	Matrix::argmax( void ) const {
	Matrix result(*this);
	for (int i = 0; i < this->rows(); i++) {
		int	max = 0;
		for (int j = 0; j < this->cols(); j++) {
			if (this->m[i][j] > this->m[i][max])
				max = j;
			result.m[i][j] = 0;
		}
		result.m[i][max] = 1;
	}
	return (result);
}

Matrix	Matrix::normalize( float min, float max ) const {
	Matrix	normalized_matrix(*this);
	for (int i = 0; i < this->rows(); i++) {
		for (int j = 0; j < this->cols(); j++) {
			normalized_matrix.m[i][j] = (this->m[i][j] - min) / (max - min);
		}
	}
	return (normalized_matrix);
}

Matrix	Matrix::denormalize( float min, float max ) const {
	Matrix	denormalized_matrix(*this);
	for (int i = 0; i < this->rows(); i++) {
		for (int j = 0; j < this->cols(); j++) {
			denormalized_matrix.m[i][j] = this->m[i][j] * (max - min) + min;
		}
	}
	return (denormalized_matrix);
}

float	Matrix::min() const {
	float	minimum = std::numeric_limits<float>::max();
	for (int i = 0; i < this->rows(); i++) {
		for (int j = 0; j < this->cols(); j++) {
			if (this->m[i][j] < minimum)
			minimum = this->m[i][j];
		}
	}
	return (minimum);
}

float	Matrix::max() const {
	float	maximum = std::numeric_limits<float>::min();
	for (int i = 0; i < this->rows(); i++) {
		for (int j = 0; j < this->cols(); j++) {
			if (this->m[i][j] > maximum)
			maximum = this->m[i][j];
		}
	}
	return (maximum);
}

Matrix	Matrix::row_max() const {
	Matrix	result(this->rows(), 1);
	for (int i = 0; i < this->rows(); i++) {
		float	maximum = std::numeric_limits<float>::min();
		for (int j = 0; j < this->cols(); j++) {
			if (this->m[i][j] > maximum)
				maximum = this->m[i][j];
		}
		result.m[i][0] = maximum;
	}
	return (result);
}


std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
	os << "[";
    for (int i = 0; i < matrix.rows(); ++i) {
		if (i != 0) os << " ";
		os << "[";
        for (int j = 0; j < matrix.cols(); ++j) {
            os << matrix.m[i][j];
			if (j != matrix.cols() - 1) os << ", ";
        }
		os << "]";
        if (i != matrix.rows() - 1) os << ", \n";
    }
	os << "]";
    return os;
}
