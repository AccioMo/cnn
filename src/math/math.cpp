
#include "math.hpp"

Matrix	ReLU( const Matrix &input ) {
	Matrix	output = input;
	for (auto &row : output.m) {
		for (auto &element : row) {
			element = fmax(0, element);
		}
	}
	return (output);
}

Matrix	ReLU_derivative( const Matrix &input ) {
	Matrix	output = input;
	for (auto &row : output.m) {
		for (auto &element : row) {
			element = element > 0;
		}
	}
	return (output);
}

Matrix	sigmoid( const Matrix &input ) {
	Matrix	output = input;
	for (auto &row : output.m) {
		for (auto &element : row) {
			element = 1.0 / (1.0 + std::exp(-element));
		}
	}
	return (output);
}

Matrix	exp( const Matrix &input ) {
	Matrix	output = input;
	for (auto &row : output.m) {
		for (auto &element : row) {
			element = std::exp(element);
		}
	}
	return (output);
}

Matrix	log( const Matrix &input ) {
	Matrix	output = input;
	for (auto &row : output.m) {
		for (auto &element : row) {
			element = std::log(element);
		}
	}
	return (output);
}

Matrix	abs( const Matrix &input ) {
	Matrix	output(input);
	for (int i = 0; i < input.rows(); i++) {
		for (int j = 0; j < input.cols(); j++) {
			output.m[i][j] = std::abs(input.m[i][j]);
		}
	}
	return (output);
}

double	sum( const Matrix &input ) {
	double	output = 0;
	for (auto &row : input.m) {
		for (auto &element : row) {
			output += element;
		}
	}
	return (output);
}

Matrix	sigmoid_derivative( const Matrix &input ) {
	Matrix	output = input;
	for (auto &row : output.m) {
		for (auto &element : row) {
			element = element * (1.0 - element);
		}
	}
	return (output);
}

double	xavier_glorot_init(int fan_in, int fan_out) {
	return (4.0 * sqrt(6.0 / (double)(fan_in + fan_out)));
}
