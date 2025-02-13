
# include "convolution.hpp"

Matrix	convolve(const Matrix &input, const Matrix &kernel) {
	if (input.rows() < kernel.rows() || input.cols() < kernel.cols()) {
		std::cerr << "error: kernel is larger than input" << std::endl;
		exit(1);
	} else if (kernel.rows() % 2 == 0 || kernel.cols() % 2 == 0) {
		std::cerr << "error: kernel must have odd dimensions" << std::endl;
		exit(1);
	}
	Matrix	output(input.rows(), input.cols());
	int size = kernel.rows() / 2;
	for (int i = 0; i < output.rows(); i++) {
		for (int j = 0; j < output.cols(); j++) {
			Matrix tmp(kernel.rows(), kernel.cols());
			for (int k = 0; k < tmp.rows(); k++) {
				for (int h = 0; h < tmp.cols(); h++) {
					int ci = i + (k - size);
					if (ci < 0)
						ci = 0;
					else if (ci >= output.rows())
						ci = output.rows() - 1;
					int cj = j + (h - size);
					if (cj < 0)
						cj = 0;
					else if (cj >= output.cols())
						cj = output.cols() - 1;
					tmp.m[k][h] = input.m[ci][cj];
				}
			}

			tmp = tmp.hadamard_product(kernel);

			output.m[i][j] = sum(tmp);
		}
	}
	return (output);
}

Matrix	gaussian_blur(int size, double sigma) {
	Matrix	kernel(size, size);
	int		center = size / 2;
	double	sum = 0;
	/*
		equation for Gaussian function in 2 dimentions:
		G(x, y) => ( 1 / (2*PI*SIGMA^2) ) * exp(-(x^2 + y^2) / 2*SIGMA^2)
		src: https://www.w3.org/Talks/2012/0125-HTML-Tehran/Gaussian.xhtml
	*/
	for (int y = 0; y < size; y++) {
		for (int x = 0; x < size; x++) {
			double	exponent = -((x-center)*(x-center)+(y-center)*(y-center)) / (2.0*sigma*sigma);
			kernel.m[y][x] = (1.0/(2.0*M_PI*sigma*sigma)) * exp(exponent);
			sum += kernel.m[y][x];
		}
	}

	return (kernel / sum);
}
