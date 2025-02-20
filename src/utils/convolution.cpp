
# include "convolution.hpp"

Tensor4D convolve(const Tensor4D &input, const Tensor4D &kernel, int stride, int padding) {
    int batch_size = input.dimension(0);
    int input_height = input.dimension(1);
    int input_width = input.dimension(2);
    int channels = input.dimension(3);

    int kernel_height = kernel.dimension(0);
    int kernel_width = kernel.dimension(1);
    int output_channels = kernel.dimension(3);

	if (kernel_width != kernel_height) {
		std::cerr << "kernel dimensions wrong" << std::endl;
		exit(1);
	
	}

    int output_height = floor(double(input_height + 2*padding - kernel_height) / double(stride)) + 1;
    int output_width = floor(double(input_width + 2*padding - kernel_width) / double(stride)) + 1;
    
    Tensor4D output(batch_size, output_height, output_width, output_channels);

    Tensor4D padded_input;
	std::array<std::pair<int, int>, 4> paddings = {
		std::make_pair(0, 0),
		std::make_pair(0, 0),
		std::make_pair(padding, padding),
		std::make_pair(padding, padding)
	};
	padded_input = input.pad(paddings);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < output_channels; ++oc) {
            for (int oh = 0; oh < output_height; ++oh) {
                for (int ow = 0; ow < output_width; ++ow) {
                    float sum = 0;
                    for (int c = 0; c < channels; ++c) {
                        for (int kh = 0; kh < kernel_height; ++kh) {
                            for (int kw = 0; kw < kernel_width; ++kw) {
                                int ih = oh * stride + kh;
                                int iw = ow * stride + kw;
                                sum += padded_input(b, ih, iw, c) * 
                                      kernel(kh, kw, c, oc);
                            }
                        }
                    }
                    output(b, oh, ow, oc) = sum;
                }
            }
        }
    }

    return (output);
}

Tensor4D rev_convolve(const Tensor4D &error, const Tensor4D &kernel, int stride, int padding) {
    int batch_size = error.dimension(0);
    int error_height = error.dimension(1);
    int error_width = error.dimension(2);
    int output_channels = error.dimension(3);

    int kernel_height = kernel.dimension(0);
    int kernel_width = kernel.dimension(1);
    int input_channels = kernel.dimension(2);

    int input_height = (error_height - 1) * stride + kernel_height - 2 * padding;
    int input_width = (error_width - 1) * stride + kernel_width - 2 * padding;

    Tensor4D propagated_error(batch_size, input_height, input_width, input_channels);
	propagated_error.setZero();

    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < output_channels; ++oc) {
            for (int eh = 0; eh < error_height; ++eh) {
                for (int ew = 0; ew < error_width; ++ew) {
                    for (int kh = 0; kh < kernel_height; ++kh) {
                        for (int kw = 0; kw < kernel_width; ++kw) {
                            int ih = eh * stride + kh - padding;
                            int iw = ew * stride + kw - padding;
                            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                for (int ic = 0; ic < input_channels; ++ic) {
                                    propagated_error(b, ih, iw, ic) += error(b, eh, ew, oc) * kernel(kh, kw, ic, oc);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return (propagated_error);
}

Tensor4D gradient_convolve(const Tensor4D &input, const Tensor4D &error, int stride, int padding) {
    int batch_size = input.dimension(0);
    int input_height = input.dimension(1);
    int input_width = input.dimension(2);
    int input_channels = input.dimension(3);

    int error_height = error.dimension(1);
    int error_width = error.dimension(2);
    int output_channels = error.dimension(3);

    int kernel_height = input_height - (error_height - 1) * stride + 2 * padding;
    int kernel_width = input_width - (error_width - 1) * stride + 2 * padding;

    Tensor4D grad_kernel(kernel_height, kernel_width, input_channels, output_channels);
	grad_kernel.setZero();

    for (int b = 0; b < batch_size; ++b) {
        for (int ic = 0; ic < input_channels; ++ic) {
            for (int oc = 0; oc < output_channels; ++oc) {
                for (int kh = 0; kh < kernel_height; ++kh) {
                    for (int kw = 0; kw < kernel_width; ++kw) {
                        float sum = 0;
                        for (int eh = 0; eh < error_height; ++eh) {
                            for (int ew = 0; ew < error_width; ++ew) {
                                int ih = eh * stride + kh - padding;
                                int iw = ew * stride + kw - padding;
                                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                    sum += input(b, ih, iw, ic) * error(b, eh, ew, oc);
                                }
                            }
                        }
                        grad_kernel(kh, kw, ic, oc) += sum;
                    }
                }
            }
        }
    }

	grad_kernel = grad_kernel / (float)batch_size;

    return (grad_kernel);
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
