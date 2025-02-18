
# include "convolution.hpp"

Tensor4D convolve(const Tensor4D &input, const Tensor4D &kernel, int stride, int padding) {
	std::cout << "input: ";
	std::cout << input.dimensions() << std::endl;
	std::cout << "kernel: ";
	std::cout << kernel.dimensions() << std::endl;
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

    padding = int(kernel_width / 2);
    int output_height = (input_height + 2*padding - kernel_height) / stride + 1;
    int output_width = (input_width + 2*padding - kernel_width) / stride + 1;
    
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

    Tensor4D rotated_kernel(kernel_height, kernel_width, input_channels, output_channels);
    for (int kh = 0; kh < kernel_height; ++kh) {
        for (int kw = 0; kw < kernel_width; ++kw) {
            for (int ic = 0; ic < input_channels; ++ic) {
                for (int oc = 0; oc < output_channels; ++oc) {
                    rotated_kernel(kh, kw, ic, oc) = kernel(kernel_height - 1 - kh, kernel_width - 1 - kw, ic, oc);
                }
            }
        }
    }

    for (int b = 0; b < batch_size; ++b) {
        for (int ic = 0; ic < input_channels; ++ic) {
            for (int ih = 0; ih < input_height; ++ih) {
                for (int iw = 0; iw < input_width; ++iw) {
                    float sum = 0;
                    for (int oc = 0; oc < output_channels; ++oc) {
                        for (int kh = 0; kh < kernel_height; ++kh) {
                            for (int kw = 0; kw < kernel_width; ++kw) {
                                int eh = (ih - kh + padding) / stride;
                                int ew = (iw - kw + padding) / stride;
                                if (eh >= 0 && eh < error_height && ew >= 0 && ew < error_width && (ih - kh + padding) % stride == 0 && (iw - kw + padding) % stride == 0) {
                                    sum += error(b, eh, ew, oc) * rotated_kernel(kh, kw, ic, oc);
                                }
                            }
                        }
                    }
                    propagated_error(b, ih, iw, ic) = sum;
                }
            }
        }
    }

    return propagated_error;
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
