
# include "convolution.hpp"

Tensor4D convolve(const Tensor4D &input, const Tensor4D &kernel, int stride, int padding) {
    int batch_size = input.dimension(0);
    int channels = input.dimension(1);
    int kernel_height = kernel.dimension(0);
    int kernel_width = kernel.dimension(1);
    int input_height = input.dimension(2);
    int input_width = input.dimension(3);
    int output_channels = kernel.dimension(3);

	if (kernel_width != kernel_height) {
		std::cerr << "kernel fucked" << std::endl;
		exit(1);
	
	}

    Tensor4D output(batch_size, input_width, input_height, output_channels);
    
    int output_height = (input_height + 2*padding - kernel_height) / stride + 1;
    int output_width = (input_width + 2*padding - kernel_width) / stride + 1;
    
    Tensor4D padded_input;
    if (padding > 0) {
        std::array<std::pair<int, int>, 4> paddings = {
            std::make_pair(0, 0),
            std::make_pair(0, 0),
            std::make_pair(padding, padding),
            std::make_pair(padding, padding)
        };
        padded_input = input.pad(paddings);
    } else {
        padded_input = input;
    }
    
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
                                sum += padded_input(b, c, ih, iw) * 
                                      kernel(kh, kw, c, oc);
                            }
                        }
                    }
                    output(b, oc, oh, ow) = sum;
                }
            }
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
