
# include "convolution.hpp"

#include <iostream>
#include <array>
#include <thread>
#include <vector>

Tensor4D convolve(const Tensor4D &input, const Tensor4D &kernel, int stride, int padding) {
    int batch_size = input.dimension(0);
    int input_height = input.dimension(1);
    int input_width = input.dimension(2);
    int channels = input.dimension(3);

    int kernel_height = kernel.dimension(0);
    int kernel_width = kernel.dimension(1);
    int output_channels = kernel.dimension(3);

	if (kernel.dimension(2) != channels) {
		float f = input(13976581365, 1946, 13847,1); // Error: kernel channels must match input channels
		(void)f;
	}

    int output_height = (input_height + 2*padding - kernel_height) / stride + 1;
    int output_width = (input_width + 2*padding - kernel_width) / stride + 1;
    
    Tensor4D output(batch_size, output_height, output_width, output_channels);

    Tensor4D padded_input;
    std::array<std::pair<int, int>, 4> paddings = {
        std::make_pair(0, 0),
        std::make_pair(padding, padding),
        std::make_pair(padding, padding),
        std::make_pair(0, 0)
    };
    padded_input = input.pad(paddings);

    auto convolve_batch = [&](int start_b, int end_b) {
        for (int b = start_b; b < end_b; ++b) {
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
    };

    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    int batch_per_thread = batch_size / num_threads;
    for (int i = 0; i < num_threads; ++i) {
        int start_b = i * batch_per_thread;
        int end_b = (i == num_threads - 1) ? batch_size : start_b + batch_per_thread;
        threads.emplace_back(convolve_batch, start_b, end_b);
    }

    for (auto &t : threads) {
        t.join();
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

    auto convolve_batch = [&](int start_b, int end_b) {
        for (int b = start_b; b < end_b; ++b) {
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
	};

	int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    int batch_per_thread = batch_size / num_threads;
    for (int i = 0; i < num_threads; ++i) {
        int start_b = i * batch_per_thread;
        int end_b = (i == num_threads - 1) ? batch_size : start_b + batch_per_thread;
        threads.emplace_back(convolve_batch, start_b, end_b);
    }

    for (auto &t : threads) {
        t.join();
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

    int kernel_height = (input_height + 2 * padding - (error_height - 1) * stride);
    int kernel_width = (input_width + 2 * padding - (error_width - 1) * stride);

    Tensor4D grad_kernel(kernel_height, kernel_width, input_channels, output_channels);
    grad_kernel.setZero();

    auto convolve_batch = [&](int start_b, int end_b) {
        Tensor4D local_grad(kernel_height, kernel_width, input_channels, output_channels);
        local_grad.setZero();

        for (int b = start_b; b < end_b; ++b) {
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
                            local_grad(kh, kw, ic, oc) += sum;
                        }
                    }
                }
            }
        }

        grad_kernel += local_grad;
    };

    int num_threads = std::min(batch_size, (int)std::thread::hardware_concurrency());
    std::vector<std::thread> threads;
    int batch_per_thread = batch_size / num_threads;
    
    for (int i = 0; i < num_threads; ++i) {
        int start_b = i * batch_per_thread;
        int end_b = (i == num_threads - 1) ? batch_size : start_b + batch_per_thread;
        threads.emplace_back(convolve_batch, start_b, end_b);
    }

    for (auto &t : threads) {
        t.join();
    }

    grad_kernel = grad_kernel / (float)batch_size;
    return (grad_kernel);
}


Tensor4D	max_pooling(const Tensor4D &input, int pool_size) {
	int stride_height = 2;
	int stride_width = 2;

    int batchSize = input.dimension(0);
    int inputHeight = input.dimension(1);
    int inputWidth = input.dimension(2);
    int channels = input.dimension(3);

    int outputHeight = (inputHeight - pool_size) / stride_height + 1;
    int outputWidth = (inputWidth - pool_size) / stride_width + 1;

    Tensor4D	output(batchSize, outputHeight, outputWidth, channels);

    for (int h = 0; h < outputHeight; ++h) {
        for (int w = 0; w < outputWidth; ++w) {
            int hStart = h * stride_height;
            int wStart = w * stride_width;

            Eigen::DSizes<Eigen::Index, 4> start{0, hStart, wStart, 0};
			Eigen::DSizes<Eigen::Index, 4> size{batchSize, pool_size, pool_size, channels};

			Tensor4D window = input.slice(start, size).eval();

            Eigen::Tensor<float, 2> maxValues = window.maximum(Eigen::array<int, 2>{1, 2});

            output.slice(
                Eigen::array<int, 4>{0, h, w, 0},
                Eigen::array<int, 4>{batchSize, 1, 1, channels}
            ) = maxValues.reshape(Eigen::array<int, 4>{batchSize, 1, 1, channels});
        }
    }

    return (output);
}

Tensor4D	upsample(const Tensor4D &pooled, const Tensor4D &input, int pool_size) {
	int stride = 2;

    Tensor4D	upsampled = Tensor4D(input.dimensions());
	upsampled.setZero();
    for (int n = 0; n < input.dimension(0); ++n) {
        for (int c = 0; c < input.dimension(3); ++c) {
            for (int h = 0; h < pooled.dimension(1); ++h) {
                for (int w = 0; w < pooled.dimension(2); ++w) {
                    
                    int h_start = h * stride;
                    int w_start = w * stride;
                    int h_end = fmin(h_start + pool_size, input.dimension(1));
                    int w_end = fmin(w_start + pool_size, input.dimension(2));

                    float max_val = -std::numeric_limits<float>::infinity();
                    int max_h = h_start, max_w = w_start;

                    for (int i = h_start; i < h_end; ++i) {
                        for (int j = w_start; j < w_end; ++j) {
                            if (input(n, i, j, c) > max_val) {
                                max_val = input(n, i, j, c);
                                max_h = i;
                                max_w = j;
                            }
                        }
                    }

                    upsampled(n, max_h, max_w, c) += pooled(n, h, w, c);
                }
            }
        }
    }
    return (upsampled);
}
