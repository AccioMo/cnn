
#include "math.hpp"

Tensor4D	ReLU(const Tensor4D &input, double alpha) {
    Tensor4D output(input.dimensions());
    
    int total_size = input.size();
    
    Eigen::Map<const Eigen::VectorXf> input_flat(input.data(), total_size);
    Eigen::Map<Eigen::VectorXf> output_flat(output.data(), total_size);
    
    if (alpha == 0) {
        output_flat = input_flat.cwiseMax(0);
    } else {
        output_flat = input_flat.cwiseMax(alpha * input_flat);
    }
    
    return (output);
}

Tensor4D	ReLU_derivative(const Tensor4D &input, double alpha) {
    Tensor4D output(input.dimensions());
    
    int total_size = input.size();
    
    Eigen::Map<const Eigen::VectorXf> input_flat(input.data(), total_size);
    Eigen::Map<Eigen::VectorXf> output_flat(output.data(), total_size);
    
    output_flat = (input_flat.array() > 1e-7).select(Eigen::VectorXf::Constant(total_size, 1.0f),
                                              Eigen::VectorXf::Constant(total_size, alpha));
    
    return (output);
}

Tensor4D	flip(const Tensor4D &tensor) {
    Eigen::array<bool, 4> reverse = {true, true, false, false};
    tensor.reverse(reverse);

    return tensor.reverse(reverse);
}

Tensor4D	transpose(const Tensor4D &tensor) {
    Eigen::array<int, 4> shuffle = {0, 1, 3, 2};
    
    return tensor.shuffle(shuffle);
}

Matrix	flatten(const Tensor4D &tensor) {
	int batch_size = tensor.dimension(0);
	int rows = tensor.dimension(1);
	int cols = tensor.dimension(2);
	int channels = tensor.dimension(3);

	Matrix	flat(batch_size, rows * cols * channels);
	for (int i = 0; i < batch_size; i++) {
		for (int j = 0; j < rows; j++) {
			for (int k = 0; k < cols; k++) {
				for (int h = 0; h < channels; h++) {
					flat.m[i][j*cols*channels + k*channels + h] = tensor(i, j, k, h);
				}
			}
		}
	}
	return (flat);
}

Tensor4D	unflatten(const Matrix &flat, int d1, int d2, int d3, int d4) {
	Tensor4D	tensor(d1, d2, d3, d4);
	for (int i = 0; i < d1; i++) {
		for (int j = 0; j < d2; j++) {
			for (int k = 0; k < d3; k++) {
				for (int h = 0; h < d4; h++) {
					tensor(i, j, k, h) = flat.m[i][j*d3*d4 + k*d4 + h];
				}
			}
		}
	}
	return (tensor);
}

Tensor4D	normalize(const Tensor4D &tensor, double min, double max) {
	Tensor4D output(tensor.dimensions());
	
	int total_size = tensor.size();
	
	Eigen::Map<const Eigen::VectorXf> tensor_flat(tensor.data(), total_size);
	Eigen::Map<Eigen::VectorXf> output_flat(output.data(), total_size);
	
	output_flat = (tensor_flat.array() - min) / (max - min);
	
	return (output);
}
