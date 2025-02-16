
#include "ConvLayer.hpp"
#include "convolution.hpp"

ConvLayer::ConvLayer( int kernel_size, int input_size, int output_size ) {
	/* input_size is the number of channels.
	output_size is the number of filter outputs */
	this->_kernel = Tensor4D(kernel_size, kernel_size, input_size, output_size);
	for (int i = 0; i < kernel_size; i++) {
		for (int j = 0; j < kernel_size; j++) {
			for (int k = 0; k < input_size; k++) {
				for (int h = 0; h < output_size; h++) {
					this->_kernel(i, j, k, h) = xavier_glorot_init(kernel_size * kernel_size * input_size, output_size);
				}
			}
		}
	}
	this->_bias = Tensor4D(1, 1, 1, output_size);
	for (int i = 0; i < kernel_size; i++) {
		for (int j = 0; j < kernel_size; j++) {
			for (int k = 0; k < input_size; k++) {
				for (int h = 0; h < output_size; h++) {
					this->_kernel(i, j, k, h) = 0.1;
				}
			}
		}
	}
	this->_m = Tensor4D(kernel_size, kernel_size, input_size, output_size);
	this->_v = Tensor4D(kernel_size, kernel_size, input_size, output_size);
}

ConvLayer::~ConvLayer() { }

Tensor4D	&ConvLayer::feedforward( const Tensor4D &prev_outputs ) {
	this->_z = convolve(prev_outputs, this->_kernel);
	/* `_z` is the convoluted output, feature map, 
	which i will pass through a max pooling or 
	average pooling to decrease the amount of 
	params the network has to process */

	this->_a = ReLU(this->_z);
	/* TODO: 
		this->_a = pooling(this->_a, "max");
		add average pooling function 
		and simple way to choose one */
	return (this->_a);
}

void	ConvLayer::backpropagation( const Tensor4D &next_error ) {
	/*
		formula for backpropagation in a convolutional:
			δ = (δ^{next} * W.rot(180)) ⊙ f'(z)

		`δ` is the error, `δ^{next}` is the error of the 
		next layer, `W` is the kernel of the current layer,
		⊙ is element-wise multiplication, `f'` is the
		derivative of the activation function, and `z` is
		the pre-activation output of the current layer.
	*/
	this->_error = convolve(next_error, flip(this->_kernel)) * ReLU_derivative(this->_z);
}

void	ConvLayer::update( const Tensor4D &inputs, 
							  double learning_rate, 
							  int timestep, 
							  double l2_reg, 
							  double beta1, 
							  double beta2 ) {

	this->_gradient = convolve(transpose(inputs), this->_error);

	(void)beta1;
	(void)beta2;
	(void)timestep;
	(void)l2_reg;
	/* ------------------------------------------------------------------- 
	Tensor4D	kernel_gradient = this->_gradient + (this->_kernel * l2_reg);

	this->_m = this->_m * beta1 + kernel_gradient * (1.0 - beta1);
	this->_v = this->_v * beta2 + kernel_gradient.square() * (1.0 - beta2);

	Tensor4D	m_hat = this->_m / (1.0 - std::pow(beta1, timestep));
	Tensor4D	v_hat = this->_v / (1.0 - std::pow(beta2, timestep));
	this->_gradient = m_hat / v_hat.sqrt() + 1e-8;
	 ------------------------------------------------------------------- */

	this->_kernel = this->_kernel - (this->_gradient * (float)learning_rate);

	this->_bias = this->_bias - (this->_error * (float)learning_rate);
}

Tensor4D	ConvLayer::getKernel( void ) const {
	return (this->_kernel);
}

Tensor4D	ConvLayer::getBias( void ) const {
	return (this->_bias);
}

Tensor4D	ConvLayer::getOutput( void ) const {
	return (this->_a);
}

Tensor4D	ConvLayer::getError( void ) const {
	return (this->_error);
}
