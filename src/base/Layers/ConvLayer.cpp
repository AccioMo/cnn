
#include "ConvLayer.hpp"

ConvLayer::ConvLayer( int index, int kernel_size )
	: BaseLayer(kernel_size, kernel_size),
	_index(index)
{
	this->_type = "conv";
}

ConvLayer::~ConvLayer() { }

Matrix	&ConvLayer::feedforward( const Matrix &prev_outputs ) {
	Matrix	kernel = this->_weight;
	std::cout << "prev_outputs: " << prev_outputs.rows() << ", " << prev_outputs.cols() << std::endl;
	this->_z = convolve(prev_outputs, kernel);
	std::cout << "this->_z: " << this->_z.rows() << ", " << this->_z.cols() << std::endl;
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

void	ConvLayer::backpropagation( const BaseLayer &next_layer ) {
	/*
		formula for backpropagation in a convolutional:
			δ = (δ^{next} * W.rot(180)) ⊙ f'(z)

		`δ` is the error, `δ^{next}` is the error of the 
		next layer, `W` is the weight of the current layer,
		⊙ is element-wise multiplication, `f'` is the
		derivative of the activation function, and `z` is
		the pre-activation output of the current layer.
	*/
	Matrix	kernel = this->getWeight();
	this->_error = convolve(next_layer.getError(), kernel.flip()).hadamard_product(ReLU_derivative(this->_z));
}

void	ConvLayer::update( const Matrix &inputs, 
							  double learning_rate, 
							  int timestep, 
							  double l2_reg, 
							  double beta1, 
							  double beta2 ) {

	this->_gradient = convolve(inputs.transpose(), this->_error);

	(void)beta1;
	(void)beta2;
	(void)timestep;
	(void)l2_reg;
	/* ------------------------------------------------------------------- 
	Matrix	weight_gradient = this->_gradient + (this->_weight * l2_reg);

	this->_m = this->_m * beta1 + weight_gradient * (1.0 - beta1);
	this->_v = this->_v * beta2 + weight_gradient.square() * (1.0 - beta2);

	Matrix	m_hat = this->_m / (1.0 - std::pow(beta1, timestep));
	Matrix	v_hat = this->_v / (1.0 - std::pow(beta2, timestep));
	this->_gradient = m_hat / v_hat.sqrt() + 1e-8;
	 ------------------------------------------------------------------- */

	this->_weight = this->_weight - this->_gradient * learning_rate;

	this->_bias = this->_bias - (this->_error.sum_cols() * learning_rate);
}
