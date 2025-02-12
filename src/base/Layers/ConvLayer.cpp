
#include "ConvLayer.hpp"

ConvLayer::ConvLayer( int index, int kernel_size )
	: BaseLayer(kernel_size, kernel_size),
	_index(index)
{
	this->_type = "conv";
}

ConvLayer::~ConvLayer() { }

Matrix	&ConvLayer::feedforward( const Matrix &prev_outputs ) {
	Matrix	&kernel = this->_weights;
	Matrix	convolution_output = convolve(prev_outputs, kernel);
	/* `convolution_output` is the feature map, 
	which i will pass through a max pooling or 
	average pooling to decrease the amount of 
	params the network has to process */
	this->_outputs = max_pooling(convolution_output);
	/* TODO: 
		add average pooling function 
		and simple way to choose one */
	return (this->_outputs);
}

void	ConvLayer::backpropagation( const BaseLayer &next_layer ) {
	this->_errors = next_layer.getDeltas() * next_layer.getWeights().transpose();
	this->_deltas = this->_errors.hadamard_product(ReLU_derivative(this->_outputs));
}

void	ConvLayer::update( const Matrix &inputs, 
							  double learning_rate, 
							  int timestep, 
							  double l2_reg, 
							  double beta1, 
							  double beta2 ) {

	this->_gradient = convolve(inputs, this->_deltas.transpose().transpose());
	Matrix	weight_gradient = this->_gradient + (this->_weights * l2_reg);

	this->_m = this->_m * beta1 + weight_gradient * (1.0 - beta1);
	this->_v = this->_v * beta2 + weight_gradient.square() * (1.0 - beta2);

	Matrix	m_hat = this->_m / (1.0 - std::pow(beta1, timestep));
	Matrix	v_hat = this->_v / (1.0 - std::pow(beta2, timestep));

	this->_weights = this->_weights - (m_hat / v_hat.sqrt() + 1e-8) * learning_rate;

	this->_biases = this->_biases - (this->_deltas.sum_columns() * learning_rate);
}

int  ConvLayer::getIndex( void ) const {
    return (this->_index);
}
