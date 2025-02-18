
#include "HiddenLayer.hpp"

HiddenLayer::HiddenLayer( int input_size, int output_size )
	: BaseLayer(input_size, output_size)
{
	this->_type = "hidden";
}

HiddenLayer::~HiddenLayer() { }

Matrix	&HiddenLayer::feedforward( const Matrix &prev_outputs ) {
	this->_z = prev_outputs.dot(this->_weight) + this->_bias;
	this->_a = ReLU(this->_z);
	return (this->_a);
}

void	HiddenLayer::backpropagation( const BaseLayer &next_layer ) {
	this->_error = next_layer.getError().dot(next_layer.getWeight().transpose()).hadamard_product(ReLU_derivative(this->_z));
}
