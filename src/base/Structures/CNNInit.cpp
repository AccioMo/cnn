
#include "CNN.hpp"

CNN::CNN( std::vector<int> convolutional_layers,
		std::vector<int> connected_layers, 
		double learning_rate,
		double l2_lambda,
		double beta1,
		double beta2 )
	: _size(convolutional_layers.size() + connected_layers.size()),
	_learning_rate(learning_rate),
	_l2_lambda(l2_lambda),
	_beta1(beta1),
	_beta2(beta2) {
	if (this->_size < 2)
		return ;
}

CNN::CNN( std::vector<ConvLayer> init_conv_layers,
		std::vector<HiddenLayer> init_hidden_layers,
		OutputLayer init_output_layer,
		double learning_rate,
		double l2_lambda,
		double beta1,
		double beta2 )
	: _size(init_conv_layers.size() + init_hidden_layers.size() + init_output_layers.size()),
	conv_layers(init_conv_layers),
	hidden_layers(init_hidden_layers),
	output_layer(init_output_layer),
	_learning_rate(learning_rate),
	_l2_lambda(l2_lambda),
	_beta1(beta1),
	_beta2(beta2)
{ }
