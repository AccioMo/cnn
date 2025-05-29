
#include "NeuralNetwork.hpp"

void	NeuralNetwork::feedforward( const Matrix &inputs ) {
	Matrix	outputs = inputs;
	for (auto &layer : hidden_layers) {
		outputs = layer.feedforward(outputs);
	}
	output_layer.feedforward(outputs);
}

void	NeuralNetwork::backpropagation( const Matrix &expected_outputs ) {
	BaseLayer	*next_layer = &output_layer;
	output_layer.backpropagation(expected_outputs);
	for (int i = hidden_layers.size() - 1; i >= 0; i--) {
		this->hidden_layers[i].backpropagation(*next_layer);
		next_layer = &this->hidden_layers[i];
	}
}

void	NeuralNetwork::update( const Matrix &inputs, int timestep ) {
	Matrix	outputs = inputs;
	for (auto &layer : hidden_layers) {
		layer.update(outputs, this->_learning_rate, timestep, \
			this->_l2_lambda, this->_beta1, this->_beta2);
		outputs = layer.getOutput();
	}
	output_layer.update(outputs, this->_learning_rate, timestep, \
		this->_l2_lambda, this->_beta1, this->_beta2);
}

void	NeuralNetwork::printData( const Matrix expected_outputs ) const {
	float max_entropy = -std::log(1.0 / (float)output_layer.getSize());
	std::cout << "accuracy (end)\t: " << this->calculateAccuracy(expected_outputs).mean() * 100 << "%" << std::endl;
	std::cout << "entropy\t\t: " << this->calculateEntropy().mean() << " (max " << max_entropy << ")" << std::endl;
	std::cout << "confidence\t: " << (1.0 - (this->calculateEntropy().mean() / max_entropy)) * 100 << "%" << std::endl;
}

Matrix	NeuralNetwork::calculateEntropy( void ) const {
	/* entropy = -sum(p * log(p)) */
	float	epsilon = 1e-15;
	Matrix	predicted_outputs = this->output_layer.getOutput();
	Matrix	entropy = predicted_outputs.hadamard_product(log(predicted_outputs + epsilon)).sum_rows() * -1.0;
	return (entropy);
}

Matrix	NeuralNetwork::calculateAccuracy( const Matrix &expected_ouputs ) const {
	/* assuming softmax activation */
	Matrix	predicted_ouputs = this->output_layer.getOutput().argmax();
	return (predicted_ouputs == expected_ouputs.argmax());
}

Matrix	NeuralNetwork::getEntropy( void ) const {
	return (_entropy);
}

Matrix	NeuralNetwork::getConfidence( void ) const {
	return (_confidence);
}

float	NeuralNetwork::getLearningRate( void ) const {
	return (_learning_rate);
}

void	NeuralNetwork::setEntropy( Matrix entropy  ) {
	this->_entropy = entropy;
}

void	NeuralNetwork::setConfidence( Matrix confidence  ) {
	this->_confidence = confidence;
}

void	NeuralNetwork::setLearningRate( float learning_rate ) {
	this->_learning_rate = learning_rate;
}
