
#include "OutputLayer.hpp"

OutputLayer::OutputLayer( int input_size, int output_size )
	: BaseLayer(input_size, output_size)
{
	_type = "output";
}

OutputLayer::OutputLayer( void )
	: BaseLayer(0, 0)
{ }

OutputLayer::~OutputLayer() { }

Matrix	&OutputLayer::feedforward( const Matrix &prev_outputs ) {
	/*
	Softmax function is:
	|	f(x) = exp(x) / ∑( exp(x) )
	i also subtract the row max for 
	numerical stability (in case `x` 
	is a large positive number).
	*/
	Matrix	logit = (prev_outputs * this->_weights) + this->_biases;
	Matrix	logit_exp = exp(logit - logit.row_max());
	this->_outputs = logit_exp / (logit_exp.sum_rows().repeat_columns(logit_exp.columns()));
	return (this->_outputs);
}

void	OutputLayer::backpropagation( const Matrix &expected_outputs ) {
	/*
	Cross-Entropy Loss function is technically:
	|	Error = -∑( expected * log(output) )
	but when using Softmax, it simplifies to the 
	difference between the expected and predicted 
	outputs. thus the following ...
	*/
	this->_errors = this->_outputs - expected_outputs;
	this->_deltas = this->_errors;
}

double	OutputLayer::getAccuracy( void ) const {
	return (_accuracy);
}
