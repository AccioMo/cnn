
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
	this->_z = prev_outputs.dot(this->_weight) + this->_bias;
	Matrix	logit_exp = exp(this->_z - this->_z.row_max());
	this->_a = logit_exp / (logit_exp.sum_rows().repeat_cols(logit_exp.cols()));
	return (this->_a);
}

void	OutputLayer::backpropagation( const Matrix &expected_outputs ) {
	/*
	Cross-Entropy Loss function is technically:
	|	Error = -∑( expected * log(output) )
	but when using Softmax, it simplifies to the 
	difference between the expected and predicted 
	outputs. thus the following ...
	*/
	this->_error = this->_a - expected_outputs;
}

double	OutputLayer::getAccuracy( void ) const {
	return (_accuracy);
}
