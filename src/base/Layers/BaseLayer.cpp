
#include "BaseLayer.hpp"

BaseLayer::BaseLayer( int input_size, int output_size )
	: _type("none"),
	_neurons(output_size),
	_weights(Matrix(input_size, output_size, \
		xavier_glorot_init(input_size, output_size))),
	_biases(Matrix(1, output_size, 0.1)),
	_m(Matrix(input_size, output_size, 0.0)),
	_v(Matrix(input_size, output_size, 0.0))
{ }

void	BaseLayer::update( const Matrix &inputs, 
							  double learning_rate, 
							  int timestep, 
							  double l2_reg, 
							  double beta1, 
							  double beta2 ) {

	this->_gradient = inputs.transpose() * this->_deltas;
	Matrix	weight_gradient = this->_gradient + (this->_weights * l2_reg);

	this->_m = this->_m * beta1 + weight_gradient * (1.0 - beta1);
	this->_v = this->_v * beta2 + weight_gradient.square() * (1.0 - beta2);

	Matrix	m_hat = this->_m / (1.0 - std::pow(beta1, timestep));
	Matrix	v_hat = this->_v / (1.0 - std::pow(beta2, timestep));

	this->_weights = this->_weights - (m_hat / v_hat.sqrt() + 1e-8) * learning_rate;

	this->_biases = this->_biases - (this->_deltas.sum_columns() * learning_rate);
}

BaseLayer::~BaseLayer() { }

Matrix  BaseLayer::getWeights( void ) const {
    return (this->_weights);
}

void    BaseLayer::setWeights( const Matrix &new_weights ) {
	this->_weights = new_weights;
}

int  BaseLayer::getSize( void ) const {
    return (this->_neurons);
}

void    BaseLayer::setSize( int new_size ) {
	this->_neurons = new_size;
}

Matrix  BaseLayer::getBiases( void ) const {
    return (this->_biases);
}

void    BaseLayer::setBiases( Matrix &new_biases ) {
	this->_biases = new_biases;
}

Matrix  BaseLayer::getOutputs( void ) const {
    return (this->_outputs);
}

void    BaseLayer::setOutputs( Matrix &new_outputs ) {
	this->_outputs = new_outputs;
}

Matrix  BaseLayer::getErrors( void ) const {
    return (this->_errors);
}

void    BaseLayer::setErrors( Matrix &new_errors ) {
	this->_errors = new_errors;
}

Matrix  BaseLayer::getDeltas( void ) const {
    return (this->_deltas);
}

void    BaseLayer::setDeltas( Matrix &new_deltas ) {
	this->_deltas = new_deltas;
}

std::string  BaseLayer::getType( void ) const {
    return (this->_type);
}

std::ostream	&operator<<( std::ostream &os, BaseLayer &nl ) {
	os << "\t --- " << nl.getType() << " layer --- " << std::endl;
	os << "  < Weights > " << std::endl << nl.getWeights() << std::endl;
	os << std::endl;
	os << "  < Biases > " << std::endl << nl.getBiases() << std::endl;
	return (os);
}
