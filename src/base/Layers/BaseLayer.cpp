
#include "BaseLayer.hpp"

BaseLayer::BaseLayer( int input_size, int output_size )
	: _type("none"),
	_size(output_size),
	_weight(Matrix(input_size, output_size, \
		he_init(input_size))),
	_bias(Matrix(1, output_size, 0.1)),
	_m(Matrix(input_size, output_size, 0.0)),
	_v(Matrix(input_size, output_size, 0.0))
{ }

BaseLayer::~BaseLayer( ) { }

void	BaseLayer::update( const Matrix &inputs, 
							  float learning_rate, 
							  int timestep, 
							  float l2_reg, 
							  float beta1, 
							  float beta2 ) {

	this->_gradient = inputs.transpose().dot(this->_error);

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

	// std::cout << "error: " << this->_error << std::endl;
	// std::cout << "gradient: " << this->_gradient << std::endl;
	this->_weight = this->_weight - this->_gradient * learning_rate;
	
	this->_bias = this->_bias - (this->_error.sum_cols() * learning_rate);
}

Matrix  BaseLayer::getWeight( void ) const {
    return (this->_weight);
}

void    BaseLayer::setWeight( const Matrix &new_weight ) {
	this->_weight = new_weight;
}

int  BaseLayer::getSize( void ) const {
    return (this->_size);
}

void    BaseLayer::setSize( int new_size ) {
	this->_size = new_size;
}

Matrix  BaseLayer::getBias( void ) const {
    return (this->_bias);
}

void    BaseLayer::setBias( Matrix &new_bias ) {
	this->_bias = new_bias;
}

Matrix  BaseLayer::getOutput( void ) const {
    return (this->_a);
}

void    BaseLayer::setOutput( Matrix &new_output ) {
	this->_a = new_output;
}

Matrix  BaseLayer::getError( void ) const {
    return (this->_error);
}

void    BaseLayer::setError( Matrix &new_error ) {
	this->_error = new_error;
}

std::string  BaseLayer::getType( void ) const {
    return (this->_type);
}

std::ostream	&operator<<( std::ostream &os, BaseLayer &nl ) {
	os << "\t --- " << nl.getType() << " layer --- " << std::endl;
	os << "  < Weights > " << std::endl << nl.getWeight() << std::endl;
	os << std::endl;
	os << "  < Bias > " << std::endl << nl.getBias() << std::endl;
	return (os);
}
