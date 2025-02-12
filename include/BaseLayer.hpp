
#ifndef BASELAYER_HPP
# define BASELAYER_HPP

# include <iostream>
# include "Matrix.hpp"
# include "math.hpp"
# include "config.hpp"

class BaseLayer {
	protected:
		/* type of the layer (output or hidden) */
		std::string	_type;

		/* number of neuron/nodes/units in the network */
		int			_neurons;

		/* these weights and biases should 
		connect this layer with the next one */
		Matrix		_weights;
		Matrix		_biases;

		/* predicted outputs based on this 
		layer's weights and biases */
		Matrix		_outputs;

		/* errors is the raw difference between 
		predicted and expected output, calculated 
		differenly in output and hidden layers 
		(technically should use loss function, 
		but took a shortcut hehe). 

		Here it's the intermediate state from 
		output to the 'real' back-propagated error */
		Matrix		_errors;

		/* loss delta: final back-propagated error for 
		the layer. calculated using the derivative of 
		the activation function and the raw error 
		value `_errors` */
		Matrix		_deltas;

		/* gradient of the loss: ðL/ðx
		(loss with respect to weights) */
		Matrix		_gradient;

		Matrix		_m;
		Matrix		_v;

	public:
		BaseLayer( int input_size, int output_size );
		virtual	~BaseLayer();

		/* feeds output of the previous layer 
		to the current one to predict output */
		virtual Matrix	&feedforward( const Matrix &prev_outputs ) = 0;

		/* backpropagation function in derived 
		HiddenLayer and OutputLayer classes */

		/* updates weights and biases using 
		delta from backpropagation function */
		void	update( const Matrix &prev_outputs,
						double learning_rate,
						int    timestep,
						double l2_reg = 0.0,
						double beta1 = 0.9,
						double beta2 = 0.999 );

		int 	getSize( void ) const;
		void    setSize( int new_size );
		Matrix  getWeights( void ) const;
		void    setWeights( const Matrix &new_weights );
		Matrix  getBiases( void ) const;
		void    setBiases( Matrix &new_biases );

		Matrix  getOutputs( void ) const;
		void    setOutputs( Matrix &new_outputs );
		Matrix  getErrors( void ) const;
		void    setErrors( Matrix &new_errors );
		Matrix  getDeltas( void ) const;
		void    setDeltas( Matrix &new_deltas );

		std::string	getType() const;
};

std::ostream	&operator<<( std::ostream &os, BaseLayer &nl );

#endif
