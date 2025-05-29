
#ifndef BASELAYER_HPP
# define BASELAYER_HPP

# include <iostream>
# include "Matrix.hpp"
# include "math.hpp"

class BaseLayer {
	protected:
		/* type of the layer (output or hidden) */
		std::string	_type;

		/* number of neuron/nodes/units in the network */
		int			_size;

		/* these weights and biases should 
		connect this layer with the next one */
		Matrix		_weight;
		Matrix		_bias;

		/* pre-activation outputs */
		Matrix		_z;

		/* post-activation outputs */
		Matrix		_a;

		/* errors is the raw difference between 
		predicted and expected output, calculated 
		differenly in output and hidden layers 
		(technically should use loss function, 
		but took a shortcut hehe). 

		Here it's the intermediate state from 
		output to the 'real' back-propagated error */
		Matrix		_error;

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
						float learning_rate,
						int    timestep,
						float l2_reg = 0.0,
						float beta1 = 0.9,
						float beta2 = 0.999 );

		int 	getSize( void ) const;
		void    setSize( int new_size );
		Matrix  getWeight( void ) const;
		void    setWeight( const Matrix &new_weight );
		Matrix  getBias( void ) const;
		void    setBias( Matrix &new_bias );

		Matrix  getOutput( void ) const;
		void    setOutput( Matrix &new_output );
		Matrix  getError( void ) const;
		void    setError( Matrix &new_error );

		std::string	getType() const;
};

std::ostream	&operator<<( std::ostream &os, const BaseLayer &nl );

#endif
