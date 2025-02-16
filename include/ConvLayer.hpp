
#ifndef CONVLAYER_HPP
# define CONVLAYER_HPP

# include "Eigen/Dense"
# include "unsupported/Eigen/CXX11/Tensor"

typedef Eigen::Tensor<float, 4>	Tensor4D;

class ConvLayer {
	private:

		Tensor4D	_kernel;
		Tensor4D	_bias;

		/* pre-activation outputs */
		Tensor4D	_z;

		/* post-activation outputs */
		Tensor4D	_a;

		/* errors is the raw difference between 
		predicted and expected output, calculated 
		differenly in output and hidden layers 
		(technically should use loss function, 
		but took a shortcut hehe). 

		Here it's the intermediate state from 
		output to the 'real' back-propagated error */
		Tensor4D	_error;

		/* gradient of the loss: ðL/ðx
		(loss with respect to weight/kernel) */
		Tensor4D	_gradient;

		Tensor4D	_m;
		Tensor4D	_v;

	public:
		ConvLayer( void );
		ConvLayer( int kernel_size, int input_size, int output_size );
		~ConvLayer( );

		Tensor4D &feedforward( const Tensor4D &prev_outputs );
		void	backpropagation( const Tensor4D &next_error );
		void	update( const Tensor4D &inputs, 
							  double learning_rate, 
							  int timestep, 
							  double l2_reg, 
							  double beta1, 
							  double beta2 );
		
		Tensor4D	getKernel( void ) const;
		Tensor4D	getBias( void ) const;
		Tensor4D	getOutput( void ) const;
		Tensor4D	getError( void ) const;
		
		int		getIndex( void ) const;
};

#endif
