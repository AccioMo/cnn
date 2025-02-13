
#ifndef CONVLAYER_HPP
# define CONVLAYER_HPP

# include "BaseLayer.hpp"
# include "convolution.hpp"

class ConvLayer : public BaseLayer {
	private:
		const int	_index;

	public:
		ConvLayer( void );
		ConvLayer( int index, int kernel_size );
		~ConvLayer( ) override;

		Matrix	&feedforward( const Matrix &prev_outputs ) override;
		void	backpropagation( const BaseLayer &next_layer );
		void	update( const Matrix &inputs, 
							  double learning_rate, 
							  int timestep, 
							  double l2_reg, 
							  double beta1, 
							  double beta2 );
		
		int		getIndex( void ) const;
};

#endif
