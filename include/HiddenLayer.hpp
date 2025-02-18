
#ifndef HIDDENLAYER_HPP
# define HIDDENLAYER_HPP

# include "BaseLayer.hpp"

class HiddenLayer: public BaseLayer {
	public:
		HiddenLayer( void );
		HiddenLayer( int input_size, int output_size );
		~HiddenLayer() override;

		Matrix	&feedforward( const Matrix &prev_outputs ) override;
		void	backpropagation( const BaseLayer &next_layer );
};

#endif
