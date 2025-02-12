
#ifndef CONVLAYER_HPP
# define CONVLAYER_HPP

# include "BaseLayer.hpp"

class ConvLayer : public BaseLayer {
	private:
		const int	_index;

	public:
		ConvLayer( void );
		ConvLayer( int index, int kernel_size );
		~ConvLayer( ) override;

		Matrix	&feedforward( const Matrix &prev_outputs ) override;
		void	backpropagation( const BaseLayer &next_layer );
		
		int		getIndex( void ) const;
};

#endif
