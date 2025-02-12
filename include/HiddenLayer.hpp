
#ifndef HIDDENLAYER_HPP
# define HIDDENLAYER_HPP

# include "BaseLayer.hpp"

class HiddenLayer: public BaseLayer {
	private:
		const int	_index;

	public:
		HiddenLayer( void );
		HiddenLayer( int index, int input_size, int output_size );
		~HiddenLayer() override;

		Matrix	&feedforward( const Matrix &prev_outputs ) override;
		void	backpropagation( const BaseLayer &next_layer );
		
		int		getIndex( void ) const;
};

#endif
