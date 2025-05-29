
#ifndef OUTPUTLAYER_HPP
# define OUTPUTLAYER_HPP

# include "BaseLayer.hpp"

class OutputLayer: public BaseLayer {
	private:
		float	_accuracy;

	public:
		OutputLayer( void );
		OutputLayer( int input_size, int output_size );
		~OutputLayer() override;

		Matrix	&feedforward( const Matrix &prev_outputs ) override;
		void	backpropagation( const Matrix &expected_outputs );

		float	getAccuracy( void ) const;
};

#endif
