
#ifndef CNN_HPP
# define CNN_HPP

# include "NeuralNetwork.hpp"
# include "ConvLayer.hpp"

class CNN : public NeuralNetwork {
	private:
		/* for convolutional layers, 
		weight matrix is the kernel */

	public:
		std::vector<ConvLayer>	conv_layers;

		CNN( std::vector<int> conv_layers, 
				std::vector<int> connected_layers, 
				double learning_rate = 0.01, 
				double l2_lambda = 0.0001, 
				double beta1 = 0.9, 
				double beta2 = 0.999 );

		CNN( std::vector<ConvLayer> conv_layers,
				std::vector<HiddenLayer> hidden_layers,
				OutputLayer output_layers,
				double learning_rate,
				double l2_lambda,
				double beta1,
				double beta2 );

		CNN( const char *filename );
		~CNN( );

		void	feedforward( const Matrix &inputs );

		/* ... backpropagation ... */
		void	backpropagation( const Matrix &expected_outputs );

		/* ... update ... */
		void	update( const Matrix &inputs, int timestep );

		/* ... training ... */
		void	train( Matrix input_batch, Matrix output_batch, int epochs, int timestep );
		void	trainOnFile( const char *filename, const char *labels, const char *output_file );

		/* ... testing ... */
		void	test( const Matrix input, const Matrix expected_outputs );
		void	testOnFile( const char *filename, const char *labels );

		Matrix	run( const Matrix input );
		Matrix	runOnImage( const char *filename );

		/* ... printing ... */
		void	printData( const Matrix expected_outputs ) const;

		/* ... saving ... */
		void	saveConfigJson(const char *filename) const;
		void	saveConfigBin(const char *filename) const;

		/* ... evaluation ... */
		Matrix	calculateEntropy( void ) const;
		Matrix	calculateAccuracy( const Matrix &expected_ouputs ) const;

		/* ... getters ... */
		Matrix	getEntropy( void ) const;
		Matrix	getConfidence( void ) const;
		double	getLearningRate( void ) const;

		/* ... setters ... */
		void	setEntropy( Matrix entropy );
		void	setConfidence( Matrix confidence );
		void	setLearningRate( double learning_rate );
};

#endif
