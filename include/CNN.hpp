
#ifndef CNN_HPP
# define CNN_HPP

# include "NeuralNetwork.hpp"
# include "ConvLayer.hpp"
# include "nlohmann/json.hpp"

class CNN : public NeuralNetwork {
	private:
		/* for convolutional layers, 
		weight matrix is the kernel */

	public:
		std::vector<ConvLayer>	conv_layers;

		CNN( nlohmann::json arch );

		CNN( std::vector<int> input_shape, 
				std::vector<int> conv_struct, 
				std::vector<int> connected_struct, 
				float learning_rate = 0.01, 
				float l2_lambda = 0.0001, 
				float beta1 = 0.9, 
				float beta2 = 0.999 );

		CNN( std::vector<ConvLayer> init_conv_layers,
			std::vector<HiddenLayer> init_hidden_layers,
			OutputLayer init_output_layer,
			float learning_rate,
			float l2_lambda,
			float beta1,
			float beta2 );

		CNN( const char *filename );
		~CNN( );

		void	feedforward( const Tensor4D &inputs );

		/* ... backpropagation ... */
		void	backpropagation( const Matrix &expected_outputs );

		/* ... update ... */
		void	update( const Tensor4D &inputs, int timestep );

		/* ... training ... */
		void	train( const Tensor4D &input_batch, Matrix &output_batch, int epochs, int timestep );
		void	trainOnFile( const char *filename, const char *labels, const char *output_file );

		/* ... testing ... */
		void	test( const Tensor4D &input, const Matrix &expected_outputs );
		void	testOnFile( const char *filename, const char *labels );

		Matrix	run( const Tensor4D &input );
		Matrix	runOnImage( const char *filename );

		/* ... printing ... */
		void	printData( const Matrix &expected_outputs ) const;

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
