
#ifndef CNN_HPP
# define CNN_HPP

# include "NeuralNetwork.hpp"
# include "ConvLayer.hpp"
# include "nlohmann/json.hpp"

# define TRAIN_DATASET_SIZE 60000
# define TEST_DATASET_SIZE 10000

class CNN : public NeuralNetwork {

	private:

		int	_conv_size;
		int	_connected_size;

	public:

		std::vector<ConvLayer>	conv_layers;

		CNN( nlohmann::json arch );
		CNN( const char *filename );
		~CNN( );

		void	feedforward( const Tensor4D &inputs );

		/* ... backpropagation ... */
		void	backpropagation( const Matrix &expected_outputs );

		/* ... update ... */
		void	update( const Tensor4D &inputs, int timestep );

		/* ... training ... */
		void	train( const Tensor4D &input_batch, Matrix &output_batch, int timestep );
		void	trainOnFile( const char *filename, const char *labels );

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
		float	calculateLoss( const Matrix &expected_outputs ) const;
		Matrix	calculateAccuracy( const Matrix &expected_ouputs ) const;

		/* ... getters ... */
		Matrix	getEntropy( void ) const;
		Matrix	getConfidence( void ) const;
		float	getLearningRate( void ) const;

		/* ... setters ... */
		void	setEntropy( Matrix entropy );
		void	setConfidence( Matrix confidence );
		void	setLearningRate( float learning_rate );
		
};

std::ostream	&operator<<(std::ostream &os, const CNN &cnn);

#endif
