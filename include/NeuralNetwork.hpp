
#ifndef NEURALNETWORK_HPP
# define NEURALNETWORK_HPP

# include <iostream>
# include <fstream>
# include <cstring>

# include "HiddenLayer.hpp"
# include "OutputLayer.hpp"
# include "Matrix.hpp"
# include "math.hpp"
# include "utils.hpp"

class NeuralNetwork {
	private:
		int			_size;

	protected:
		std::string	_name;
		int		_epochs;
		int		_timestep;
		int 	_batch_size;
		int		_batches;

		Matrix	_entropy;
		Matrix	_confidence;

		float	_learning_rate;
		float	_l2_lambda;
		float	_beta1;
		float	_beta2;

	public:
		std::vector<HiddenLayer>	hidden_layers;
		OutputLayer					output_layer;

		NeuralNetwork( );
		NeuralNetwork( std::vector<int> nodes, 
					float learning_rate = 0.01, 
					float l2_lambda = 0.0001, 
					float beta1 = 0.9, 
					float beta2 = 0.999 );
		NeuralNetwork( std::vector<HiddenLayer> hidden_layers, 
					OutputLayer output_layer,
					float learning_rate = 0.01, 
					float l2_lambda = 0.0001, 
					float beta1 = 0.9, 
					float beta2 = 0.999 );

		NeuralNetwork( const char *filename );
		~NeuralNetwork( );

		/* ... feedforward ... */
		void	feedforward( const Matrix &inputs );

		/* ... backpropagation ... */
		void	backpropagation( const Matrix &expected_outputs );

		/* ... update ... */
		void	update( const Matrix &inputs, int timestep );

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
		float	getLearningRate( void ) const;

		/* ... setters ... */
		void	setEntropy( Matrix entropy );
		void	setConfidence( Matrix confidence );
		void	setLearningRate( float learning_rate );
};

#endif
