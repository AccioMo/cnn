
#ifndef CNN_HPP
# define CNN_HPP

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
				double beta2 )
}

#endif
