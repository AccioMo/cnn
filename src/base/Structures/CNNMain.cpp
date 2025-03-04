
#include "CNN.hpp"

void	CNN::feedforward( const Tensor4D &inputs ) {
	// double start = ft_get_time();
	Tensor4D	filter_outputs = inputs;
	for (auto &layer : conv_layers) {
		filter_outputs = layer.feedforward(filter_outputs);
	}
	Matrix	outputs = flatten(filter_outputs);
	for (auto &layer : hidden_layers) {
		outputs = layer.feedforward(outputs);
	}
	output_layer.feedforward(outputs);
	// std::cout << "feedforward time	: " << (ft_get_time() - start) / 1000 << "s" << std::endl;
}

void	CNN::backpropagation( const Matrix &expected_outputs ) {
	// double start = ft_get_time();
	BaseLayer	*next_layer = &output_layer;
	output_layer.backpropagation(expected_outputs);
	for (int i = hidden_layers.size() - 1; i >= 0; i--) {
		this->hidden_layers[i].backpropagation(*next_layer);
		next_layer = &this->hidden_layers[i];
	}
	
	ConvLayer	*next_conv_layer = &this->conv_layers[conv_layers.size() - 1];
	next_conv_layer->backpropagation(*next_layer);
	for (int i = conv_layers.size() - 2; i >= 0; i--) {
		this->conv_layers[i].backpropagation(*next_conv_layer);
		next_conv_layer = &this->conv_layers[i];
	}
	// std::cout << "backpropagation time	: " << (ft_get_time() - start) / 1000 << "s" << std::endl;
}

void	CNN::update( const Tensor4D &inputs, int timestep ) {
	// double start = ft_get_time();
	Tensor4D	outputs = inputs;
	for (auto &layer : conv_layers) {
		layer.update(outputs, this->_learning_rate, timestep, \
			this->_l2_lambda, this->_beta1, this->_beta2);
		outputs = layer.getOutput();
	}
	Matrix	flat_outputs = flatten(outputs);
	for (auto &layer : hidden_layers) {
		layer.update(flat_outputs, this->_learning_rate, timestep, \
			this->_l2_lambda, this->_beta1, this->_beta2);
		flat_outputs = layer.getOutput();
	}
	output_layer.update(flat_outputs, this->_learning_rate, timestep, \
		this->_l2_lambda, this->_beta1, this->_beta2);
	// std::cout << "update time		: " << (ft_get_time() - start) / 1000 << "s" << std::endl;
}

void	CNN::train( const Tensor4D &input_batch, Matrix &output_batch, int epochs, int timestep ) {
	for (int age = 0; age < epochs; age++) {
		this->feedforward(input_batch);
		this->backpropagation(output_batch);
		this->update(input_batch, timestep);
		if (age == 0)
			std::cout << "accuracy (start): " << std::fixed << std::setprecision(2) \
				<< this->calculateAccuracy(output_batch).mean() * 100.0 << "%" << std::endl;
		std::cout << "\033[2Kepochs\t\t: " << age << std::endl << "\033[A\r";
	}
}

void	CNN::trainOnFile( const char *filename, const char *labels, const char *output_file ) {

	std::vector<Tensor4D>	inputs = get_mnist_batch(filename);
	std::vector<Matrix>	outputs = get_mnist_labels(labels);

	std::cout << "Network constructed!" << std::endl;

	double start = ft_get_time();
	std::cout << std::endl << "   --- TRAINING ---	" << std::endl;
	int	total_iterations = TRAIN_SIZE / _batch_size;
	for (int i = 0; i < total_iterations; i++) {
		std::cout << "Iteration " << i + 1 << " of " << total_iterations << std::endl;
		Tensor4D	normalized_inputs = normalize(inputs[i], INPUT_MIN, INPUT_MAX);
		Matrix		normalized_outputs = outputs[i].normalize(OUTPUT_MIN, OUTPUT_MAX);
		this->train(normalized_inputs, normalized_outputs, _epochs, i + 1);
		this->printData(normalized_outputs);
		std::cout << "   ---	" << std::endl;
	}

	std::cout << "Training done!" << std::endl << std::endl;
	std::cout << "time		: " << (ft_get_time() - start) / 1000 << "s" << std::endl;
	std::cout << "   ---	" << std::endl;

	this->saveConfigBin(output_file);

	std::cout << "Network saved!" << std::endl;
}

void	CNN::test( const Tensor4D &input, const Matrix &expected_outputs ) {
	this->feedforward(input);
	this->printData(expected_outputs);
}

void	CNN::testOnFile( const char *filename, const char *labels ) {

	std::vector<Tensor4D>	t_inputs = get_mnist_batch(filename);
	std::vector<Matrix>	t_outputs = get_mnist_labels(labels);

	std::cout << "   --- TESTING ---" << std::endl;

	double	accuracy = 0.0;
	int	test_iterations = TEST_SIZE / _batch_size;
	for (int i = 0; i < test_iterations; i++) {
		std::cout << "Iteration " << i + 1 << " of " << test_iterations << std::endl;
		Tensor4D	normalized_inputs = normalize(t_inputs[i], INPUT_MIN, INPUT_MAX);
		Matrix	normalized_outputs = t_outputs[i].normalize(OUTPUT_MIN, OUTPUT_MAX);
		this->feedforward(normalized_inputs);
		this->backpropagation(normalized_outputs);
		this->printData(normalized_outputs);
		accuracy += this->calculateAccuracy(normalized_outputs).mean();
		std::cout << "   ---	" << std::endl;
	}
	accuracy /= test_iterations + 1;
	std::cout << "TEST ACCURACY\t: " << accuracy * 100 << "%" << std::endl;
}

Matrix	CNN::run( const Tensor4D &input ) {
	Tensor4D	normalized_input = normalize(input, INPUT_MIN, INPUT_MAX);

	this->feedforward(normalized_input);

	int	possible_outputs[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
	for (int i = 0; i < POSSIBILE_OUTPUTS; i++) {
		std::cout << possible_outputs[i] << ": " << std::fixed << std::setprecision(2) << this->output_layer.getOutput().m[0][i] * 100 << "%" << std::endl;
	}
	return (this->output_layer.getOutput());
}

Matrix	CNN::runOnImage( const char *filename ) {
	int	width, height, channels;

	unsigned char	*image = load_image(filename, &width, &height, &channels, 1);
	if (image == NULL) {
		std::cerr << "Error loading image" << std::endl;
		return (Matrix());
	}
	if (width != 28 || height != 28) {
		std::cerr << "Image must be 28x28" << std::endl;
		return (Matrix());
	}

	Tensor4D	input(1, width, height, channels);
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < height; k++) {
				if (image[j] > 255 * 0.95) 
					input(0, i, j, k) = 0.0;
				else
					input(0, i, j, k) = static_cast<double>(255 - image[j]);
			}
		}
	}
	
	Matrix	output = this->run(input);

	free_image(image);
	return (output);
}

void	CNN::saveConfigBin(const char *filename) const {
	std::ofstream file(filename, std::ios::binary);

    if (file.is_open()) {

		/* saving sizes */
		const char *conv_size = reinterpret_cast<const char *>(&this->_conv_size);
        file.write(conv_size, sizeof(this->_conv_size));

		const char *connected_size = reinterpret_cast<const char *>(&this->_connected_size);
        file.write(connected_size, sizeof(this->_connected_size));

		/* saving params */
		const char *learning_rate = reinterpret_cast<const char *>(&this->_learning_rate);
        file.write(learning_rate, sizeof(this->_learning_rate));

		const char *epochs = reinterpret_cast<const char *>(&this->_epochs);
        file.write(epochs, sizeof(this->_epochs));

		const char *batch_size = reinterpret_cast<const char *>(&this->_batch_size);
        file.write(batch_size, sizeof(this->_batch_size));

		const char *l2_lambda = reinterpret_cast<const char *>(&this->_l2_lambda);
        file.write(l2_lambda, sizeof(this->_l2_lambda));

		/* saving convolutional layers */
		for (const auto &c_layer : this->conv_layers) {
			int kernel_size = c_layer.getKernel().dimension(0);
			file.write(reinterpret_cast<const char *>(&kernel_size), sizeof(kernel_size));
			int input_size = c_layer.getKernel().dimension(2);
			file.write(reinterpret_cast<const char *>(&input_size), sizeof(input_size));
			int filter_size = c_layer.getKernel().dimension(3);
			file.write(reinterpret_cast<const char *>(&filter_size), sizeof(filter_size));
			int stride = c_layer.getStride();
			file.write(reinterpret_cast<const char *>(&stride), sizeof(stride));
			int padding = c_layer.getPadding();
			file.write(reinterpret_cast<const char *>(&padding), sizeof(padding));
			/* saving data */
			const char *kernel_data = reinterpret_cast<const char *>(c_layer.getKernel().data());
			file.write(kernel_data, c_layer.getKernel().size() * sizeof(float));
			// const char *kernel_bias = reinterpret_cast<const char *>(c_layer.getBias().data());
			// file.write(kernel_bias, c_layer.getBias().size() * sizeof(float));
		}

		/* saving hidden layers */
		int input_size = this->hidden_layers[0].getWeight().rows();
        const char *input_size_cast = reinterpret_cast<const char *>(&input_size);
        file.write(input_size_cast, sizeof(input_size));

        for (const auto &layer : this->hidden_layers) {
			int layer_size = layer.getSize();
            const char *layer_neurons = reinterpret_cast<const char *>(&layer_size);
            file.write(layer_neurons, sizeof(layer_size));

			for (int i = 0; i < layer.getWeight().rows(); i++) {
				const char *row_weights = reinterpret_cast<const char *>(layer.getWeight().m[i].data());
				file.write(row_weights, layer.getWeight().m[i].size() * sizeof(double));
			}
			const char *row_bias = reinterpret_cast<const char *>(layer.getBias().m[0].data());
			file.write(row_bias, layer.getBias().m[0].size() * sizeof(double));
        }

		/* saving output layer */
		int output_size = this->output_layer.getSize();
        const char *output_size_cast = reinterpret_cast<const char *>(&output_size);
        file.write(output_size_cast, sizeof(output_size));

        for (int i = 0; i < this->output_layer.getWeight().rows(); i++) {
			const char *row_weights = reinterpret_cast<const char *>(this->output_layer.getWeight().m[i].data());
			file.write(row_weights, this->output_layer.getWeight().m[i].size() * sizeof(double));
		}
		const char *bias_data = reinterpret_cast<const char *>(this->output_layer.getBias().m[0].data());
		file.write(bias_data, this->output_layer.getBias().m[0].size() * sizeof(double));

        file.close();
    } else {
        std::cerr << "Error opening file: " << filename;
    }
}

void	CNN::printData( const Matrix &expected_outputs ) const {
	double max_entropy = -std::log(1.0 / (double)POSSIBILE_OUTPUTS);
	std::cout << "accuracy (end)\t: " << this->calculateAccuracy(expected_outputs).mean() * 100 << "%" << std::endl;
	std::cout << "entropy\t\t: " << this->calculateEntropy().mean() << " (max " << max_entropy << ")" << std::endl;
	std::cout << "confidence\t: " << (1.0 - (this->calculateEntropy().mean() / max_entropy)) * 100 << "%" << std::endl;
}

Matrix	CNN::calculateEntropy( void ) const {
	/* entropy = -sum(p * log(p)) */
	double	epsilon = 1e-15;
	Matrix	predicted_outputs = this->output_layer.getOutput();
	Matrix	entropy = predicted_outputs.hadamard_product(log(predicted_outputs + epsilon)).sum_rows() * -1.0;
	return (entropy);
}

Matrix	CNN::calculateAccuracy( const Matrix &expected_ouputs ) const {
	/* assuming softmax activation */
	Matrix	predicted_ouputs = this->output_layer.getOutput().argmax();
	return (predicted_ouputs == expected_ouputs.argmax());
}

Matrix	CNN::getEntropy( void ) const {
	return (_entropy);
}

Matrix	CNN::getConfidence( void ) const {
	return (_confidence);
}

double	CNN::getLearningRate( void ) const {
	return (_learning_rate);
}

void	CNN::setEntropy( Matrix entropy  ) {
	this->_entropy = entropy;
}

void	CNN::setConfidence( Matrix confidence  ) {
	this->_confidence = confidence;
}

void	CNN::setLearningRate( double learning_rate ) {
	this->_learning_rate = learning_rate;
}
