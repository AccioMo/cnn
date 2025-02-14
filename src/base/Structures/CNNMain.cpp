
#include "CNN.hpp"

void	CNN::feedforward( const Matrix &inputs ) {
	Matrix	outputs = inputs;
	for (auto &layer : conv_layers) {
		outputs = layer.feedforward(outputs);
	}
	for (auto &layer : hidden_layers) {
		outputs = layer.feedforward(outputs);
	}
	output_layer.feedforward(outputs);
}

void	CNN::backpropagation( const Matrix &expected_outputs ) {
	BaseLayer	*next_layer = &output_layer;
	output_layer.backpropagation(expected_outputs);
	for (int i = hidden_layers.size() - 1; i >= 0; i--) {
		this->hidden_layers[i].backpropagation(*next_layer);
		next_layer = &this->hidden_layers[i];
	}
	for (int i = conv_layers.size() - 1; i >= 0; i--) {
		this->conv_layers[i].backpropagation(*next_layer);
		next_layer = &this->conv_layers[i];
	}
}

void	CNN::update( const Matrix &inputs, int timestep ) {
	Matrix	outputs = inputs;
	for (auto &layer : conv_layers) {
		layer.update(outputs, this->_learning_rate, timestep, \
			this->_l2_lambda, this->_beta1, this->_beta2);
		outputs = layer.getOutput();
	}
	for (auto &layer : hidden_layers) {
		layer.update(outputs, this->_learning_rate, timestep, \
			this->_l2_lambda, this->_beta1, this->_beta2);
		outputs = layer.getOutput();
	}
	output_layer.update(outputs, this->_learning_rate, timestep, \
		this->_l2_lambda, this->_beta1, this->_beta2);
}

void	CNN::train( Matrix input_batch, Matrix output_batch, int epochs, int timestep ) {
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

	std::vector<Matrix>	inputs = get_input_batch(filename);
	std::vector<Matrix>	outputs = get_input_labels(labels);

	std::cout << "Network constructed!" << std::endl;

	double start = ft_get_time();
	std::cout << std::endl << "   --- TRAINING ---	" << std::endl;
	int	total_iterations = TRAIN_SIZE / BATCH_SIZE;
	for (int i = 0; i < total_iterations; i++) {
		std::cout << "Iteration " << i + 1 << " of " << total_iterations << std::endl;
		Matrix	normalized_inputs = inputs[i].normalize(INPUT_MIN, INPUT_MAX);
		Matrix	normalized_outputs = outputs[i].normalize(OUTPUT_MIN, OUTPUT_MAX);
		this->train(normalized_inputs, normalized_outputs, EPOCHS, i + 1);
		this->printData(normalized_outputs);
		std::cout << "   ---	" << std::endl;
	}

	std::cout << "Training done!" << std::endl << std::endl;
	std::cout << "time		: " << (ft_get_time() - start) / 1000 << "s" << std::endl;
	std::cout << "   ---	" << std::endl;

	this->saveConfigBin(output_file);

	std::cout << "Network saved!" << std::endl;
}

void	CNN::test( const Matrix input, const Matrix expected_outputs ) {
	this->feedforward(input);
	this->printData(expected_outputs);
}

void	CNN::testOnFile( const char *filename, const char *labels ) {

	std::vector<Matrix>	t_inputs = get_input_batch(filename);
	std::vector<Matrix>	t_outputs = get_input_labels(labels);

	std::cout << "   --- TESTING ---" << std::endl;

	double	accuracy = 0.0;
	int	test_iterations = TEST_SIZE / BATCH_SIZE;
	for (int i = 0; i < test_iterations; i++) {
		std::cout << "Iteration " << i + 1 << " of " << test_iterations << std::endl;
		Matrix	normalized_inputs = t_inputs[i].normalize(INPUT_MIN, INPUT_MAX);
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

Matrix	CNN::run( const Matrix input ) {
	Matrix	normalized_input = input.normalize(INPUT_MIN, INPUT_MAX);

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

	Matrix	input(1, IMAGE_SIZE);
	for (int j = 0; j < IMAGE_SIZE; j++) {
		if (image[j] > 255 * 0.95) 
			input.m[0][j] = 0.0;
		else
			input.m[0][j] = static_cast<double>(255 - image[j]);
	}
	
	Matrix	output = this->run(input);

	free_image(image);
	return (output);
}

void	CNN::printData( const Matrix expected_outputs ) const {
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
