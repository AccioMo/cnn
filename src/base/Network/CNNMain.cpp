
#include "CNN.hpp"

void	CNN::feedforward( const Tensor4D &inputs ) {
	/*
	auto start = std::chrono::high_resolution_clock::now();
	*/

	Tensor4D	filter_outputs = inputs;
	for (auto &layer : conv_layers) {
		filter_outputs = layer.feedforward(filter_outputs);
	}
	Matrix	outputs = flatten(filter_outputs);
	for (auto &layer : hidden_layers) {
		outputs = layer.feedforward(outputs);
	}
	output_layer.feedforward(outputs);

	/*
	auto end = std::chrono::high_resolution_clock::now();
	auto step = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << " feedforward step took " << step << " ms ";
	*/
}

void	CNN::backpropagation( const Matrix &expected_outputs ) {
	/*
	auto start = std::chrono::high_resolution_clock::now();
	*/

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

	/*
	auto end = std::chrono::high_resolution_clock::now();
	auto step = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << " backpropagation step took " << step << " ms ";
	*/
}

void	CNN::update( const Tensor4D &inputs, int timestep ) {
	/*
	auto start = std::chrono::high_resolution_clock::now();
	*/

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

	/*
	auto end = std::chrono::high_resolution_clock::now();
	auto step = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << " update step took " << step << " ms ";
	*/
}

void	CNN::train( const Tensor4D &input_batch, Matrix &output_batch, int timestep ) {
	this->feedforward(input_batch);
	this->backpropagation(output_batch);
	this->update(input_batch, timestep);
}

void	CNN::trainOnFile( const char *filename, const char *labels ) {

	std::vector<Tensor4D>	inputs = get_mnist_batch(filename, 
		_batch_size, _batches);
	std::vector<Matrix>		outputs = get_mnist_labels(labels, 
		_batch_size, _batches);

	float input_min = 0.0;
	float input_max = 255.0;
	float output_min = 0.0;
	float output_max = 1.0;

	float start = ft_get_time();
	std::cout << "convolutional network '" << _name << "' built" << std::endl;
	for (int e = 0; e < _epochs; e++) {
		auto epoch_chrono = std::chrono::high_resolution_clock::now();
		std::cout << "\033[1mepoch " << e + 1 << " of " << _epochs << "\033[0m" << std::endl;
		for (int b = 0; b < _batches; b++) {
			int progress = (b * 100) / _batches;
			int bar_width = 30;
			int filled = (progress * bar_width) / 100;
			std::cout << "\r" << b + 1 << "/" << _batches << " ";
			for (int i = 0; i < bar_width; i++) {
				if (i < filled) std::cout << "\033[32m━\033[0m";
				else if (i == filled && progress < 100) std::cout << "\033[32m━\033[0m";
				else std::cout << "━";
			}
			auto epoch_chrono_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
				std::chrono::high_resolution_clock::now() - epoch_chrono);
			std::cout << " " << epoch_chrono_elapsed.count() << "s ";

			/* DATA NORMALIZATION */
			Tensor4D	normalized_inputs = normalize(inputs[b], input_min, input_max);
			Matrix		normalized_outputs = outputs[b].normalize(output_min, output_max);

			auto batch_chrono_start = std::chrono::high_resolution_clock::now();

			/* TRAINING */
			this->train(normalized_inputs, normalized_outputs, b + 1);

			auto batch_chrono_end = std::chrono::high_resolution_clock::now();
			auto step = std::chrono::duration_cast<std::chrono::milliseconds>(batch_chrono_end - batch_chrono_start).count();
			std::cout << step << "ms/step ";
			float max_entropy = -std::log(1.0 / (float)output_layer.getSize());
			std::cout << "| accuracy: " << std::fixed << std::setprecision(4) << this->calculateAccuracy(normalized_outputs).mean() << " ";
			std::cout << "| entropy: " << this->calculateEntropy().mean() << " ";
			std::cout << "| loss: " << this->calculateLoss(normalized_outputs) << " ";
			std::cout << "| confidence: " << (1.0 - (this->calculateEntropy().mean() / max_entropy));
			std::cout.flush();
		}
		std::cout << std::endl;
	}

	std::cout << "total time: " << (ft_get_time() - start) / 1000 << "s" << std::endl;

	this->saveConfigBin((_name + ".bin").c_str());

	std::cout << "weights saved to " << _name + ".bin" << "." << std::endl;
}

void	CNN::testOnFile( const char *filename, const char *labels ) {

	float input_min = 0.0;
	float input_max = 255.0;
	float output_min = 0.0;
	float output_max = 1.0;

	int batches = TEST_DATASET_SIZE / _batch_size;

	std::vector<Tensor4D>	t_inputs = get_mnist_batch(filename, 
		_batch_size, batches);
	std::vector<Matrix>	t_outputs = get_mnist_labels(labels, 
		_batch_size, batches);

	std::cout << "testing network '" << _name << "'" << std::endl;

	float total_accuracy = 0.0,
		  total_loss = 0.0,
		  total_entropy = 0.0,
		  total_confidence = 0.0;

	float max_entropy = -std::log(1.0 / (float)output_layer.getSize());
	auto epoch_chrono = std::chrono::high_resolution_clock::now();
	for (int b = 0; b < batches; b++) {

		int progress = (b * 100) / batches;
		int bar_width = 30;
		int filled = (progress * bar_width) / 100;
		std::cout << "\r" << b + 1 << "/" << batches << " ";
		for (int i = 0; i < bar_width; i++) {
			if (i < filled) std::cout << "\033[33m━\033[0m";
			else if (i == filled && progress < 100) std::cout << "\033[33m━\033[0m";
			else std::cout << "━";
		}
		auto epoch_chrono_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
			std::chrono::high_resolution_clock::now() - epoch_chrono);
		std::cout << " " << epoch_chrono_elapsed.count() << "s ";

		Tensor4D	normalized_inputs = normalize(t_inputs[b], input_min, input_max);
		Matrix		normalized_outputs = t_outputs[b].normalize(output_min, output_max);

		this->feedforward(normalized_inputs);

		total_accuracy += this->calculateAccuracy(normalized_outputs).mean();
		total_loss += this->calculateLoss(normalized_outputs);
		total_entropy += this->calculateEntropy().mean();
		total_confidence += 1.0 - (this->calculateEntropy().mean() / max_entropy);
		
		std::cout << "| accuracy: " << std::fixed << std::setprecision(4) << this->calculateAccuracy(normalized_outputs).mean() << " ";
		std::cout << "| entropy: " << this->calculateEntropy().mean() << " ";
		std::cout << "| loss: " << this->calculateLoss(normalized_outputs) << " ";
		std::cout << "| confidence: " << (1.0 - (this->calculateEntropy().mean() / max_entropy));
		std::cout.flush();

	}

	std::cout << std::endl;
	std::cout << "\033[1mtest results: \033[0m";
	std::cout << "accuracy: " << total_accuracy / batches << " ";
	std::cout << "| entropy: " << total_entropy / batches << " ";
	std::cout << "| loss: " << total_loss / batches << " ";
	std::cout << "| confidence: " << (1.0 - (total_entropy / batches / max_entropy));
	std::cout << std::endl;

}

Matrix	CNN::run( const Tensor4D &input ) {
	int	possible_outputs[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

	Tensor4D normalized_input = input;


	std::ofstream debug_file("debug_input.txt");
	if (debug_file.is_open()) {
		debug_file << "Input tensor (1x" << normalized_input.dimension(1) 
			<< "x" << normalized_input.dimension(2) 
			<< "x" << normalized_input.dimension(3) << "):\n";
		for (int y = 0; y < normalized_input.dimension(2); ++y) {
			for (int x = 0; x < normalized_input.dimension(1); ++x) {
				debug_file << static_cast<float>(normalized_input(0, y, x, 0)) << " ";
			}
			debug_file << "\n";
		}
		debug_file.close();
		std::cout << "Input tensor saved to debug_input.txt" << std::endl;
	}

	this->feedforward(normalized_input);

	for (int i = 0; i < output_layer.getSize(); i++) {
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
	// if (channels != 1) {
	// 	std::cerr << "Image must be grayscale" << std::endl;
	// 	return (Matrix());
	// }
	channels = 1;
	Tensor4D	input(1, width, height, channels);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			for (int c = 0; c < channels; ++c) {
				input(0, y, x, c) = static_cast<float>((255.0f - image[y * width * channels + x * channels + c]) / 255.0f);
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

		const char *iterations = reinterpret_cast<const char *>(&this->_batches);
        file.write(iterations, sizeof(this->_batches));

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
			/* saving kernel && bias */
			Tensor4D kernel = c_layer.getKernel();
			const char *kernel_data = reinterpret_cast<const char *>(kernel.data());
			file.write(kernel_data, c_layer.getKernel().size() * sizeof(float));
			Tensor4D bias = c_layer.getBias();
			const char *kernel_bias = reinterpret_cast<const char *>(bias.data());
			file.write(kernel_bias, bias.size() * sizeof(float));
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
				Matrix weights = layer.getWeight();
				const char *row_weights = reinterpret_cast<const char *>(weights.m[i].data());
				file.write(row_weights, weights.m[i].size() * sizeof(float));
			}
			Matrix bias = layer.getBias();
			const char *row_bias = reinterpret_cast<const char *>(bias.m[0].data());
			file.write(row_bias, bias.m[0].size() * sizeof(float));
        }

		/* saving output layer */
		int output_size = this->output_layer.getSize();
        const char *output_size_cast = reinterpret_cast<const char *>(&output_size);
        file.write(output_size_cast, sizeof(output_size));

        for (int i = 0; i < this->output_layer.getWeight().rows(); i++) {
			Matrix weights = this->output_layer.getWeight();
			const char *row_weights = reinterpret_cast<const char *>(weights.m[i].data());
			file.write(row_weights, weights.m[i].size() * sizeof(float));
		}
		Matrix bias = this->output_layer.getBias();
		const char *bias_data = reinterpret_cast<const char *>(bias.m[0].data());
		file.write(bias_data, bias.m[0].size() * sizeof(float));

        file.close();
    } else {
        std::cerr << "Error opening file: " << filename;
    }
}

float	CNN::calculateLoss( const Matrix &expected_outputs ) const {
	/* assuming softmax activation */
	Matrix	predicted_outputs = this->output_layer.getOutput();
	float	loss = -expected_outputs.hadamard_product(log(predicted_outputs + 1e-15)).sum_rows().mean();
	return (loss);
}

Matrix	CNN::calculateEntropy( void ) const {
	/* entropy = -sum(p * log(p)) */
	float	epsilon = 1e-15;
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

float	CNN::getLearningRate( void ) const {
	return (_learning_rate);
}

void	CNN::setEntropy( Matrix entropy  ) {
	this->_entropy = entropy;
}

void	CNN::setConfidence( Matrix confidence  ) {
	this->_confidence = confidence;
}

void	CNN::setLearningRate( float learning_rate ) {
	this->_learning_rate = learning_rate;
}

std::ostream &operator<<(std::ostream &os, const CNN &cnn) {
	os << "CNN with " << cnn.conv_layers.size() << " convolutional layers and "
	   << cnn.hidden_layers.size() << " hidden layers.";
	if (!cnn.conv_layers.empty()) {
		os << "\nConvolutional layers:\n";
		for (const auto &layer : cnn.conv_layers) {
			os << "  - " << layer << "\n";
		}
	}
	if (!cnn.hidden_layers.empty()) {
		os << "Hidden layers:\n";
		for (const auto &layer : cnn.hidden_layers) {
			os << "  - " << layer << "\n";
		}
	}
	os << "Output layer:\n" << cnn.output_layer;
	os << "\nLearning rate: " << cnn.getLearningRate();
	os << "\nEntropy: " << cnn.getEntropy();
	os << "\nConfidence: " << cnn.getConfidence();
	return os;
}
