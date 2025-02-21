
#include "CNN.hpp"

CNN::CNN( nlohmann::json arch ) : NeuralNetwork() {
	_learning_rate = arch["learning_rate"].get<float>();
	_l2_lambda = arch["l2_reg"].get<float>();
	_batch_size = arch["batch_size"].get<int>();
	_epochs = arch["epochs"].get<int>();
	int input_width = arch["input"]["width"].get<int>();
	int input_height = arch["input"]["height"].get<int>();
	int channels = arch["input"]["channels"].get<int>();
	int prev_layer_size = 0;
	for (auto &layer : arch["conv_layers"]) {
		int kernel_size = layer["kernel_size"].get<int>();
		int filters = layer["filters"].get<int>();
		int stride = layer["stride"].get<int>();
		int padding = layer["padding"].get<std::string>() == "same" ? (kernel_size-1) / 2 : 0;
		this->conv_layers.emplace_back(ConvLayer(kernel_size, channels, filters, stride, padding));
		if ((input_width - kernel_size + 2*padding) % stride != 0 || \
			(input_height - kernel_size + 2*padding) % stride != 0) {
			std::cerr << "error: invalid stride for input width. try different image or kernel size." << std::endl;
			exit(1);
		}
		input_width = (input_width - kernel_size + 2*padding) / stride + 1;
		input_height = (input_height - kernel_size + 2*padding) / stride + 1;
		channels = filters;
		std::cout << "input: " << input_width << "x" << input_height << "x" << channels << std::endl;
		prev_layer_size = (input_width*input_height*channels);
	}
	for (auto &layer : arch["hidden_layers"]) {
		int neurons = layer["units"].get<int>();
		this->hidden_layers.emplace_back(HiddenLayer(prev_layer_size, neurons));
		prev_layer_size = neurons;
	}
	int output_size = arch["output_layer"]["units"].get<int>();
	this->output_layer = OutputLayer(prev_layer_size, output_size);
}

CNN::CNN( std::vector<int> input_shape,
		std::vector<int> convolutional_layers,
		std::vector<int> connected_layers,
		float learning_rate,
		float l2_lambda,
		float beta1,
		float beta2 )
	: NeuralNetwork(connected_layers, learning_rate, l2_lambda, beta1, beta2) {
	this->conv_layers.reserve(convolutional_layers.size());
	int input_channel = input_shape[3];
	for (size_t i = 0; i < convolutional_layers.size(); i++) {
		this->conv_layers.emplace_back(ConvLayer(convolutional_layers[i], input_channel, 9));
		input_channel = 9;
	}
}

CNN::CNN( std::vector<ConvLayer> init_conv_layers,
		std::vector<HiddenLayer> init_hidden_layers,
		OutputLayer init_output_layer,
		float learning_rate,
		float l2_lambda,
		float beta1,
		float beta2 )
	: NeuralNetwork(init_hidden_layers, init_output_layer, learning_rate, l2_lambda, beta1, beta2),
	conv_layers(init_conv_layers)
{ }

CNN::CNN( const char *filename ) {

	std::cout << "loading network from file: " << filename << std::endl;
	std::streamsize size = get_file_size(filename);
	std::vector<unsigned char> mnist_train_images = read_binary_file(filename, (size_t)size);
	
	int	i = 0;
	
	std::memcpy(&this->_size, &mnist_train_images[i], sizeof(int));
	i += sizeof(int);
	std::memcpy(&this->_learning_rate, &mnist_train_images[i], sizeof(double));
	i += sizeof(double);

	int	*config_nodes = new int[this->_size];
	for (int j = 0; j < this->_size; j++) {
		std::memcpy(&config_nodes[j], &mnist_train_images[i], sizeof(int));
		i += sizeof(int);
	}

	for (int k = 0; k < this->_size - 2; k++) {

		int		neurons = config_nodes[k + 1];

		Matrix	weight = Matrix(config_nodes[k], neurons);
		for (int j = 0; j < config_nodes[k]; j++) {
			for (int l = 0; l < neurons; l++) {
				std::memcpy(&weight.m[j][l], &mnist_train_images[i], sizeof(double));
				i += sizeof(double);
			}
		}
		Matrix	bias = Matrix(1, neurons);

		for (int j = 0; j < neurons; j++) {
			std::memcpy(&bias.m[0][j], &mnist_train_images[i], sizeof(double));
			i += sizeof(double);
		}

		HiddenLayer	hidden_layer(config_nodes[k], neurons);
		hidden_layer.setWeight(weight);
		hidden_layer.setBias(bias);
		hidden_layer.setSize(neurons);
		this->hidden_layers.push_back(hidden_layer);
	}

	int		neurons = config_nodes[this->_size - 1];

	Matrix	weight = Matrix(config_nodes[this->_size - 2], neurons);

	for (int j = 0; j < config_nodes[this->_size - 2]; j++) {
		for (int l = 0; l < neurons; l++) {
			std::memcpy(&weight.m[j][l], &mnist_train_images[i], sizeof(double));
			i += sizeof(double);
		}
	}

	Matrix	bias = Matrix(1, neurons);
	
	for (int j = 0; j < neurons; j++) {
		std::memcpy(&bias.m[0][j], &mnist_train_images[i], sizeof(double));
		i += sizeof(double);
	}

	OutputLayer	new_output_layer(config_nodes[this->_size - 2], neurons);
	new_output_layer.setWeight(weight);
	new_output_layer.setBias(bias);
	new_output_layer.setSize(neurons);
	this->output_layer = new_output_layer;
}

CNN::~CNN() { }

void	CNN::saveConfigJson( const char *filename ) const {
    std::ofstream file(filename, std::ios::out);
    
    if (file.is_open()) {
		file << "{\n";
   		file << "\"size\": " << this->_size << "," << std::endl;
		file << "\"learning_rate\": " << this->_learning_rate << ", " << std::endl;
		file << "\"layers\": [" << this->hidden_layers[0].getWeight().rows() << ", ";
		for (auto &layer_neurons : this->hidden_layers)
			file << layer_neurons.getSize() << ", ";
		file << this->output_layer.getSize();
		file << "], " << std::endl;

		file << "\"hidden_layers\": [\n";
		for (auto &layer : this->hidden_layers) {
			file << "{" << std::endl;
			file << "\"weight\": " << layer.getWeight() << ", " << std::endl;
			file << "\"bias\": " << layer.getBias() << std::endl;
			file << "}";
			if (&layer != &this->hidden_layers.back())
				file << std::endl << "," << std::endl;
		}
		file << "],\n";
		file << "\"output_layer\": {" << std::endl;
		file << "\"weight\": " << this->output_layer.getWeight() << ", " << std::endl;
		file << "\"bias\": " << this->output_layer.getBias() << std::endl;
		file << "}";
		file << "}" << std::endl;
        file.close();
    } else {
        std::cerr << "Error opening file: " << filename << std::endl;
    }
}

void	CNN::saveConfigBin(const char *filename) const {
	std::ofstream file(filename, std::ios::binary);

    if (file.is_open()) {

        file.write(reinterpret_cast<const char *>(&this->_size), sizeof(this->_size));

        file.write(reinterpret_cast<const char *>(&this->_learning_rate), sizeof(this->_learning_rate));

        int rows = this->hidden_layers[0].getWeight().rows();
        file.write(reinterpret_cast<const char *>(&rows), sizeof(rows));

        for (const auto &layer : this->hidden_layers) {
            int size = layer.getSize();
            file.write(reinterpret_cast<const char *>(&size), sizeof(size));
        }

        int output_size = this->output_layer.getSize();
        file.write(reinterpret_cast<const char *>(&output_size), sizeof(output_size));

        for (const auto &layer : this->hidden_layers) {
			for (int i = 0; i < layer.getWeight().rows(); i++) {
				file.write(reinterpret_cast<const char *>(layer.getWeight().m[i].data()), layer.getWeight().m[i].size() * sizeof(double));
			}
			file.write(reinterpret_cast<const char *>(layer.getBias().m[0].data()), layer.getBias().m[0].size() * sizeof(double));
        }

        for (int i = 0; i < this->output_layer.getWeight().rows(); i++) {
			file.write(reinterpret_cast<const char *>(this->output_layer.getWeight().m[i].data()), this->output_layer.getWeight().m[i].size() * sizeof(double));
		}
		file.write(reinterpret_cast<const char *>(this->output_layer.getBias().m[0].data()), this->output_layer.getBias().m[0].size() * sizeof(double));

        file.close();
    } else {
        std::cerr << "Error opening file: " << filename;
    }
}
