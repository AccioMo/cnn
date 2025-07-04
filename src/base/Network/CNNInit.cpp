
#include "CNN.hpp"

CNN::CNN( nlohmann::json arch ) : NeuralNetwork() {
	_conv_size = 0;
	_connected_size = 0;
	_learning_rate = arch["learning_rate"].get<float>();
	_l2_lambda = arch["l2_reg"].get<float>();
	_batch_size = arch["batch_size"].get<int>();
	_epochs = arch["epochs"].get<int>();
	if (arch.contains("batches")) {
		_batches = arch["batches"].get<int>();
	} else {
		_batches = TRAIN_DATASET_SIZE / _batch_size;
	}
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
			throw std::runtime_error("invalid input dimensions for convolutional layer.");
		}
		input_width = (input_width - kernel_size + 2*padding) / stride + 1;
		input_height = (input_height - kernel_size + 2*padding) / stride + 1;
		input_width = (input_width - 2) / 2 + 1;
		input_height = (input_height - 2) / 2 + 1;
		channels = filters;
		prev_layer_size = (input_width*input_height*channels);
		_conv_size++;
	}
	for (auto &layer : arch["hidden_layers"]) {
		int neurons = layer["units"].get<int>();
		this->hidden_layers.emplace_back(HiddenLayer(prev_layer_size, neurons));
		prev_layer_size = neurons;
		_connected_size++;
	}
	int output_size = arch["output_layer"]["units"].get<int>();
	this->output_layer = OutputLayer(prev_layer_size, output_size);
	if (arch.contains("name")) {
		this->_name = arch["name"].get<std::string>();
	} else {
		this->_name = "cnn-dials";
	}
	_connected_size++;
}

CNN::CNN( const char *filename ) {

	std::streamsize size = get_file_size(filename);
	if (size <= 0) {
		return ;
	}
	std::vector<unsigned char> net_config = read_binary_file(filename, (size_t)size);
	
	int	i = 0;
	
	/* retrieving sizes */
	std::memcpy(&this->_conv_size, &net_config[i], sizeof(int));
	i += sizeof(int);
	if (this->_conv_size < 0) {
		throw std::runtime_error("corrupt config.");
	}

	std::memcpy(&this->_connected_size, &net_config[i], sizeof(int));
	i += sizeof(int);
	if (this->_connected_size < 0) {
		throw std::runtime_error("corrupt config.");
	}

	/* retrieving params */
	std::memcpy(&this->_learning_rate, &net_config[i], sizeof(float));
	i += sizeof(float);

	std::memcpy(&this->_epochs, &net_config[i], sizeof(int));
	i += sizeof(int);

	std::memcpy(&this->_batch_size, &net_config[i], sizeof(int));
	i += sizeof(int);

	std::memcpy(&this->_l2_lambda, &net_config[i], sizeof(float));
	i += sizeof(float);

	std::memcpy(&this->_batches, &net_config[i], sizeof(float));
	i += sizeof(float);

	/* retrieving convolutional layers */
	for (int j = 0; j < this->_conv_size; j++) {
		int kernel_size, input, filters, stride, padding;
		std::memcpy(&kernel_size, &net_config[i], sizeof(int));
		i += sizeof(int);
		std::memcpy(&input, &net_config[i], sizeof(int));
		i += sizeof(int);
		std::memcpy(&filters, &net_config[i], sizeof(int));
		i += sizeof(int);
		std::memcpy(&stride, &net_config[i], sizeof(int));
		i += sizeof(int);
		std::memcpy(&padding, &net_config[i], sizeof(int));
		i += sizeof(int);
		int total_size = kernel_size*kernel_size*input*filters;
		if (size_t(total_size*sizeof(float) + i) > net_config.size()) {
			throw std::runtime_error("corrupt config.");
		}
		char *kernel_data = new char[total_size*sizeof(float)+1];
		std::memcpy(kernel_data, &net_config[i], total_size*sizeof(float));
		i += total_size*sizeof(float);
		Eigen::TensorMap<Tensor4D> kernel((float *)kernel_data, kernel_size, kernel_size, input, filters);
		ConvLayer conv_layer(kernel_size, input, filters, stride, padding);
		conv_layer.setKernel(kernel);
		Tensor4D	bias(1, 1, 1, filters);
		std::memcpy(bias.data(), &net_config[i], filters*sizeof(float));
		i += filters*sizeof(float);
		conv_layer.setBias(bias);
		this->conv_layers.push_back(conv_layer);
		delete[] kernel_data;
	}

	/* retrieving hidden layers */
	int input_size;
	std::memcpy(&input_size, &net_config[i], sizeof(int));
	i += sizeof(int);
	for (int j = 0; j < this->_connected_size - 1; j++) {
		int output_size;
		std::memcpy(&output_size, &net_config[i], sizeof(int));
		i += sizeof(int);

		if (size_t(output_size*sizeof(float) + i) > net_config.size()) {
			throw std::runtime_error("corrupt config.");
		}
		
		Matrix	weight(input_size, output_size);
		for (int k = 0; k < input_size; k++) {
			std::memcpy(weight.m[k].data(), &net_config[i], output_size*sizeof(float));
			i += output_size*sizeof(float);
		}
		
		Matrix	bias = Matrix(1, output_size);
		std::memcpy(bias.m[0].data(), &net_config[i], output_size*sizeof(float));
		i += output_size*sizeof(float);

		HiddenLayer	hidden_layer(input_size, output_size);
		hidden_layer.setWeight(weight);
		hidden_layer.setBias(bias);
		this->hidden_layers.push_back(hidden_layer);

		input_size = output_size;
	}

	/* retrieving output layer */
	int output_size;
	std::memcpy(&output_size, &net_config[i], sizeof(int));
	i += sizeof(int);

	Matrix	weight(input_size, output_size);
	for (int k = 0; k < input_size; k++) {
		std::memcpy(weight.m[k].data(), &net_config[i], output_size*sizeof(float));
		i += output_size*sizeof(float);
	}
	
	Matrix	bias(1, output_size);
	std::memcpy(bias.m[0].data(), &net_config[i], output_size*sizeof(float));
	i += output_size*sizeof(float);

	OutputLayer	tmp_output_layer(input_size, output_size);
	tmp_output_layer.setWeight(weight);
	tmp_output_layer.setBias(bias);
	this->output_layer = tmp_output_layer;
	_name = std::string(filename).substr(std::string(filename).find_last_of("/\\") + 1, 
		std::string(filename).find_last_of(".") - std::string(filename).find_last_of("/\\") - 1);
}

CNN::~CNN() { }

void	CNN::saveConfigJson( const char *filename ) const {
    std::ofstream file(filename, std::ios::out);
    
    if (file.is_open()) {
		file << "{\n";
   		file << "\"size\": " << this->_connected_size << "," << std::endl;
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
