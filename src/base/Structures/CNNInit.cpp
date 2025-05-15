
#include "CNN.hpp"

CNN::CNN( nlohmann::json arch ) : NeuralNetwork() {
	_conv_size = 0;
	_connected_size = 0;
	_learning_rate = arch["learning_rate"].get<float>();
	_l2_lambda = arch["l2_reg"].get<float>();
	_batch_size = arch["batch_size"].get<int>();
	_iterations = arch["iterations"].get<int>();
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
		input_width = (input_width - 2) / 2 + 1;
		input_height = (input_height - 2) / 2 + 1;
		channels = filters;
		std::cout << "input: " << input_width << "x" << input_height << "x" << channels << std::endl;
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
	_connected_size++;
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

	std::cout << "Loading network from " << filename << "..." << std::endl;
	std::streamsize size = get_file_size(filename);
	std::vector<unsigned char> net_config = read_binary_file(filename, (size_t)size);
	
	int	i = 0;
	
	/* retrieving sizes */
	std::memcpy(&this->_conv_size, &net_config[i], sizeof(int));
	i += sizeof(int);

	std::memcpy(&this->_connected_size, &net_config[i], sizeof(int));
	i += sizeof(int);

	/* retrieving params */
	std::memcpy(&this->_learning_rate, &net_config[i], sizeof(float));
	i += sizeof(float);

	std::memcpy(&this->_epochs, &net_config[i], sizeof(int));
	i += sizeof(int);

	std::memcpy(&this->_batch_size, &net_config[i], sizeof(int));
	i += sizeof(int);

	std::memcpy(&this->_l2_lambda, &net_config[i], sizeof(float));
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
		char *kernel_data = new char[total_size*sizeof(float)];
		std::memcpy(kernel_data, &net_config[i], total_size*sizeof(float));
		i += total_size*sizeof(float);
		Eigen::TensorMap<Tensor4D> kernel((float *)kernel_data, kernel_size, kernel_size, input, filters);
		ConvLayer conv_layer(kernel_size, input, filters, stride, padding);
		conv_layer.setKernel(kernel);
		std::cout << conv_layer.getBias() << std::endl;
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
		
		Matrix	weight(input_size, output_size);
		for (int k = 0; k < input_size; k++) {
			std::memcpy(weight.m[k].data(), &net_config[i], output_size*sizeof(double));
			i += output_size*sizeof(double);
		}
		
		Matrix	bias = Matrix(1, output_size);
		std::memcpy(bias.m[0].data(), &net_config[i], output_size*sizeof(double));
		i += output_size*sizeof(double);

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
		std::memcpy(weight.m[k].data(), &net_config[i], output_size*sizeof(double));
		i += output_size*sizeof(double);
	}
	
	Matrix	bias(1, output_size);
	std::memcpy(bias.m[0].data(), &net_config[i], output_size*sizeof(double));
	i += output_size*sizeof(double);

	OutputLayer	tmp_output_layer(input_size, output_size);
	tmp_output_layer.setWeight(weight);
	tmp_output_layer.setBias(bias);
	this->output_layer = tmp_output_layer;
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
