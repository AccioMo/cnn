
#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork( void ) : NeuralNetwork(0, (int []){})
 { }

NeuralNetwork::NeuralNetwork( int size,
							int *nodes,
							double learning_rate,
							double l2_lambda,
							double beta1,
							double beta2 )
	: _size(size),
	_learning_rate(learning_rate),
	_l2_lambda(l2_lambda),
	_beta1(beta1),
	_beta2(beta2) {
	if (size < 2)
		return ;
	this->output_layer = OutputLayer(nodes[size - 2], nodes[size - 1]);
	this->hidden_layers.reserve(size - 2);
	for (int i = 0; i < size - 2; i++) {
		this->hidden_layers.emplace_back(HiddenLayer(i, nodes[i], nodes[i + 1]));
	}
}

NeuralNetwork::NeuralNetwork( const char *filename ) : NeuralNetwork(0, (int []){}) {

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

		Matrix	weights = Matrix(config_nodes[k], neurons);
		for (int j = 0; j < config_nodes[k]; j++) {
			for (int l = 0; l < neurons; l++) {
				std::memcpy(&weights.m[j][l], &mnist_train_images[i], sizeof(double));
				i += sizeof(double);
			}
		}
		Matrix	biases = Matrix(1, neurons);

		for (int j = 0; j < neurons; j++) {
			std::memcpy(&biases.m[0][j], &mnist_train_images[i], sizeof(double));
			i += sizeof(double);
		}

		HiddenLayer	hidden_layer(k, config_nodes[k], neurons);
		hidden_layer.setWeights(weights);
		hidden_layer.setBiases(biases);
		hidden_layer.setSize(neurons);
		this->hidden_layers.push_back(hidden_layer);
	}

	int		neurons = config_nodes[this->_size - 1];

	Matrix	weights = Matrix(config_nodes[this->_size - 2], neurons);

	for (int j = 0; j < config_nodes[this->_size - 2]; j++) {
		for (int l = 0; l < neurons; l++) {
			std::memcpy(&weights.m[j][l], &mnist_train_images[i], sizeof(double));
			i += sizeof(double);
		}
	}

	Matrix	biases = Matrix(1, neurons);
	
	for (int j = 0; j < neurons; j++) {
		std::memcpy(&biases.m[0][j], &mnist_train_images[i], sizeof(double));
		i += sizeof(double);
	}

	OutputLayer	new_output_layer(config_nodes[this->_size - 2], neurons);
	new_output_layer.setWeights(weights);
	new_output_layer.setBiases(biases);
	new_output_layer.setSize(neurons);
	this->output_layer = new_output_layer;
}

NeuralNetwork::~NeuralNetwork() { }

void	NeuralNetwork::saveConfigJson( const char *filename ) const {
    std::ofstream file(filename, std::ios::out);
    
    if (file.is_open()) {
		file << "{\n";
   		file << "\"size\": " << this->_size << "," << std::endl;
		file << "\"learning_rate\": " << this->_learning_rate << ", " << std::endl;
		file << "\"layers\": [" << this->hidden_layers[0].getWeights().rows() << ", ";
		for (auto &layer_neurons : this->hidden_layers)
			file << layer_neurons.getSize() << ", ";
		file << this->output_layer.getSize();
		file << "], " << std::endl;

		file << "\"hidden_layers\": [\n";
		for (auto &layer : this->hidden_layers) {
			file << "{" << std::endl;
			file << "\"weights\": " << layer.getWeights() << ", " << std::endl;
			file << "\"biases\": " << layer.getBiases() << std::endl;
			file << "}";
			if (&layer != &this->hidden_layers.back())
				file << std::endl << "," << std::endl;
		}
		file << "],\n";
		file << "\"output_layer\": {" << std::endl;
		file << "\"weights\": " << this->output_layer.getWeights() << ", " << std::endl;
		file << "\"biases\": " << this->output_layer.getBiases() << std::endl;
		file << "}";
		file << "}" << std::endl;
        file.close();
    } else {
        std::cerr << "Error opening file: " << filename << std::endl;
    }
}

void	NeuralNetwork::saveConfigBin(const char *filename) const {
	std::ofstream file(filename, std::ios::binary);

    if (file.is_open()) {

        file.write(reinterpret_cast<const char *>(&this->_size), sizeof(this->_size));

        file.write(reinterpret_cast<const char *>(&this->_learning_rate), sizeof(this->_learning_rate));

        int rows = this->hidden_layers[0].getWeights().rows();
        file.write(reinterpret_cast<const char *>(&rows), sizeof(rows));

        for (const auto &layer : this->hidden_layers) {
            int size = layer.getSize();
            file.write(reinterpret_cast<const char *>(&size), sizeof(size));
        }

        int output_size = this->output_layer.getSize();
        file.write(reinterpret_cast<const char *>(&output_size), sizeof(output_size));

        for (const auto &layer : this->hidden_layers) {
			for (int i = 0; i < layer.getWeights().rows(); i++) {
				file.write(reinterpret_cast<const char *>(layer.getWeights().m[i].data()), layer.getWeights().m[i].size() * sizeof(double));
			}
			file.write(reinterpret_cast<const char *>(layer.getBiases().m[0].data()), layer.getBiases().m[0].size() * sizeof(double));
        }

        for (int i = 0; i < this->output_layer.getWeights().rows(); i++) {
			file.write(reinterpret_cast<const char *>(this->output_layer.getWeights().m[i].data()), this->output_layer.getWeights().m[i].size() * sizeof(double));
		}
		file.write(reinterpret_cast<const char *>(this->output_layer.getBiases().m[0].data()), this->output_layer.getBiases().m[0].size() * sizeof(double));

        file.close();
    } else {
        std::cerr << "Error opening file: " << filename;
    }
}
