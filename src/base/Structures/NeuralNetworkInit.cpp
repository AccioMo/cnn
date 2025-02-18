
#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork( void )
 { }

NeuralNetwork::NeuralNetwork( std::vector<int> nodes,
							double learning_rate,
							double l2_lambda,
							double beta1,
							double beta2 )
	: _size(nodes.size()),
	_learning_rate(learning_rate),
	_l2_lambda(l2_lambda),
	_beta1(beta1),
	_beta2(beta2) {
	if (_size < 2)
		return ;
	this->hidden_layers.reserve(_size - 2);
	for (int i = 0; i < _size - 2; i++) {
		this->hidden_layers.emplace_back(HiddenLayer(nodes[i], nodes[i + 1]));
	}
	this->output_layer = OutputLayer(nodes[_size - 2], nodes[_size - 1]);
}

NeuralNetwork::NeuralNetwork( std::vector<HiddenLayer> hidden_layers,
							OutputLayer output_layer,
							double learning_rate,
							double l2_lambda,
							double beta1,
							double beta2 )
	: _size(hidden_layers.size() + 2),
	_learning_rate(learning_rate),
	_l2_lambda(l2_lambda),
	_beta1(beta1),
	_beta2(beta2),
	hidden_layers(hidden_layers),
	output_layer(output_layer) { }

NeuralNetwork::NeuralNetwork( const char *filename ) {

	std::cout << "loading network from file: NN" << filename << std::endl;
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

NeuralNetwork::~NeuralNetwork() { }

void	NeuralNetwork::saveConfigJson( const char *filename ) const {
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

void	NeuralNetwork::saveConfigBin(const char *filename) const {
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
