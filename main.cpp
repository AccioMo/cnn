
#include "CNN.hpp"
#include "Matrix.hpp"

int main( int ac, char **av )
{
	(void)ac;
	(void)av;

	std::ifstream	config_file("architectures/v0-000.json");
	if (!config_file.is_open()) {
		std::cerr << "Error opening file: architectures/v0-000.json" << std::endl;
		return (1);
	}
	nlohmann::json	arch;
	config_file >> arch;
	config_file.close();

	CNN	network("config.bin");
	std::cout << "network created" << std::endl;
	network.trainOnFile("mnist/mnist_train_images.bin", 
						"mnist/mnist_train_labels.bin", 
						"config-1.bin");

	/*
	if (ac > 1)
	{
		if (std::string(av[1]) == "train") {
			if (ac > 2) {
				CNN	network(av[2]);
				network.trainOnFile("mnist/mnist_train_images.bin", 
									"mnist/mnist_train_labels.bin", 
									config.c_str());
			} else {
				CNN	network = CNN(std::vector<int>{7, 5, 3},
									std::vector<int>{128, 64, 10},
									0.01, 
									0.001, 
									0.9, 
									0.999);
				network.trainOnFile("mnist/mnist_train_images.bin", 
									"mnist/mnist_train_labels.bin", 
									config.c_str());
	// 		}
		} else if (std::string(av[1]) == "test") {
			CNN	network = CNN(config.c_str());
			network.testOnFile("mnist/mnist_test_images.bin", "mnist/mnist_test_labels.bin");
		} else {
			if (ac > 2) {
				CNN	network(av[2]);
				network.runOnImage(av[1]);
			} else {
				CNN	network = CNN(config.c_str());
				network.runOnImage(av[1]);
			}
		}
	} else {
		std::cerr << "usage: ./cnn <image_path>" << std::endl;
	}
	*/

    return (0);
}
