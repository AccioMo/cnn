
#include "CNN.hpp"
#include "Matrix.hpp"
#include <vector>

int main( int ac, char **av )
{
	(void)ac;
	(void)av;

	std::string config = "configs/20i-92.14%.bin";

	CNN	network = CNN(std::vector<int>{7, 5, 3},
						std::vector<int>{784, 128, 64, 10},
						0.0001, 
						0.001, 
						0.9, 
						0.999);
	network.trainOnFile("mnist/mnist_train_images.bin", 
						"mnist/mnist_train_labels.bin", 
						config.c_str());

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
