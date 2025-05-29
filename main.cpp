
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

	CNN	network(arch);
	std::cout << "network created" << std::endl;
	network.trainOnFile("mnist/mnist_train_images.bin", 
						"mnist/mnist_train_labels.bin", 
						"conf.bin");
						
	// std::cout << "RUNNING ON [ 0 ]" << std::endl;
	// network.runOnImage("wild_images/sample_0.png");
	// std::cout << "RUNNING ON [ 1 ]" << std::endl;
	// network.runOnImage("wild_images/sample_1.png");
	// std::cout << "RUNNING ON [ 2 ]" << std::endl;;
	// network.runOnImage("wild_images/sample_2.jpg");
	// std::cout << "RUNNING ON [ 3 ]" << std::endl;
	// network.runOnImage("wild_images/sample_3.png");
	// std::cout << "RUNNING ON [ 4 ]" << std::endl;
	// network.runOnImage("wild_images/sample_4.png");
	// std::cout << "RUNNING ON [ 5 ]" << std::endl;
	// network.runOnImage("wild_images/sample_5.png");
	// std::cout << "RUNNING ON [ 6 ]" << std::endl;
	// network.runOnImage("wild_images/sample_6.png");
	// std::cout << "RUNNING ON [ 7 ]" << std::endl;
	// network.runOnImage("wild_images/sample_7.png");
	std::cout << "RUNNING ON [ 8 ]" << std::endl;
	network.runOnImage("wild_images/sample_8.png");
	// std::cout << "RUNNING ON [ 9 ]" << std::endl;
	// network.runOnImage("wild_images/sample_9.png");
	// std::cout << "RUNNING ON [ 9 ]" << std::endl;
	// network.runOnImage("wild_images/sample_10.png");

	// std::cout << "###################################" << std::endl;
	// std::cout << network << std::endl;
	// std::cout << "###################################" << std::endl;
	network = CNN("conf.bin");
	std::cout << "RUNNING ON [ 7 ]" << std::endl;
	network.runOnImage(av[1]);
	// std::cout << network << std::endl;
	// std::cout << "###################################" << std::endl;
	// network.trainOnFile("mnist/mnist_train_images.bin", 
	// 					"mnist/mnist_train_labels.bin", 
	// 					"config.bin");
	// network.runOnImage(av[1]);

	// network.testOnFile("mnist/mnist_test_images.bin", 
	// 					"mnist/mnist_test_labels.bin");


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
