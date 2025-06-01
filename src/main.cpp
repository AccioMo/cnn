
#include "CNN.hpp"
#include "Matrix.hpp"

static int	train_network( std::string arch_file, std::string output_file ) {

	std::ifstream	config_file(arch_file);
	if (!config_file.is_open()) {
		std::cerr << "error: couldn't open file: architectures/v0-000.json" << std::endl;
		return (1);
	}

	nlohmann::json	arch;
	config_file >> arch;
	config_file.close();

	CNN	network(arch);
	network.trainOnFile("mnist/mnist_train_images.bin",
		"mnist/mnist_train_labels.bin",
		output_file.c_str());
	return (0);

}

static int test_network( std::string config_file ) {

	CNN	network(config_file.c_str());
	network.testOnFile("mnist/mnist_test_images.bin", "mnist/mnist_test_labels.bin");
	return (0);

}

static int run_on_image( std::string image_path, std::string config_file ) {

	CNN	network(config_file.c_str());
	network.runOnImage(image_path.c_str());
	return (0);

}

int main( int ac, char **av ){

	if (ac < 2) {
		std::cerr << "usage: " << av[0] << " <train|test|image_path>" << std::endl;
		  std::cerr << "       " << av[0] << " train [architecture_file] [output_file]" << std::endl;
		    std::cerr << "       " << av[0] << " test <config_file>" << std::endl;
			  std::cerr << "       " << av[0] << " <image_path> [config_file]" << std::endl;
		return (1);
	}
	
	try {
		if (av[1] == std::string("train")) {
			return (train_network(ac > 2 ? av[2] : "architectures/v0-000.json",
					ac > 3 ? av[3] : "dials.bin"));
		} else if (av[1] == std::string("test")) {
			return (test_network(ac > 2 ? av[2] : "dials.bin"));
		} else {
			return (run_on_image(av[1], ac > 2 ? av[2] : "dials.bin"));
		}
	} catch (const std::exception &e) {
		std::cerr << "uh oh: " << e.what() << std::endl;
		return (1);
	}
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
	// std::cout << "RUNNING ON [ 8 ]" << std::endl;
	// network.runOnImage("wild_images/sample_8.png");
	// std::cout << "RUNNING ON [ 9 ]" << std::endl;
	// network.runOnImage("wild_images/sample_9.png");
	// std::cout << "RUNNING ON [ 9 ]" << std::endl;
	// network.runOnImage("wild_images/sample_10.png");

    return (0);
}
