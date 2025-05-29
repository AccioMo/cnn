
#include "utils.hpp"

#define STBI_NO_GIF
#define STBI_NO_HDR
#define STBI_NO_BMP
#define STBI_NO_PSD
#define STBI_NO_TGA
#define STBI_NO_PIC
#define STBI_NO_PNM

#define STB_IMAGE_IMPLEMENTATION
# include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
# include "stb_image_write.h"

float	ft_get_time(void) {
	struct timeval	counter;

	gettimeofday(&counter, NULL);
	return (counter.tv_sec * 1000.0 + counter.tv_usec / 1000.0);
}

std::vector<Tensor4D>	get_mnist_batch( const char *filename, int batch_size, int iterations ) {
	int img_width = 28;
	int img_height = 28;
	int img_channels = 1;
	int img_size = img_width * img_height * img_channels;

    std::ifstream file(filename, std::ios::binary);
	std::vector<Tensor4D>	inputs;

    if (file.is_open()) {

		for (int b = 0; b < iterations; b++) {
			Tensor4D	batch_tensor(batch_size, img_width, img_height, img_channels);

			std::vector<unsigned char> data(batch_size * img_size);
			file.read(reinterpret_cast<char *>(data.data()), batch_size * img_size);

			for (int i = 0; i < batch_size; i++) {
				for (int j = 0; j < img_height; j++) {
					for (int k = 0; k < img_width; k++) {
						for (int h = 0; h < img_channels; h++) {
							int index = (i*img_size) + j*(img_width*img_channels) + k*img_channels + h;
							batch_tensor(i, j, k, h) = static_cast<float>(data[index]);
						}
					}
				}
			}
			inputs.push_back(batch_tensor);
		}
		file.close();
    } else {
        std::cerr << "Error opening file: " << filename << std::endl;
    }
	return (inputs);
}

std::vector<Matrix>	get_mnist_labels( const char *filename, int batch_size, int iterations ) {
    std::ifstream file(filename, std::ios::binary);
	std::vector<Matrix>	outputs;
	int possible_outputs = 10;

    if (file.is_open()) {

		for (int k = 0; k < iterations; k++) {
			Matrix	batch_matrix(batch_size, possible_outputs);

			std::vector<unsigned char> data(batch_size);
			file.read(reinterpret_cast<char *>(data.data()), batch_size);

			for (int i = 0; i < batch_size; i++) {
				batch_matrix.m[i][static_cast<int>(data[i])] = 1.0;
			}
			outputs.push_back(batch_matrix);
		}
		file.close();
    } else {
        std::cout << "Error opening file: " << filename << std::endl;
    }
	return (outputs);
}

std::vector<unsigned char> read_binary_file(const char *filename, size_t size) {
    std::ifstream file(filename, std::ios::binary);
    
    if (file.is_open()) {
   		std::vector<unsigned char> data(size);
        file.read(reinterpret_cast<char *>(data.data()), size);
        file.close();
	    return data;
    } else {
        std::cerr << "Error opening file: " << filename << std::endl;
    }

	std::vector<unsigned char> none;
    return none;
}

std::streamsize get_file_size(const char *filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (file.is_open()) {
        std::streamsize size = file.tellg();
        file.close();
        return size;
    } else {
        throw std::runtime_error("Unable to open file: ");
    }
}

unsigned char *load_image(const char *filename, int *width, int *height, int *channels, int force_channels) {

    unsigned char *data = stbi_load(filename, width, height, channels, force_channels);
    
    if (data == NULL) {
        fprintf(stderr, "Error: Could not load image '%s'\n", filename);
        exit(1);
    }

    return data;
}

void free_image(unsigned char *data) {
    stbi_image_free(data);
}

void write_image(const char* filename, int width, int height, int channels, const void* data) {
    const char* ext = strrchr(filename, '.');
    if (ext == NULL) {
        fprintf(stderr, "Error: No file extension found in '%s'\n", filename);
        return;
    }

    int success = 0;
    if (strcmp(ext, ".png") == 0) {
        success = stbi_write_png(filename, width, height, channels, data, width * channels);
    } else if (strcmp(ext, ".jpg") == 0 || strcmp(ext, ".jpeg") == 0) {
        success = stbi_write_jpg(filename, width, height, channels, data, 90);
    } else if (strcmp(ext, ".bmp") == 0) {
        success = stbi_write_bmp(filename, width, height, channels, data);
    } else if (strcmp(ext, ".tga") == 0) {
        success = stbi_write_tga(filename, width, height, channels, data);
    } else {
        fprintf(stderr, "Error: Unsupported file format '%s'\n", ext);
        return;
    }

    if (!success) {
        fprintf(stderr, "Error: Could not write image to '%s'\n", filename);
		perror("");
    }
}
