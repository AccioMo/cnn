
#include "utils.hpp"

#define STBI_NO_GIF
#define STBI_NO_HDR
#define STBI_NO_BMP
#define STBI_NO_PSD
#define STBI_NO_TGA
#define STBI_NO_PIC
#define STBI_NO_PNM

# define STB_IMAGE_IMPLEMENTATION
# include "stb_image.h"
# include "stb_image_write.h"

double	ft_get_time(void) {
	struct timeval	counter;

	gettimeofday(&counter, NULL);
	return (counter.tv_sec * 1000.0 + counter.tv_usec / 1000.0);
}

std::vector<Matrix>	get_input_batch( const char *filename ) {
    std::ifstream file(filename, std::ios::binary);
	std::vector<Matrix>	inputs;

	int	size = TRAIN_SIZE / BATCH_SIZE;
    if (file.is_open()) {

		for (int k = 0; k < size; k++) {
			Matrix	batch_matrix(BATCH_SIZE, IMAGE_SIZE);

			std::vector<unsigned char> data(BATCH_SIZE * IMAGE_SIZE);
			file.read(reinterpret_cast<char *>(data.data()), BATCH_SIZE * IMAGE_SIZE);

			for (int i = 0; i < BATCH_SIZE; i++) {
				for (int j = 0; j < IMAGE_SIZE; j++) {
					batch_matrix.m[i][j] = static_cast<double>(data[(i * IMAGE_SIZE) + j]);
				}
			}
			inputs.push_back(batch_matrix);
		}
		file.close();
    } else {
        std::cerr << "Error opening file: " << filename << std::endl;
    }
	return (inputs);
}

std::vector<Matrix>	get_input_labels( const char *filename ) {
    std::ifstream file(filename, std::ios::binary);
	std::vector<Matrix>	outputs;

	int	size = TRAIN_SIZE / BATCH_SIZE;
    if (file.is_open()) {

		for (int k = 0; k < size; k++) {
			Matrix	batch_matrix(BATCH_SIZE, POSSIBILE_OUTPUTS);

			std::vector<unsigned char> data(BATCH_SIZE);
			file.read(reinterpret_cast<char *>(data.data()), BATCH_SIZE);

			for (int i = 0; i < BATCH_SIZE; i++) {
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
        throw std::runtime_error("Unable to open file");
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
    }
}
