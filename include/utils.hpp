
#ifndef UTILS_HPP
# define UTILS_HPP

# include <sys/time.h>
# include <fstream>

# include "convolution.hpp"
# include "Matrix.hpp"

double						ft_get_time(void);
std::vector<Tensor4D>		get_mnist_batch( const char *filename, int batch_size, int iterations );
std::vector<Matrix>			get_mnist_labels( const char *filename, int batch_size, int iterations );
std::vector<unsigned char>	read_binary_file(const char *filename, size_t size);
std::streamsize 			get_file_size(const char *filename);
void						write_image(const char* filename, int width, int height, int channels, const void* data);
unsigned char				*load_image(const char *filename, int *width, int *height, int *channels, int force_channels);
void						free_image(unsigned char *data);


#endif
