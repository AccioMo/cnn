
#ifndef UTILS_HPP
# define UTILS_HPP

# include <sys/time.h>
# include <fstream>

# include "config.hpp"
# include "convolution.hpp"
# include "Matrix.hpp"

double						ft_get_time(void);
std::vector<Tensor4D>		get_mnist_batch( const char *filename );
std::vector<Matrix>			get_mnist_labels( const char *filename );
std::vector<unsigned char>	read_binary_file(const char *filename, size_t size);
std::streamsize 			get_file_size(const char *filename);
void						write_image(const char* filename, int width, int height, int channels, const void* data);
unsigned char				*load_image(const char *filename, int *width, int *height, int *channels, int force_channels);
void						free_image(unsigned char *data);


#endif
