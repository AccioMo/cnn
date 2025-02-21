
#include "ConvLayer.hpp"
#include "utils.hpp"

ConvLayer::ConvLayer( int kernel_size, int input_size, int output_size, int stride, int padding )
	: _stride(stride), _padding(padding) {
	double std_deviation = he_init(kernel_size * kernel_size * input_size);
	std::random_device					rd;
	std::mt19937						gen(rd());
	std::normal_distribution<>	dis(0.0, std_deviation);
	/* input_size is the number of channels.
	output_size is the number of filter outputs */
	this->_kernel = Tensor4D(kernel_size, kernel_size, input_size, output_size);
	for (int i = 0; i < kernel_size; i++) {
		for (int j = 0; j < kernel_size; j++) {
			for (int k = 0; k < input_size; k++) {
				for (int h = 0; h < output_size; h++) {
					this->_kernel(i, j, k, h) = dis(gen);
				}
			}
		}
	}
	this->_bias = Tensor4D(1, 1, 1, output_size);
	for (int h = 0; h < output_size; h++) {
		this->_bias(0, 0, 0, h) = 0.1;
	}
	this->_m = Tensor4D(kernel_size, kernel_size, input_size, output_size);
	this->_v = Tensor4D(kernel_size, kernel_size, input_size, output_size);
}

ConvLayer::~ConvLayer() { }

Tensor4D	&ConvLayer::feedforward( const Tensor4D &prev_outputs ) {
	static int round = 0;
	this->_z = convolve(prev_outputs, this->_kernel, this->_stride, this->_padding);
	/* `_z` is the convoluted output, feature map, 
	which i will pass through a max pooling or 
	average pooling to decrease the amount of 
	params the network has to process */

	int batch_size = this->_z.dimension(0);
	int	width = this->_z.dimension(1);
	int	height = this->_z.dimension(2);
	int	channels = this->_z.dimension(3);
	Eigen::array<long, 4>	dims = {batch_size, width, height, 1};
	Tensor4D	bias = this->_bias.broadcast(dims);
	this->_z = this->_z + bias;

	this->_a = ReLU(this->_z);

	int	og_width = prev_outputs.dimension(1);
	int	og_height = prev_outputs.dimension(2);
	std::vector<char> og_data(og_width * og_height);
	for (int y = 0; y < og_height; ++y) {
		for (int x = 0; x < og_width; ++x) {
			og_data[y * og_width + x] = static_cast<char>(prev_outputs(0, y, x, 0) * 255.0f);
		}
	}
	std::string og_filename = "outputs/original.png";
	write_image(og_filename.c_str(), og_width, og_height, 1, og_data.data());

	for (int c = 0; c < channels; ++c) {
		std::vector<char> channel_data(width * height);
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				channel_data[y * width + x] = static_cast<char>(this->_a(0, y, x, c) * 255.0f);
			}
		}

		std::string channel_filename = "outputs/" + std::to_string(round % 3) + "-channel_" + std::to_string(c) + ".png";
		write_image(channel_filename.c_str(), width, height, 1, channel_data.data());
	}
	round++;
	
	/* TODO: 
		this->_a = pooling(this->_a, "max");
		add average pooling function 
		and simple way to choose one */
	return (this->_a);
}

void	ConvLayer::backpropagation( const BaseLayer &next_layer ) {
	/*
		backpropagation in last conv layer, 
		which connects directly to a fully 
		connected layer, takes a Matrix 
		backpropagated error, calculates 
		its own error, and unflattens it
		into a 4D Tensor.
	*/
	Matrix	output = flatten(this->_z);
	Matrix	flat_error = next_layer.getError().dot(next_layer.getWeight().transpose()).hadamard_product(ReLU_derivative(output));
	int d1 = this->getOutput().dimension(0);
	int d2 = this->getOutput().dimension(1);
	int d3 = this->getOutput().dimension(2);
	int d4 = this->getOutput().dimension(3);
	this->_error = unflatten(flat_error, d1, d2, d3, d4);
}

void	ConvLayer::backpropagation( const ConvLayer &next_layer ) {
	/*
		formula for backpropagation in a convolutional:
			δ = (δ^{next} * W.rot(180)) ⊙ f'(z)

		`δ` is the error, `δ^{next}` is the error of the 
		next layer, `W` is the kernel of the current layer,
		⊙ is element-wise multiplication, `f'` is the
		derivative of the activation function, and `z` is
		the pre-activation output of the current layer.
	*/
	this->_error = rev_convolve(next_layer.getError(), flip(next_layer.getKernel()), next_layer.getStride(), next_layer.getPadding()) * ReLU_derivative(this->_z);
}

void	ConvLayer::update( const Tensor4D &inputs, 
							float learning_rate, 
							int timestep, 
							float l2_reg, 
							float beta1, 
							float beta2 ) {

	this->_gradient = gradient_convolve(inputs, this->_error, this->_stride, this->_padding);
	(void)beta1;
	(void)beta2;
	(void)timestep;
	(void)l2_reg;

	/* ------------------------------------------------------------------- 
	Tensor4D	kernel_gradient = this->_gradient + (this->_kernel * l2_reg);

	this->_m = this->_m * beta1 + kernel_gradient * (1.0 - beta1);
	this->_v = this->_v * beta2 + kernel_gradient.square() * (1.0 - beta2);

	Tensor4D	m_hat = this->_m / (1.0 - std::pow(beta1, timestep));
	Tensor4D	v_hat = this->_v / (1.0 - std::pow(beta2, timestep));
	this->_gradient = m_hat / v_hat.sqrt() + 1e-8;
	 ------------------------------------------------------------------- */

	this->_kernel = this->_kernel - (this->_gradient * learning_rate);

	Eigen::array<int, 3> sum_dims = {0, 1, 2};
	Eigen::array<long, 4>	dims = {1, 1, 1, this->_bias.dimension(3)};
	Tensor4D	bias_gradient = this->_error.sum(sum_dims).reshape(dims);
	this->_bias = this->_bias - (bias_gradient * (float)learning_rate);
}

Tensor4D	ConvLayer::getKernel( void ) const {
	return (this->_kernel);
}

Tensor4D	ConvLayer::getBias( void ) const {
	return (this->_bias);
}

Tensor4D	ConvLayer::getOutput( void ) const {
	return (this->_a);
}

Tensor4D	ConvLayer::getError( void ) const {
	return (this->_error);
}

int			ConvLayer::getStride( void ) const {
	return (this->_stride);
}

int			ConvLayer::getPadding( void ) const {
	return (this->_padding);
}
