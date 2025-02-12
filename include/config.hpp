
#ifndef CONFIG_HPP
# define CONFIG_HPP

const int EPOCHS = 20;
const int IMAGE_SIZE = 28 * 28;
const int POSSIBILE_OUTPUTS = 10;

const int TRAIN_SIZE = 60000;
const int TEST_SIZE = 10000;

/* for SGD (stochastic gradient 
descent) using sequential data 
sampling */
const int BATCH_SIZE = 128;

const double INPUT_MIN = 0.0;
const double INPUT_MAX = 255.0;

const double OUTPUT_MIN = 0.0;
const double OUTPUT_MAX = 1.0;

#endif
