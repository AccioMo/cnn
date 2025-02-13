CLANG = g++
FLAGS = -Wall -Wextra -Werror -Wshadow -std=c++11
DEBUG_FLAGS = -g -fsanitize=address
OPTIMIZATION_FLAGS = -flto -O3
HEADERS = math.hpp utils.hpp convolution.hpp Matrix.hpp \
			NeuralNetwork.hpp BaseLayer.hpp HiddenLayer.hpp \
			OutputLayer.hpp ConvLayer.hpp stb_image.h stb_image_write.h
STRUCTURE_FILES = NeuralNetworkInit.cpp NeuralNetworkMain.cpp CNNInit.cpp CNNMain.cpp
LAYER_FILES = BaseLayer.cpp HiddenLayer.cpp OutputLayer.cpp ConvLayer.cpp
MATH_FILES = math.cpp Matrix.cpp 
UTILS_FILES = utils.cpp convolution.cpp
MAIN_FILE = main.cpp

INCLUDE_DIR = include/
SRC_DIR = src/
BASE_DIR = $(SRC_DIR)base/
MATH_DIR = $(SRC_DIR)math/
UTILS_DIR = $(SRC_DIR)utils/
LAYER_DIR = $(BASE_DIR)Layers/
STRUCTURE_DIR = $(BASE_DIR)Structures/
OBJ_DIR = obj/
DEBUG_OBJ_DIR = debug_obj/

INCLUDES = $(addprefix $(INCLUDE_DIR), $(HEADERS))
FILES = $(STRUCTURE_FILES) $(LAYER_FILES) $(MATH_FILES) $(UTILS_FILES) $(MAIN_FILE)
OBJ_FILES = $(addprefix $(OBJ_DIR), $(FILES:.cpp=.opp))
DEBUG_OBJ_FILES = $(addprefix $(DEBUG_OBJ_DIR), $(FILES:.cpp=_debug.opp))
NAME_DEBUG = cnn_debug
NAME = cnn

all: FLAGS += $(OPTIMIZATION_FLAGS)
all: $(OBJ_DIR) $(NAME)

# ==== RELEASE ==== #
$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

$(NAME): $(OBJ_FILES) $(INCLUDES)
	$(CLANG) $(FLAGS) $(OBJ_FILES) -o $(NAME)

$(OBJ_DIR)%.opp: $(LAYER_DIR)%.cpp $(INCLUDES)
	$(CLANG) $(FLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(OBJ_DIR)%.opp: $(STRUCTURE_DIR)%.cpp $(INCLUDES)
	$(CLANG) $(FLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(OBJ_DIR)%.opp: $(MATH_DIR)%.cpp $(INCLUDES)
	$(CLANG) $(FLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(OBJ_DIR)%.opp: $(UTILS_DIR)%.cpp $(INCLUDES)
	$(CLANG) $(FLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(OBJ_DIR)%.opp: %.cpp $(INCLUDES)
	$(CLANG) $(FLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# ==== DEBUG ==== #
$(DEBUG_OBJ_DIR):
	@mkdir -p $(DEBUG_OBJ_DIR)

$(NAME_DEBUG): $(DEBUG_OBJ_FILES) $(INCLUDES)
	$(CLANG) $(FLAGS) $(DEBUG_OBJ_FILES) -o $(NAME_DEBUG)

$(DEBUG_OBJ_DIR)%_debug.opp: $(BASE_DIR)%.cpp $(INCLUDES)
	$(CLANG) $(FLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(DEBUG_OBJ_DIR)%_debug.opp: $(MATH_DIR)%.cpp $(INCLUDES)
	$(CLANG) $(FLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(DEBUG_OBJ_DIR)%_debug.opp: $(UTILS_DIR)%.cpp $(INCLUDES)
	$(CLANG) $(FLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(DEBUG_OBJ_DIR)%_debug.opp: %.cpp $(INCLUDES)
	$(CLANG) $(FLAGS) -I$(INCLUDE_DIR) -c $< -o $@

debug: FLAGS += $(DEBUG_FLAGS)
debug: $(DEBUG_OBJ_DIR) $(NAME_DEBUG)

clean:
	rm -f $(OBJ_FILES)
	rm -f $(DEBUG_OBJ_FILES)

fclean: clean
	rm -f $(NAME) $(NAME_DEBUG)

re: fclean all

.PHONY: all debug clean fclean re
