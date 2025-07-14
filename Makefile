CXX = c++
CXXFLAGS = -Wall -Wextra -Werror -std=c++17 -MMD -MP
DEBUG_FLAGS = -g -fsanitize=address -fsanitize=undefined -fsanitize=leak
OPTIMIZATION_FLAGS = -O3

SRC_DIR = src
DEBUG_BUILD_DIR = debug_build
BUILD_DIR = build

SRC_FILES = $(wildcard src/*.cpp) \
			$(wildcard src/base/*.cpp) \
			$(wildcard src/base/Layers/*.cpp) \
			$(wildcard src/base/Network/*.cpp) \
			$(wildcard src/math/*.cpp) \
			$(wildcard src/utils/*.cpp)

INCLUDE = -Iinclude
CXXFLAGS += $(INCLUDE)

OBJ_FILES = $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(SRC_FILES))
DEBUG_OBJ_FILES = $(addprefix $(DEBUG_BUILD_DIR), $(SRC_FILES:.cpp=_debug.o))

# Targets
TARGET_DEBUG = cnn_debug
TARGET = cnn

# Dependencies
DEPS = $(OBJ_FILES:.o=.d) $(DEBUG_OBJ_FILES:.o=.d)

# ==== RELEASE ==== #

all: CXXFLAGS += $(OPTIMIZATION_FLAGS)
all: $(BUILD_DIR) $(TARGET)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(TARGET): $(OBJ_FILES)
	$(CXX) $(CXXFLAGS) $^ -o $@
	
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run
run: all
	./$(TARGET)

clean:
	rm -f $(OBJ_FILES)
	rm -f $(DEBUG_OBJ_FILES)

fclean: clean
	rm -f $(TARGET) $(TARGET_DEBUG)

re: fclean all

# ==== DEBUG ==== #

gdb: debug
	gdb $(TARGET_DEBUG)

d: debug

debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: $(DEBUG_BUILD_DIR) $(TARGET_DEBUG)

$(DEBUG_BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(TARGET_DEBUG): $(OBJ_FILES)
	$(CXX) $(CXXFLAGS) $^ -o $@
	
$(DEBUG_BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: all debug clean fclean re

# =============== #

-include $(DEPS)
