CWD = $(shell pwd)

CPP = g++
CPPFLAGS = -g -std=c++17 -MMD -MP
CPPFLAGS += -O3
CPPFLAGS += -Wall -Wextra -Werror -Wshadow -Wconversion -Wunreachable-code
CPPFLAGS += -isystem lib
LDFLAGS  = `libpng-config --ldflags` 

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
TARGET = ppcast

SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
DEPS = $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.d)

.PHONY: all clean info

# Compile main program by default
all: info $(TARGET)

clean:
	rm -rf $(BIN_DIR)
	rm -rf $(OBJ_DIR)

info:
	@echo "Compiling with flags $(CPPFLAGS)"

# Compile and link target binary
$(TARGET): $(OBJ_DIR) $(OBJS) $(BIN_DIR)
	$(CPP) -o $(BIN_DIR)/$(TARGET) $(OBJS) $(LDFLAGS)

# Create directory for dependency and object files
$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

# Create directory for target binary
$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

# Compile objects
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "$(SRC_DIR)/$*.cpp -> $@"
	@$(CPP) $(CPPFLAGS) -o $@ -c $(SRC_DIR)/$*.cpp

-include $(DEPS)