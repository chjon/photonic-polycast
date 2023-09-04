CWD = $(shell pwd)

CPP = g++
CPPFLAGS = -g -std=c++17 -Wall
# CPPFLAGS += -Wextra -Werror -Wshadow -Wconversion -Wunreachable-code

# Libraries
CPPFLAGS += -Ilib
LPPFLAGS += `libpng-config --ldflags` 

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
TARGET = ppcast

SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
DEPS = $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.d)

.PHONY: all clean

# Compile main program by default
all: $(TARGET)

clean:
	rm -rf $(BIN_DIR)
	rm -rf $(OBJ_DIR)

# Compile and link target binary
$(TARGET): $(OBJ_DIR) $(OBJS) $(BIN_DIR)
	$(CPP) -o $(BIN_DIR)/$(TARGET) $(OBJS) $(LPPFLAGS)

# Create directory for dependency and object files
$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

# Create directory for target binary
$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

# Compile objects
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CPP) -MM -MP -MT $(OBJ_DIR)/$*.o -MT $(OBJ_DIR)/$*.d $(CPPFLAGS) $< > $(OBJ_DIR)/$*.d
	$(CPP) $(CPPFLAGS) -o $@ -c $(SRC_DIR)/$*.cpp

-include $(DEPS)