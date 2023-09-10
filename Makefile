CWD = $(shell pwd)

DEBUGMODE      = -g
LANGVERSION    = -std=c++17
DEPGENERATION  = -MMD -MP
OPTIMIZATION   = -O3
SYSTEMINCLUDES = lib /usr/local/cuda-12.2/bin/../targets/x86_64-linux/include/
COMPILERFLAGS  = $(DEBUGMODE) $(LANGVERSION) $(DEPGENERATION) $(OPTIMIZATION) $(addprefix -isystem ,$(SYSTEMINCLUDES))

CPP = g++
CPPWARNS = -Wall -Wextra -Werror -Wshadow -Wconversion -pedantic
CPPFLAGS = $(COMPILERFLAGS) $(CPPWARNS)

# CPP = nvcc
# CPPFLAGS = $(COMPILERFLAGS)

CU = nvcc
CUFLAGS = $(COMPILERFLAGS) --relocatable-device-code=true

LDFLAGS = `libpng-config --ldflags`

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
TARGET = ppcast

CPPSRCS = $(wildcard $(SRC_DIR)/*.cpp)
CUSRCS  = $(wildcard $(SRC_DIR)/*.cu)
CPPOBJS = $(CPPSRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
CUOBJS  = $(CUSRCS:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
DEPS    = $(CPPSRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.d) $(CUSRCS:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.d)

.PHONY: all clean info

# Compile main program by default
all: $(TARGET)

clean:
	rm -rf $(BIN_DIR)
	rm -rf $(OBJ_DIR)

info: $(CPPSRCS) $(CUSRCS)
	@echo "Compiling with flags: $(COMPILERFLAGS)"
	@echo "Using C++ warnings: $(CPPWARNS)"

# Compile and link target binary
$(TARGET): info $(OBJ_DIR) $(CPPOBJS) $(CUOBJS) $(BIN_DIR)
	@echo "Linking with flags: $(LDFLAGS)"
	@$(CU) -o $(BIN_DIR)/$(TARGET) $(CPPOBJS) $(CUOBJS) $(LDFLAGS)

# Create directory for dependency and object files
$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

# Create directory for target binary
$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

# Compile objects
$(CPPOBJS): $(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	@echo "$(SRC_DIR)/$*.cpp -> $@"
	@$(CPP) $(CPPFLAGS) -o $@ -c $(SRC_DIR)/$*.cpp

$(CUOBJS): $(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu
	@echo "$(SRC_DIR)/$*.cu -> $@"
	@$(CU) $(CUFLAGS) -o $(OBJ_DIR)/$*.o -c $(SRC_DIR)/$*.cu

-include $(DEPS)