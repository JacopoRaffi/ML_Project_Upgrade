CXX = g++
CXXFLAGS = -O3

SRC_DIR = src
INCLUDES_DIR = includes
OBJ_DIR = build

.PHONY: all clean

all : $(OBJ_DIR)/mlp.o $(OBJ_DIR)/layer.o $(OBJ_DIR)/activation_function.o $(OBJ_DIR)/loss_function.o

clean :
	rm -f $(OBJ_DIR)/*.o

$(OBJ_DIR)/mlp.o : $(SRC_DIR)/mlp.cpp
	$(CXX) $(CXXFLAGS) -c $(SRC_DIR)/mlp.cpp -o $(OBJ_DIR)/mlp.o

$(OBJ_DIR)/layer.o : $(SRC_DIR)/layer.cpp
	$(CXX) $(CXXFLAGS) -c $(SRC_DIR)/layer.cpp -o $(OBJ_DIR)/layer.o

$(OBJ_DIR)/activation_function.o : $(SRC_DIR)/activation_function.cpp
	$(CXX) $(CXXFLAGS) -c $(SRC_DIR)/activation_function.cpp -o $(OBJ_DIR)/activation_function.o

$(OBJ_DIR)/loss_function.o : $(SRC_DIR)/loss_function.cpp
	$(CXX) $(CXXFLAGS) -c $(SRC_DIR)/loss_function.cpp -o $(OBJ_DIR)/loss_function.o
