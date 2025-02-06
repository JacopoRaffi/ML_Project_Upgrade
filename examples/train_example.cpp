#include <fstream>
#include <iostream>
#include <eigen3/Eigen/Dense>

#include "../includes/mlp.hpp"

/**
 * @brief Write the loss history to a file
 * 
 * @param loss_history the loss history
 * @param epochs the number of epochs
 * @param filename the name of the file
 * 
 * @return void
 */
void write_to_file(std::vector<std::pair<double, double>> loss_history, int epochs, std::string filename){
    std::ofstream file;
    file.open(filename);
    
    for(int i = 0; i < epochs; i++){
        file << i << " " << loss_history[i].first << " " << loss_history[i].second << "\n";
    }

    file.close();
}

/**
 * @brief Generate synthetic target data
 * 
 * @param input the input data
 * 
 * @return Eigen::MatrixXd the target data
 */
Eigen::MatrixXd synthetic_target(Eigen::MatrixXd input){
    Eigen::MatrixXd target = Eigen::MatrixXd::Zero(input.rows(), 3);
    for(int i = 0; i < input.rows(); i++){
        target(i, 0) = 2*input(i, 0) + 3*input(i, 1) + 4*input(i, 2) + 5*input(i, 3) + 6*input(i, 4); // y = 2x1 + 3x2 + 4x3 + 5x4 + 6x5
        target(i, 1) = 3*input(i, 0) + 4*input(i, 1) + 5*input(i, 2) + 6*input(i, 3) + 7*input(i, 4); // y = 3x1 + 4x2 + 5x3 + 6x4 + 7x5
        target(i, 2) = 4*input(i, 0) + 5*input(i, 1) + 6*input(i, 2) + 7*input(i, 3) + 8*input(i, 4); // y = 4x1 + 5x2 + 6x3 + 7x4 + 8x5
    }
    return target;
}

int main(int argc, char* argv[]){
    int num_samples = 1000;
    int num_features = 5;

    Eigen::MatrixXd train_x = Eigen::MatrixXd::Random(num_samples, num_features);
    Eigen::MatrixXd train_y = synthetic_target(train_x);

    Eigen::MatrixXd test_x = Eigen::MatrixXd::Random(num_samples, num_features);
    Eigen::MatrixXd test_y = synthetic_target(test_x);

    ActivationFunction* activation = new Sigmoid();
    ActivationFunction* output_activation = new Linear(); 

    std::vector<std::pair<int, ActivationFunction*>> layers;

    layers.push_back(std::make_pair(50, activation));
    layers.push_back(std::make_pair(50, activation));
    layers.push_back(std::make_pair(3, output_activation));

    // simple MLP with 2 hidden layers (50 neuron each and Sigmoid activation) and a linear output layer (3 neurons)
    MLP mlp(num_features, layers); 

    MSE mse;
    int epochs = 10;
    double lr = 0.01;
    double weight_decay = 0.0000001;
    double momentum = 0.9;
    int num_minibatches = 10;

    std::vector<std::pair<double, double>> loss_history = mlp.fit(train_x, train_y, test_x, test_y, epochs, num_minibatches, lr, weight_decay, momentum, &mse);

    write_to_file(loss_history, epochs, "loss.txt");

    return 0;
}