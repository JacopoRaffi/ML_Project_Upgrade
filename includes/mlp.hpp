#ifndef MLP_HPP
#define MLP_HPP

#include <eigen3/Eigen/Dense>
#include <vector>
#include <string>
#include <utility>
#include "layer.hpp"
#include "../includes/loss_function.hpp"

/**
 * @brief Multi-layer perceptron class
 * 
 * This class implements a multi-layer perceptron (MLP) model, trained with standard SGD.
 */
class MLP{
private:
    std::vector<std::unique_ptr<FCLayer>> layers;

    /**
     * @brief Backward pass
     * 
     * @param loss_grad Gradient of the loss function
     */
    void backward(Eigen::MatrixXd loss_grad);

    /**
     * @brief Update weights
     */
    void update(double lr, double weight_decay, double momentum);

public:
    /**
     * @brief Construct a new MLP object
     * 
     * @param input_size Input size
     * @param layers List of pairs of (number of neurons, activation function) for each layer. Last layer will be the output layer
     */
    MLP(int input_size, std::vector<std::pair<int, ActivationFunction*>> layers);

    /**
     * @brief Re-Initialize weights and biases given the ranges. the i-th ranges is used for layer i
     * 
     * @param weight_ranges Weight ranges
     * @param bias_ranges Bias ranges
     */
    void init_weights(std::vector<std::pair<int, int>> weight_ranges, std::vector<std::pair<int, int>> bias_ranges);

    /**
     * @brief Forward pass
     * 
     * @param x Input data
     * @return Eigen::MatrixXd Output data
     */
    Eigen::MatrixXd predict(Eigen::MatrixXd x);

    /**
     * @brief Fit the model
     * 
     * @param x Input data
     * @param y Target data
     * @param epochs Number of epochs
     * @param num_minibatches Number of minibatches
     * @param learning_rate Learning rate
     * @param weight_decay Weight decay
     * @param momentum Momentum
     */
    std::vector<std::pair<double, double>> fit(Eigen::MatrixXd x_train, Eigen::MatrixXd y_train, Eigen::MatrixXd x_test, Eigen::MatrixXd y_test, 
            int epochs, int num_minibatches, double learning_rate, double weight_decay, double momentum, LossFunction* loss_function);


    /**
     * @brief Evaluate the model
     * 
     * @param x Input data
     * @param y Target data
     * @param loss_function Loss function
     * @return double Loss value
     */
    double evaluate(Eigen::MatrixXd x, Eigen::MatrixXd y, LossFunction* loss_function);

    ~MLP() = default;
};

#endif // MLP_HPP