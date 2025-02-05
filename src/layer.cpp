#include "../includes/layer.hpp"
#include <memory>
#include "../includes/activation_function.hpp"
#include "../includes/loss_function.hpp"
#include <iostream>

FCLayer::FCLayer(int input_size, int output_size, std::unique_ptr<ActivationFunction> func, 
            float min_val, float max_val, float bias_max_val, float bias_min_val){
    
    activation = std::move(func);
    this->input_size = input_size;
    this->output_size = output_size;

    // Uniform random initialization in the range [min_val, max_val] both for weights and bias
    weights = min_val + (Eigen::MatrixXd::Random(output_size, input_size).array() + 1.0) * (max_val - min_val) / 2.0;
    bias =  bias_min_val + (Eigen::MatrixXd::Random(output_size, 1).array() + 1.0) * (bias_max_val - bias_min_val) / 2.0; 
};

void FCLayer::init_weights(float min_val, float max_val, float bias_max_val, float bias_min_val){
    weights = min_val + (Eigen::MatrixXd::Random(output_size, input_size).array() + 1.0) * (max_val - min_val) / 2.0;
    bias =  bias_min_val + (Eigen::MatrixXd::Random(output_size, 1).array() + 1.0) * (bias_max_val - bias_min_val) / 2.0; 
};

Eigen::MatrixXd FCLayer::forward(const Eigen::MatrixXd& x){
    input = x;
    output = x * weights.transpose() + bias.transpose().replicate(x.rows(), 1); // X*W^T + b^T
    return activation->activate(output); // activation(X*W^T + b^T)
};

Eigen::MatrixXd FCLayer::backward(const Eigen::MatrixXd& grad){
    Eigen::MatrixXd delta = grad.cwiseProduct(activation->derivative(output)); // grad * activation'(output)
    grad_weights = delta.transpose() * input; // delta^T * X
    grad_bias = delta.colwise().sum();

    return delta * weights; // error to backpropagate
};

void FCLayer::update(double learning_rate, double weight_decay, double momentum){
    Eigen::MatrixXd weights_update = learning_rate * grad_weights + momentum * prev_weights_update; // lr*grad + momentum*prev_update
    Eigen::MatrixXd bias_update = learning_rate * grad_bias + momentum * prev_bias_update;

    weights -= weights_update - weight_decay * weights; // w = w - lr*grad - wd*w
    bias -= bias_update; // do not apply regularization for the bias

    prev_weights_update = weights_update;
    prev_bias_update = bias_update;
};