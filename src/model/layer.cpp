#include "layer.hpp"
#include <memory>
#include "../activation/activation_function.hpp"
#include "../loss/loss_function.hpp"
#include <iostream>

FCLayer::FCLayer(int input_size, int output_size, std::unique_ptr<ActivationFunction> func, 
            float min_val, float max_val, float bias_max_val, float bias_min_val){
    
    activation = std::move(func);
    weights = min_val + (Eigen::MatrixXd::Random(output_size, input_size).array() + 1.0) * (max_val - min_val) / 2.0;
    bias =  bias_min_val + (Eigen::MatrixXd::Random(output_size, 1).array() + 1.0) * (bias_max_val - bias_min_val) / 2.0;
};

Eigen::MatrixXd FCLayer::forward(const Eigen::MatrixXd& x){
    input = x;
    output = x * weights.transpose() + bias.transpose().replicate(x.rows(), 1);
    return activation->activate(output);
};

Eigen::MatrixXd FCLayer::backward(const Eigen::MatrixXd& grad){
    Eigen::MatrixXd delta = grad.cwiseProduct(activation->derivative(output)); 
    grad_weights = delta.transpose() * input;
    grad_bias = delta.colwise().sum();

    return delta * weights;
};

void FCLayer::update(double learning_rate, double weight_decay, double momentum){
    Eigen::MatrixXd weights_update = learning_rate * grad_weights + momentum * prev_weights_update; // lr*grad + momentum*prev_update
    Eigen::MatrixXd bias_update = learning_rate * grad_bias + momentum * prev_bias_update;

    weights -= weights_update - weight_decay * weights; // w = w - lr*grad - wd*w
    bias -= bias_update; // do not apply regularization for the bias

    prev_weights_update = weights_update;
    prev_bias_update = bias_update;
};

int main(){
    MSE mse;
    std::unique_ptr<ActivationFunction> activation(new Sigmoid());
    FCLayer layer(2, 2, std::move(activation));
    Eigen::MatrixXd x(2, 2);
    x << 1, 2, 3, 4;
    Eigen::MatrixXd y = layer.forward(x);

    Eigen::MatrixXd y_true = Eigen::MatrixXd::Ones(2, 2);

    Eigen::MatrixXd grad = mse.backward(y_true, y);
    Eigen::MatrixXd delta = layer.backward(grad);
    return 0;
}