#include "layer.hpp"
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