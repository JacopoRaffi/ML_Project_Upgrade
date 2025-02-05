#include "mlp.hpp"
#include "layer.hpp"
#include "../activation/activation_function.hpp"


MLP::MLP(int input_size, std::vector<std::pair<int, ActivationFunction*>> layers){
    for(int i = 0; i < layers.size(); i++){   
        this->layers.push_back(FCLayer(input_size, layers[i].first, std::unique_ptr<ActivationFunction>(layers[i].second))); //create layers
        input_size = layers[i].first;
    }
}

void MLP::init_weights(std::vector<std::pair<int, int>> weight_ranges, std::vector<std::pair<int, int>> bias_ranges){
    for(int i = 0; i < layers.size(); i++){
        layers[i].init_weights(weight_ranges[i].first, weight_ranges[i].second, bias_ranges[i].first, bias_ranges[i].second); //re-init weights
    }
}

Eigen::MatrixXd MLP::predict(Eigen::MatrixXd x){
    for(int i = 0; i < layers.size(); i++){
        x = layers[i].forward(x);
    }

    return x;
}