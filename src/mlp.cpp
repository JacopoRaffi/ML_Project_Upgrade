#include "../includes/mlp.hpp"
#include "../includes/layer.hpp"
#include "../includes/activation_function.hpp"


MLP::MLP(int input_size, std::vector<std::pair<int, ActivationFunction*>> layers) {
    for (int i = 0; i < layers.size(); i++) {  
        std::unique_ptr<ActivationFunction> activation_function = std::unique_ptr<ActivationFunction>(layers[i].second);
        
        this->layers.push_back(std::make_unique<FCLayer>(input_size, layers[i].first, std::move(activation_function)));
        input_size = layers[i].first;
    }
}

void MLP::init_weights(std::vector<std::pair<int, int>> weight_ranges, std::vector<std::pair<int, int>> bias_ranges){
    for(int i = 0; i < layers.size(); i++){
        layers[i]->init_weights(weight_ranges[i].first, weight_ranges[i].second, bias_ranges[i].first, bias_ranges[i].second); //re-init weights
    }
}

Eigen::MatrixXd MLP::predict(Eigen::MatrixXd x){
    for(int i = 0; i < layers.size(); i++){
        x = layers[i]->forward(x);
    }

    return x;
}

void MLP::backward(Eigen::MatrixXd grad){
    for(int i = layers.size() - 1; i >= 0; i--){
        grad = layers[i]->backward(grad);
    }
}

void MLP::update(double lr, double weight_decay, double momentum){
    for(int i = 0; i < layers.size(); i++){
        layers[i]->update(lr, weight_decay, momentum);
    }
}

double MLP::evaluate(Eigen::MatrixXd x, Eigen::MatrixXd y, LossFunction* loss_function){
    Eigen::MatrixXd y_pred = predict(x);
    return loss_function->loss(y, y_pred);
}

std::vector<std::pair<double, double>> MLP::fit(Eigen::MatrixXd x, Eigen::MatrixXd y, Eigen::MatrixXd x_test, Eigen::MatrixXd y_test, int epochs, int num_minibatches, double learning_rate, double weight_decay, double momentum, LossFunction* loss_function){
    std::vector<std::pair<double, double>> loss_history; //store the loss history both for training and testing
    
    for(int i = 0; i < epochs; i++){
        double tmp_train_loss = 0;
        for(int j = 0; j < num_minibatches; j++){
            int batch_size = x.rows() / num_minibatches;
            Eigen::MatrixXd x_train_batch = x.block(j * batch_size, 0, batch_size, x.cols()); //take the input minibatch 
            Eigen::MatrixXd y_true_batch = y.block(j * batch_size, 0, batch_size, y.cols()); //take the targets minibatch

            Eigen::MatrixXd y_pred = predict(x_train_batch); //forward pass
            tmp_train_loss += loss_function->loss(y_true_batch, y_pred); //compute loss of the minibatch
            Eigen::MatrixXd loss_grad = loss_function->backward(y_true_batch, y_pred); //compute loss gradient

            backward(loss_grad); //backward pass
            update(learning_rate, weight_decay, momentum); //update weights
        }

        loss_history.insert(loss_history.begin(), std::make_pair(tmp_train_loss / num_minibatches, evaluate(x_test, y_test, loss_function)));
    }
}