#include "../includes/activation_function.hpp"
#include <iostream>

// ---------------------------------------- Linear ----------------------------------------
Eigen::MatrixXd Linear::activate(const Eigen::MatrixXd& x){
    return x;
};

Eigen::MatrixXd Linear::derivative(const Eigen::MatrixXd& x){
    return Eigen::MatrixXd::Ones(x.rows(), x.cols());
};


// ---------------------------------------- RELU ----------------------------------------
Eigen::MatrixXd ReLU::activate(const Eigen::MatrixXd& x){
    return x.cwiseMax(0);
};

Eigen::MatrixXd ReLU::derivative(const Eigen::MatrixXd& x){
    return (x.array() > 0).cast<double>();
};

// ---------------------------------------- Sigmoid ----------------------------------------
Eigen::MatrixXd Sigmoid::activate(const Eigen::MatrixXd& x){
    return 1 / (1 + (-x.array()).exp());
};

Eigen::MatrixXd Sigmoid::derivative(const Eigen::MatrixXd& x){
    return activate(x).array() * (1 - activate(x).array());
};

// ---------------------------------------- Tanh ----------------------------------------
Eigen::MatrixXd Tanh::activate(const Eigen::MatrixXd& x){
    return x.array().tanh();
};

Eigen::MatrixXd Tanh::derivative(const Eigen::MatrixXd& x){
    return 1 - activate(x).array().square();
};