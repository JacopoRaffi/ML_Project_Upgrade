#include "activation_function.hpp"

// ---------------------------------------- Linear ----------------------------------------
Eigen::VectorXd Linear::activate(const Eigen::VectorXd& x){
    return x;
};

Eigen::VectorXd Linear::derivative(const Eigen::VectorXd& x){
    return Eigen::VectorXd::Ones(x.size());
};


// ---------------------------------------- RELU ----------------------------------------
Eigen::VectorXd ReLU::activate(const Eigen::VectorXd& x){
    return x.array().max(0);
};

Eigen::VectorXd ReLU::derivative(const Eigen::VectorXd& x){
    return (x.array() > 0).cast<double>();
};

// ---------------------------------------- Sigmoid ----------------------------------------
Eigen::VectorXd Sigmoid::activate(const Eigen::VectorXd& x){
    return 1 / (1 + (-x.array()).exp());
};

Eigen::VectorXd Sigmoid::derivative(const Eigen::VectorXd& x){
    return activate(x).array() * (1 - activate(x).array());
};

// ---------------------------------------- Tanh ----------------------------------------
Eigen::VectorXd Tanh::activate(const Eigen::VectorXd& x){
    return x.array().tanh();
};

Eigen::VectorXd Tanh::derivative(const Eigen::VectorXd& x){
    return 1 - activate(x).array().square();
};