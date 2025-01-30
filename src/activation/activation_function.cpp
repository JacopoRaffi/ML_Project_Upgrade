#include "activation_function.hpp"

Eigen::VectorXd ReLU::activate(const Eigen::VectorXd& x){
    return x.array().max(0);
};

Eigen::VectorXd ReLU::derivative(const Eigen::VectorXd& x){
    return (x.array() > 0).cast<double>();
};
