#include "../includes/loss_function.hpp"
#include <iostream>


double MSE::loss(const Eigen::MatrixXd& y_true, const Eigen::MatrixXd& y_pred) {
    // Compute the Mean Squared Error (MSE) loss
    return (y_true - y_pred).array().square().mean();
}

Eigen::MatrixXd MSE::backward(const Eigen::MatrixXd& y_true, const Eigen::MatrixXd& y_pred) {
    // Compute the derivative of the Mean Squared Error (MSE) loss
    return 2 * (y_pred - y_true) / y_true.rows();
}