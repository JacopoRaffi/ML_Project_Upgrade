#include "activation_function.hpp"

class ReLU : public ActivationFunction {
public:
    Eigen::VectorXd activate(Eigen::VectorXd x) override {
        return x.array().max(0);
    }

    Eigen::VectorXd derivative(Eigen::VectorXd x) override {
        return (x.array() > 0).cast<double>();
    }
};