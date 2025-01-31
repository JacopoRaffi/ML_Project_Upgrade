#ifndef ACTIVATION_FUNCTION_HPP
#define ACTIVATION_FUNCTION_HPP

#include <eigen3/Eigen/Dense>

/**
 * @brief Abstract base class for activation functions.
 * 
 * This class defines the interface for all activation functions that can be applied element-wise
 * to vectors, such as ReLU, Sigmoid, etc.
 */
class ActivationFunction {
public:

    /**
     * @brief Virtual function to apply an activation function to a vector.
     * 
     * This method must be implemented by any derived class to apply a specific activation function.
     * 
     * @param x Input matrix (batch) on which the activation function will be applied.
     * @return Eigen::MatrixXd The vector after applying the activation function element-wise.
     */
    virtual Eigen::MatrixXd activate(const Eigen::MatrixXd& x) = 0;

    /**
     * @brief Virtual function to compute the derivative of the activation function.
     * 
     * This method must be implemented by any derived class to compute the derivative
     * of the specific activation function.
     * 
     * @param x Input matrix (batch) for which the derivative is computed.
     * @return Eigen::MatrixXd The derivative of the activation function applied element-wise.
     */
    virtual Eigen::MatrixXd derivative(const Eigen::MatrixXd& x) = 0;

    virtual ~ActivationFunction() = default;
};


/**
 * @brief Linear activation function.
 * 
 * The Linear activation function is defined as f(x) = x.
 */
class Linear : public ActivationFunction {
public:

    /**
     * @brief Apply the Linear activation function to a vector.
     * 
     * This method applies the Linear activation function element-wise to a vector.
     * 
     * @param x Input matrix (batch) on which the Linear activation function will be applied.
     * @return Eigen::MatrixXd The vector after applying the Linear activation function element-wise.
     */
    Eigen::MatrixXd activate(const Eigen::MatrixXd& x) override;

    /**
     * @brief Compute the derivative of the Linear activation function.
     * 
     * This method computes the derivative of the Linear activation function element-wise.
     * 
     * @param x Input matrix (batch) for which the derivative is computed.
     * @return Eigen::MatrixXd The derivative of the Linear activation function applied element-wise.
     */
    Eigen::MatrixXd derivative(const Eigen::MatrixXd& x) override;
};


/**
 * @brief Rectified Linear Unit (ReLU) activation function.
 * 
 * The ReLU activation function is defined as f(x) = max(0, x).
 */
class ReLU : public ActivationFunction {
public:

    /**
     * @brief Apply the ReLU activation function to a vector.
     * 
     * This method applies the ReLU activation function element-wise to a vector.
     * 
     * @param x Input matrix (batch) on which the ReLU activation function will be applied.
     * @return Eigen::MatrixXd The vector after applying the ReLU activation function element-wise.
     */
    Eigen::MatrixXd activate(const Eigen::MatrixXd& x) override;

    /**
     * @brief Compute the derivative of the ReLU activation function.
     * 
     * This method computes the derivative of the ReLU activation function element-wise.
     * 
     * @param x Input matrix (batch) for which the derivative is computed.
     * @return Eigen::MatrixXd The derivative of the ReLU activation function applied element-wise.
     */
    Eigen::MatrixXd derivative(const Eigen::MatrixXd& x) override;
};


/**
 * @brief Sigmoid activation function.
 * 
 * The Sigmoid activation function is defined as f(x) = 1 / (1 + exp(-x)).
 */
class Sigmoid : public ActivationFunction {
public:

    /**
     * @brief Apply the Sigmoid activation function to a vector.
     * 
     * This method applies the Sigmoid activation function element-wise to a vector.
     * 
     * @param x Input matrix (batch) on which the Sigmoid activation function will be applied.
     * @return Eigen::MatrixXd The vector after applying the Sigmoid activation function element-wise.
     */
    Eigen::MatrixXd activate(const Eigen::MatrixXd& x) override;

    /**
     * @brief Compute the derivative of the Sigmoid activation function.
     * 
     * This method computes the derivative of the Sigmoid activation function element-wise.
     * 
     * @param x Input matrix (batch) for which the derivative is computed.
     * @return Eigen::MatrixXd The derivative of the Sigmoid activation function applied element-wise.
     */
    Eigen::MatrixXd derivative(const Eigen::MatrixXd& x) override;
};


/**
 * @brief Hyperbolic Tangent (Tanh) activation function.
 * 
 * The Tanh activation function is defined as f(x) = tanh(x).
 */
class Tanh : public ActivationFunction {
public:

    /**
     * @brief Apply the Tanh activation function to a vector.
     * 
     * This method applies the Tanh activation function element-wise to a vector.
     * 
     * @param x Input matrix (batch) on which the Tanh activation function will be applied.
     * @return Eigen::MatrixXd The vector after applying the Tanh activation function element-wise.
     */
    Eigen::MatrixXd activate(const Eigen::MatrixXd& x) override;

    /**
     * @brief Compute the derivative of the Tanh activation function.
     * 
     * This method computes the derivative of the Tanh activation function element-wise.
     * 
     * @param x Input matrix (batch) for which the derivative is computed.
     * @return Eigen::MatrixXd The derivative of the Tanh activation function applied element-wise.
     */
    Eigen::MatrixXd derivative(const Eigen::MatrixXd& x) override;
};

#endif // ACTIVATION_FUNCTION_HPP