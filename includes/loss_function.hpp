#ifndef LOSS_FUNCTION_HPP
#define LOSS_FUNCTION_HPP

#include <eigen3/Eigen/Dense>

/**
 * @brief Abstract base class for loss functions.
 * 
 * This class defines the interface for all loss functions that can be used to compute the loss
 * between the true and predicted values of a neural network.
 */
class LossFunction {
public:

    /**
     * @brief Virtual function to compute the loss between true and predicted values.
     * 
     * This method must be implemented by any derived class to compute the loss between the true
     * and predicted values of a neural network.
     * 
     * @param y_true True values of the target variable.
     * @param y_pred Predicted values of the target variable.
     * @return double The loss between the true and predicted values.
     */
    virtual double loss(const Eigen::MatrixXd& y_true, const Eigen::MatrixXd& y_pred) = 0;

    /**
     * @brief Virtual function to compute the derivative of the loss function.
     * 
     * This method must be implemented by any derived class to compute the derivative
     * of the specific loss function.
     * 
     * @param y_true True values of the target variable.
     * @param y_pred Predicted values of the target variable.
     * @return Eigen::MatrixXd The derivative of the loss function.
     */
    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& y_true, const Eigen::MatrixXd& y_pred) = 0;

    virtual ~LossFunction() = default;
};

/**
 * @brief Mean Squared Error (MSE) loss function.
 * 
 * The Mean Squared Error (MSE) loss function is defined as L(y_true, y_pred) = (1 / n) * sum((y_true - y_pred)^2),
 * where n is the number of samples in the batch.
 */
class MSE : public LossFunction {
public:

    /**
     * @brief Compute the Mean Squared Error (MSE) loss between true and predicted values.
     * 
     * This method computes the Mean Squared Error (MSE) loss between the true and predicted values of a neural network.
     * 
     * @param y_true True values of the target variable.
     * @param y_pred Predicted values of the target variable.
     * @return double The Mean Squared Error (MSE) loss between the true and predicted values.
     */
    double loss(const Eigen::MatrixXd& y_true, const Eigen::MatrixXd& y_pred) override;

    Eigen::MatrixXd backward(const Eigen::MatrixXd& y_true, const Eigen::MatrixXd& y_pred) override;
};

#endif // LOSS_FUNCTION_HPP