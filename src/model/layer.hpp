#ifndef LAYER_HPP
#define LAYER_HPP

#include <eigen3/Eigen/Dense>
#include <memory>
#include "../activation/activation_function.hpp"

/**
 * @brief Abstract base class for neural network layers.
 * 
 * This class defines the interface for all neural network layers that can be used to build a neural network.
 */
class Layer{
public:
    /**
     * @brief Virtual function to perform the forward pass of the layer.
     * 
     * This method must be implemented by any derived class to perform the forward pass of the layer.
     * 
     * @param x Input matrix (batch) to the layer.
     * @return Eigen::MatrixXd Output matrix (batch) of the layer.
     */
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& x) = 0;

    /**
     * @brief Virtual function to perform the backward pass of the layer.
     * 
     * This method must be implemented by any derived class to perform the backward pass of the layer.
     * 
     * @param grad Gradient of the loss with respect to the output of the layer.
     * @return Eigen::MatrixXd Gradient of the loss with respect to the input of the layer.
     */
    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& grad) = 0;

    /**
     * @brief Virtual function to update the weights of the layer.
     * 
     * This method must be implemented by any derived class to update the weights of the layer.
     * 
     * @param learning_rate Learning rate used to update the weights.
     * @param weight_decay Weight decay used to update the weights.
     * @param momentum Momentum used to update the weights.
     */
    virtual void update(double learning_rate, double weight_decay, double momentum) = 0;

    virtual ~Layer() = default;
};

/**
 * @brief Fully Connected layer.
 * 
 * The FullyConnected class implements a fully connected layer in a neural network.
 */
class FCLayer : public Layer {
private:
    std::unique_ptr<ActivationFunction> activation;
    Eigen::MatrixXd weights;
    Eigen::MatrixXd bias;
    Eigen::MatrixXd input;
    Eigen::MatrixXd output;
    Eigen::MatrixXd grad_weights;
    Eigen::MatrixXd grad_bias;
    Eigen::MatrixXd grad_input;
    Eigen::MatrixXd grad_output;
    Eigen::MatrixXd prev_weights_update;
    Eigen::MatrixXd prev_bias_update;

public:
    /**
     * @brief Construct a new FCLayer object.
     * 
     * This constructor initializes the weights and bias of the fully connected layer using a random uniform distribution.
     * 
     * @param input_size Size of the input to the layer.
     * @param output_size Size of the output of the layer.
     * @param min_val Minimum value for the random initialization of the weights.
     * @param max_val Maximum value for the random initialization of the weights.
     * @param bias_max_val Maximum value for the random initialization of the bias.
     * @param bias_min_val Minimum value for the random initialization of the bias.
     */
    FCLayer(int input_size, int output_size, std::unique_ptr<ActivationFunction> func, 
            float min_val = -0.5, float max_val = 0.5, float bias_max_val = 0.1, float bias_min_val = -0.1);

    Eigen::MatrixXd forward(const Eigen::MatrixXd& x) override;

    Eigen::MatrixXd backward(const Eigen::MatrixXd& grad) override;

    void update(double learning_rate, double weight_decay, double momentum) override;
};

#endif // LAYER_HPP