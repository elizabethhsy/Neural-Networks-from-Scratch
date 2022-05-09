#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <functional>
#include <iostream>
#include <math.h>
#include <numeric>
#include <tuple>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

struct Layer_Dense {
    MatrixXd weights;
    VectorXd biases;
    MatrixXd outputs;

    // constructor
    Layer_Dense(int n_inputs, int n_neurons) {
        weights = MatrixXd::Random(n_inputs, n_neurons);
        biases = VectorXd::Zero(n_neurons);
    }
    // forward pass
    void forward(MatrixXd inputs) {
        outputs = inputs*weights;
        outputs.rowwise() += biases.transpose();
    }
};

struct Activation_ReLU {
    MatrixXd outputs;
    void forward(MatrixXd inputs) {
        outputs = inputs.unaryExpr([](double i) {
            return std::max(0.0, i);
        });
    }
};

struct Activation_Softmax {
    MatrixXd outputs;
    void forward(MatrixXd inputs) {
        VectorXd maximum = inputs.rowwise().maxCoeff();
        // prevent overflow
        inputs.colwise() -= maximum;

        // exponentiate
        MatrixXd output_layer = inputs.unaryExpr([](double i) {
            return exp(i);
        });

        // normalize
        VectorXd output_sum = output_layer.rowwise().sum();
        MatrixXd normalized_output(output_layer.rows(), output_layer.cols()+1);
        normalized_output << output_layer, output_sum;

        outputs = normalized_output.rowwise().hnormalized();
    }
};

struct Loss {
    virtual VectorXd forward(MatrixXd outputs, MatrixXd y) = 0;

    float calculate(MatrixXd outputs, MatrixXd y) { // y is the intended target values
        VectorXd sample_losses(outputs.rows());
        sample_losses = forward(outputs, y);

        float data_loss = sample_losses.mean();
        return data_loss;
    }
};

// categorical cross entropy loss
struct Loss_CategoricalCrossentropy : Loss {
    VectorXd forward(MatrixXd y_pred, MatrixXd y_true) { // y_pred are the NN's predicted values, y_true are the target values
        int samples = y_pred.rows(); // get the number of samples in a batch
        MatrixXd y_pred_clipped = y_pred.unaryExpr([](double i) {
            return std::clamp(i, 1e-7, 1-1e-7); // prevent log(0)
        });

        // check if the y_true values are passed in as scalar values or one hot encoded values
        VectorXd correct_confidences(y_pred.rows());
        if (y_true.rows() == 1) {// scalar values
            for (int i=0; i<y_pred.rows(); i++) {
                correct_confidences[i] = y_pred_clipped.coeff(i, y_true.coeff(0, i)); // ith row, index specified by y_true
            }
        }
        else { // one hot encoded values
            correct_confidences = (y_pred_clipped*y_true).rowwise().sum();
        }

        VectorXd negative_log_likelihoods = correct_confidences.unaryExpr([](double i) {
            return -1*log(i);
        });

        return negative_log_likelihoods;
    }
};

int main(int, char**) {
    // srand((unsigned int) time(0));
    Eigen::IOFormat CleanFmt(6, 0, ", ", "\n", "[", "]");

    // input data
    MatrixXd X(3, 4);
    X << 1, 2, 3, 2.5, 2, 5, -1, 2, -1.5, 2.7, 3.3, -0.8;

    Layer_Dense layer1(4, 5);
    Activation_ReLU activation1;
    Activation_Softmax softmax1;
    Layer_Dense layer2(5, 3);
    Loss_CategoricalCrossentropy loss_function;

    layer1.forward(X);
    activation1.forward(layer1.outputs);
    layer2.forward(activation1.outputs);
    softmax1.forward(layer2.outputs);

    VectorXd target_output(3);
    target_output << 1, 0, 2;

    // VectorXd loss(3);
    // for (int i=0; i<target_output.size(); i++) {
    //     // categorical cross entropy loss
    //     loss[i] = -1*log(softmax1.outputs.coeff(target_output[i], i));
    // }
    // float loss_value = loss.mean();

    float loss = loss_function.calculate(softmax1.outputs, target_output.matrix());

    // std::cout << layer1.outputs.format(CleanFmt) << "\n";
    // std::cout << activation1.outputs.format(CleanFmt) << "\n";
    // std::cout << layer2.outputs.format(CleanFmt) << "\n";
    std::cout << softmax1.outputs.format(CleanFmt) << "\n";
    // std::cout << loss.format(CleanFmt) << "\n";
    // std::cout << loss_value << "\n";
    std::cout << loss << "\n";
    return 0;
}