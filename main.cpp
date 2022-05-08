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

int main(int, char**) {
    srand((unsigned int) time(0));
    Eigen::IOFormat CleanFmt(6, 0, ", ", "\n", "[", "]");

    // input data
    MatrixXd X(3, 4);
    X << 1, 2, 3, 2.5, 2, 5, -1, 2, -1.5, 2.7, 3.3, -0.8;

    Layer_Dense layer1(4, 5);
    Activation_ReLU activation1;
    Activation_Softmax softmax1;
    Layer_Dense layer2(5, 2);

    layer1.forward(X);
    activation1.forward(layer1.outputs);
    layer2.forward(activation1.outputs);
    softmax1.forward(layer2.outputs);


    std::cout << layer1.outputs.format(CleanFmt) << "\n";
    std::cout << activation1.outputs.format(CleanFmt) << "\n";
    std::cout << layer2.outputs.format(CleanFmt) << "\n";
    std::cout << softmax1.outputs.format(CleanFmt) << "\n";
    return 0;
}