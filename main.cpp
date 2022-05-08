#include <Eigen/Dense>
#include <iostream>
#include <numeric>
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

int main(int, char**) {
    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

    // input data
    MatrixXd X(3, 4);
    X << 1, 2, 3, 2.5, 2, 5, -1, 2, -1.5, 2.7, 3.3, -0.8;

    Layer_Dense layer1(4, 5);
    Layer_Dense layer2(5, 2);

    layer1.forward(X);
    layer2.forward(layer1.outputs);

    std::cout << layer1.outputs.format(CleanFmt) << "\n";
    std::cout << layer2.outputs.format(CleanFmt) << "\n";
    return 0;
}