#include <Eigen/Dense>
#include <iostream>
#include <numeric>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main(int, char**) {
    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

    MatrixXd inputs(3, 4);
    inputs << 1, 2, 3, 2.5, 2, 5, -1, 2, -1.5, 2.7, 3.3, -0.8;

    MatrixXd weights(3, 4);
    weights << 0.2, 0.8, -0.5, 1, 0.5, -0.91, 0.26, -0.5, -0.26, -0.27, 0.17, 0.87;

    VectorXd biases(3);
    biases << 2, 3, 0.5;

    MatrixXd weights2(3, 3);
    weights2 << 0.1, -0.14, 0.5, -0.5, 0.12, -0.33, -0.44, 0.73, -0.13;

    VectorXd biases2(3);
    biases2 << -1, 2, -0.5;

    MatrixXd layer1_outputs(3, 3);
    layer1_outputs << inputs*weights.transpose();
    layer1_outputs.rowwise() += biases.transpose();

    MatrixXd layer2_outputs(3, 3);
    layer2_outputs << layer1_outputs*weights2.transpose();
    layer2_outputs.rowwise() += biases2.transpose();

    std::cout << layer1_outputs.format(CleanFmt) << "\n";
    std::cout << layer2_outputs.format(CleanFmt) << "\n";

    return 0;
}