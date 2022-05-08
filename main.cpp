#include <Eigen/Dense>
#include <iostream>
#include <numeric>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main(int, char**) {
    VectorXd inputs(4);
    inputs << 1, 2, 3, 2.5;

    MatrixXd weights(3, 4);
    weights << 0.2, 0.8, -0.5, 1, 0.5, -0.91, 0.26, -0.5, -0.26, -0.27, 0.17, 0.87;

    VectorXd biases(3);
    biases << 2, 3, 0.5;

    VectorXd outputs(3);
    outputs << weights*inputs.matrix() + biases;

    std::cout << outputs.transpose() << "\n";

    // std::vector <float> inputs = {1, 2, 3, 2.5};

    // std::vector <std::vector<float> > weights = {
    //     {0.2, 0.8, -0.5, 1},
    //     {0.5, -0.91, 0.26, -0.5},
    //     {-0.26, -0.27, 0.17, 0.87}
    // };

    // std::cout << outputs.transpose() << '\n';

    // std::vector <float> biases = {2, 3, 0.5};

    // // std::vector <float> output = {
    // //     std::inner_product(std::begin(inputs), std::end(inputs), std::begin(weights[0]), biases[0]),
    // //     std::inner_product(std::begin(inputs), std::end(inputs), std::begin(weights[1]), biases[1]),
    // //     std::inner_product(std::begin(inputs), std::end(inputs), std::begin(weights[2]), biases[2])
    // // };

    // std::vector <float> output = {
    //     std::inner_product(std::begin(weights), std::end(weights), std::begin(inputs), biases)
    // };

    // std::vector <float> layer_outputs;
    // float neuron_output;

    // for (int i=0;i<weights.size(); i++) {
    //     auto neuron_weights = weights[i];
    //     auto neuron_bias = biases[i];
    //     neuron_output = 0;

    //     for (int j=0; j<neuron_weights.size();j++) {
    //         auto neuron_weight = neuron_weights[j];
    //         auto neuron_input = inputs[j];
    //         neuron_output += neuron_weight*neuron_input;
    //     }
    //     neuron_output += neuron_bias;
    //     layer_outputs.push_back(neuron_output);
    // }

    // printf("Output: ");
    // for (float i:layer_outputs) {
    //     printf("%.6g ", i);
    // }
    // printf("\n");

    return 0;
}