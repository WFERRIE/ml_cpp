#include "NumCpp.hpp"
#include "csvreader.h"
#include <iostream>


nc::NdArray<double> fisher_yates_shuffle(nc::NdArray<double> input, const int& n_samples) {

    // shuffle an array, row-wise

    for (int i = n_samples - 1; i > 0; i--) {
        int j = std::rand() % (i + 1);

        nc::NdArray<double> temp = input(j, input.cSlice());
        input.put(j, input.cSlice(), input(i, input.cSlice())); 
        input.put(i, input.cSlice(), temp); 

    }

    return input;
}

nc::NdArray<double> sigmoid(nc::NdArray<double> z) {
    return 1.0 / (1.0 + nc::exp(-z));
}

double compute_BCE_cost(nc::NdArray<double> predictions, nc::NdArray<double> y, const nc::uint32 n_samples) {

    auto cost = -(1.0 / n_samples) * nc::sum<double>(y * nc::log(predictions) + (1.0 - y) * nc::log(1.0 - predictions)); // cross entropy loss

    return cost(0, 0); // return it as a double instead of a 1 element nc::NdArray

}

int main() {

    std::vector<std::vector<double>> data = readCSV("iris_binary.csv");

    const nc::uint32 n_samples = 150;
    const nc::uint32 n_features = 5; // this counts the labels as a feature, so n_features-1 input features, 1 output feature

    auto matrix = nc::NdArray<double>(n_samples, n_features);

    for(nc::uint32 row = 0; row < n_samples; ++row) {
        for (nc::uint32 col = 0; col < n_features; ++col) {
            matrix(row, col) = data[row][col];
        }
    }

    matrix = fisher_yates_shuffle(matrix, n_samples);


    auto y = matrix(matrix.rSlice(), n_features - 1);
    auto X = matrix(matrix.rSlice(), {0, n_features - 1});


    int n_iters = 100;

    nc::NdArray<double> weights = nc::zeros<double>(n_features - 1, 1);
    double bias = 0.0;

    double lr = 0.01;

    nc::NdArray<double> predictions;


    for (int i = 0; i < n_iters; i++) {
        auto z = nc::dot<double>(X, weights) + bias;
        predictions = sigmoid(z);

        auto dw = 1.0 / n_samples * nc::dot(X.transpose(), (predictions - y));
        auto db = 1.0 / n_samples * nc::sum<double>(predictions - y);

        // std::cout << weights << std::endl;
        // std::cout << bias << std::endl;
        // std::cout << dw << std::endl;
        // std::cout << db << std::endl;
        weights = weights - lr * dw;
        bias = bias - (lr * db)(0, 0); // convert 1 element nc::NdArray to a double by accessing the (0, 0) index
        // std::cout << weights << std::endl;
        // std::cout << bias << std::endl;


        if (i % 10 == 0) {
            auto cost = compute_BCE_cost(predictions, y, n_samples);
            std::cout << "Cost at iteration " << i << ": " << cost << std::endl;

        }        

    }

    std::cout << "Final set of predictions: " << "\n";
    std::cout << predictions << std::endl;


    return 0;

}
