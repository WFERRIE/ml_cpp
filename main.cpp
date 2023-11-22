#include "NumCpp.hpp"
#include <iostream>
#include "utils/csvreader.h"
#include "linear/logistic_regression.h"

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


int main() {

    std::vector<std::vector<double>> data = read_csv("data/iris_binary.csv");

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


    nc::NdArray<double> a1 = { { 0.25, 0.45}, { 0.55, 0.75 }, { 0.01, 0.99 } };

    logistic_regression logit_reg(10000, 0.01);
    std::cout << logit_reg.get_bias() << std::endl;
    logit_reg.fit(X, y, false);
    std::cout << logit_reg.get_bias() << std::endl;
    std::cout << logit_reg.predict(X) << std::endl;


    // return 0;

}
