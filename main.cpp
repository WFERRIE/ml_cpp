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

int main() {

    std::vector<std::vector<double>> data = readCSV("iris.csv");

    const nc::uint32 n_samples = 150;
    const nc::uint32 n_features = 5;

    auto matrix = nc::NdArray<double>(n_samples, n_features);

    for(nc::uint32 row = 0; row < n_samples; ++row) {
        for (nc::uint32 col = 0; col < n_features; ++col) {
            matrix(row, col) = data[row][col];
        }
    }

    // std::cout << matrix({0, n_samples}, {0, n_features}) << std::endl;

    matrix = fisher_yates_shuffle(matrix, n_samples);

    std::cout << matrix << std::endl;

    return 0;

}
