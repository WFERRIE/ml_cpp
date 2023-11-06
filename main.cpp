#include "NumCpp.hpp"
#include "csvreader.h"
#include <iostream>

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

    std::cout << matrix << std::endl;
    return 0;
}
