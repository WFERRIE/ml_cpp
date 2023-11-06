#include "NumCpp.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>




int main() {
    std::vector<std::vector<double>> data;  // Vector to store the CSV data

    // Open the CSV file
    std::ifstream file("iris.csv");

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open the CSV file." << std::endl;
        return 1;
    }

    std::string line;

    bool first_line = true;
    std::vector<std::string> column_names;

    while (std::getline(file, line)) {
        if (first_line) {
            // skip the first line containing headers
            first_line = false;

        } else {
            std::vector<double> row;  // Vector to store each row

            // Use a stringstream to split the line into values
            std::stringstream lineStream(line);
            std::string cell;

            while (std::getline(lineStream, cell, ',')) {
                try {
                    double value = std::stod(cell);
                    row.push_back(value);
                } catch (const std::invalid_argument&) {
                    std::cerr << "Error: Invalid data format in the CSV file." << std::endl;
                    return 1;
                }
            }

            data.push_back(row);
        }
    }

    // Close the file
    file.close();

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
