#include "csvreader.h"
#include <iostream>
#include <fstream>
#include <sstream>

std::vector<std::vector<double>> read_csv(const std::string& filename) {
    std::vector<std::vector<double>> data;  // Vector to store the CSV data

    // Open the CSV file
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open the CSV file." << std::endl;
        return data; // Return an empty vector in case of an error
    }

    std::string line;
    bool first_line = true;

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
                    return data;
                }
            }

            data.push_back(row);
        }
    }

    // Close the file
    file.close();

    return data;
}
