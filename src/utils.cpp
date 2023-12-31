#include "NumCpp.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include "../include/utils.hpp"
#include <vector>
#include <unordered_map>

nc::NdArray<double> read_csv(const std::string& filepath, bool skip_header) {
    /*
    Helper function to read a .csv file and insert its contents into an NumCpp Array.
    Note that this function assumes the .csv file contains headers, meaning the first 
    line of the .csv file is skipped.

    Parameters
    ----------
    filepath: path to the .csv file.

    Returns 
    ---------
    matrix: the contents of the .csv file in a NumCpp array.

    */

    std::vector<std::vector<double>> data;  // vec to store the .csv data
    nc::NdArray<double> matrix; // matrix to put data into and return

    
    std::ifstream file(filepath); // open the .csv file

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open .csv file." << std::endl;
        return matrix; // empty matrix
    }

    std::string line;

    while (std::getline(file, line)) {
        if (skip_header) {
            // skip the first line containing header
            skip_header = false;

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
                    std::cerr << "Error: Invalid data format in .csv file." << std::endl;
                    return matrix;
                }
            }

            data.push_back(row);
        }
    }

    file.close();

    const int n_samples = data.size();
    const int n_columns = data[0].size();

    matrix = nc::NdArray<double>(n_samples, n_columns);

    for(nc::uint32 row = 0; row < n_samples; ++row) {
        for (nc::uint32 col = 0; col < n_columns; ++col) {
            matrix(row, col) = data[row][col];
        }
    }

    return matrix;
}



template<typename T>
T get_most_frequent_element(std::vector<T>& vec) {
    // returns the most frequently occuring element 
    // in input vector vec. Vec may hold either integers or doubles.

    std::unordered_map<T, int> freq_map;

    for (const T& element : vec) {
        freq_map[element]++;
    }


    T most_freq_element;
    int max_freq = 0;

    for (const auto& pair : freq_map) {
        if (pair.second > max_freq) {
            most_freq_element = pair.first;
            max_freq = pair.second;
        }
    }

    return most_freq_element;
}

template int get_most_frequent_element(std::vector<int>& vec);
template double get_most_frequent_element(std::vector<double>& vec);