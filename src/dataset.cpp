#include "../include/dataset.hpp"
#include "NumCpp.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

dataset::dataset(nc::NdArray<double> data) : data(data) {
    // Constructor
}
dataset::~dataset() {
    // Destructor
}


nc::NdArray<double> dataset::fisher_yates_shuffle(nc::NdArray<double>& input) {

    // shuffle an array, row-wise

    int n_samples = input.shape().rows;

    for (int i = n_samples - 1; i > 0; i--) {
        int j = std::rand() % (i + 1);

        nc::NdArray<double> temp = input(j, input.cSlice());
        input.put(j, input.cSlice(), input(i, input.cSlice())); 
        input.put(i, input.cSlice(), temp); 

    }

    return input;
}


void dataset::train_test_split(double train_size, bool shuffle) {
    if (train_size <= 0.0 || train_size >= 1.0) {
        std::runtime_error("Error: Please ensure train_size is between 0.0 and 1.0.");
    }

    if (shuffle) {
        data = fisher_yates_shuffle(data);
    }


    int n_columns = data.shape().cols;
    int n_features = n_columns - 1;
    int n_samples = data.shape().rows;

    y = data(data.rSlice(), n_columns - 1);
    X = data(data.rSlice(), {0, n_columns - 1});

    int train_idx_end = (int)(train_size * (double)n_samples);
    std::cout << train_idx_end << std::endl;

    y_train = y({0, train_idx_end} , y.cSlice());
    y_test = y({train_idx_end, n_samples}, y.cSlice());

    X_train = X({0, train_idx_end} , X.cSlice());
    X_test = X({train_idx_end, n_samples}, X.cSlice());

    train_test_set = true;
}

// need to run some checks to ensure if you try to get X or y, they have been properly set

void dataset::set_X(nc::NdArray<double>& new_X) {
    if (!train_test_set) {
        std::runtime_error("Error: please call .train_test_split() before attempting to use getters or setters.");
    }
    X = new_X;
}
void dataset::set_y(nc::NdArray<double>& new_y) {
    if (!train_test_set) {
        std::runtime_error("Error: please call .train_test_split() before attempting to use getters or setters.");
    }
    y = new_y;
}

nc::NdArray<double> dataset::get_X() {
    if (!train_test_set) {
        std::runtime_error("Error: please call .train_test_split() before attempting to use getters or setters.");
    }
    return X;
}
nc::NdArray<double> dataset::get_y() {
    if (!train_test_set) {
        std::runtime_error("Error: please call .train_test_split() before attempting to use getters or setters.");
    }
    return y;
}
