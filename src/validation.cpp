#include "NumCpp.hpp"
#include <iostream>
#include "../include/validation.hpp"


void check_consistent_shapes(nc::NdArray<double>& a, nc::NdArray<double>& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Error: Dimensions of inputs are not consistent.");
    }
}

int get_n_samples(nc::NdArray<double>& y_true) {
    int n_predictions;
    if (y_true.shape().rows > y_true.shape().cols) {
        // here we are checking if the labels have more rows or columns, and using the larger
        // as the total number of predictions. The idea is to make it robust to whether it 
        // is passed a 1xn or an nx1 set of labels. This should probably be checked in the validation.cpp
        // file somewhere, but for now Im just going to do it here

        n_predictions = y_true.shape().rows;
    }
    else {
        n_predictions = y_true.shape().cols;
    }

    return n_predictions;
}