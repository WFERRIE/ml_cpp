#include "NumCpp.hpp"
#include <iostream>
#include "../include/validation.hpp"


void check_consistent_shapes(nc::NdArray<double>& a, nc::NdArray<double>& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Error: Dimensions of inputs are not consistent.");
    }
}