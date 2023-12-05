#include "NumCpp.hpp"
#include <iostream>
#include "../include/validation.hpp"


void check_consistent_shapes(nc::NdArray<double>& a, nc::NdArray<double>& b) {
    /*
    Checks two arrays to ensure they are the same shape. If they are not, throws an error.

    Parameters
    ----------
    a: one of the two arrays to check the shape of. 
    b: the other array to check the shape of.

    Returns 
    ---------
    N/A

    */

    if (a.shape() != b.shape()) {
        throw std::runtime_error("Error: Dimensions of inputs are not consistent.");
    }
}


nc::NdArray<double> replace_nan(nc::NdArray<double>& X, double val) {
    /*
    Replaces all nan values within an nc::NdArray with a desired value.

    Parameters
    ----------
    X: Array in which to replace all nan values.
    val: value with which to replace the nan values.


    Returns 
    ----------
    X: Modified array in which all nan values have been replaced with val.

    */

    for (auto& element : X) {
        if (nc::isnan(element)) {
            element = val;
        }
    }

    return X;
}