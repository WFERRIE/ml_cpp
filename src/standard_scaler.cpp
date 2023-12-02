#include "NumCpp.hpp"
#include <iostream>
#include "../include/standard_scaler.hpp"


standard_scaler::standard_scaler(bool with_mean, bool with_std) : with_mean(with_mean), with_std(with_std), is_fit(false) {
    // constructor
}


standard_scaler::~standard_scaler() {
    // destructor
}


void standard_scaler::fit(nc::NdArray<double>& X) {
    means = nc::mean(X, nc::Axis::ROW);
    stds = nc::stdev(X, nc::Axis::ROW);

    is_fit = true;

}


nc::NdArray<double> standard_scaler::transform(nc::NdArray<double>& X) {

    if (!is_fit) {
        throw std::runtime_error("Scaler has not been fit. Please call the fit() method before attempting to call the transform() method.");
    }

    nc::NdArray<double> scaled_X = X;

    if (with_mean) {
        scaled_X -= means;
    }

    if (with_std) {
        scaled_X /= stds;
    }

    return scaled_X;

}


nc::NdArray<double> standard_scaler::fit_transform(nc::NdArray<double>& X) {
    fit(X);
    return transform(X);

}


nc::NdArray<double> standard_scaler::inverse_transform(nc::NdArray<double>& X) {
    if (!is_fit) {
        throw std::runtime_error("Scaler has not been fit. Please call the fit() method before attempting to call the inverse_transform() method.");
    }

    nc::NdArray<double> unscaled_X = X;

    if (with_std) {
        unscaled_X *= stds;
    }

    if (with_mean) { 
        unscaled_X += means; 
    }

    return unscaled_X;

}


nc::NdArray<double> standard_scaler::get_means() {
    return means;
}


nc::NdArray<double> standard_scaler::get_stds() {
    return stds;
}