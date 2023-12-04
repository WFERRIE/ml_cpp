#include "NumCpp.hpp"
#include <iostream>
#include "../include/minmax_scaler.hpp"
#include "../include/validation.hpp"




minmax_scaler::minmax_scaler(int feature_min, int feature_max) : feature_min(feature_min), feature_max(feature_max) {
    // constructor
    // checks if feautre_min and feature_max are valid, otherwise throws a runetime error
    if (feature_min >= feature_max) {
        throw std::runtime_error("Feature minimum must be less than feature maximum");
    }
}

minmax_scaler::~minmax_scaler() {
    // destructor
}

void minmax_scaler::fit(nc::NdArray<double>& X) {

    min_vals = nc::min(X, nc::Axis::ROW);
    max_vals = nc::max(X, nc::Axis::ROW);    

    is_fit = true;

}

nc::NdArray<double> minmax_scaler::transform(nc::NdArray<double>& X) {
    if (!is_fit) {
        throw std::runtime_error("Scaler has not been fit. Please call the fit() or fit_transform() method before attempting to call the transform() method.");
    }

    auto X_scaled = (double)feature_min + (X - min_vals) * (double)(feature_max - feature_min) / (max_vals - min_vals);

    X_scaled = replace_nan(X_scaled, 0.0);

    return X_scaled;

}

nc::NdArray<double> minmax_scaler::fit_transform(nc::NdArray<double>& X) {
    fit(X);
    return transform(X);

}

nc::NdArray<double> minmax_scaler::inverse_transform(nc::NdArray<double>& X) {
    if (!is_fit) {
        throw std::runtime_error("Scaler has not been fit. Please call the fit() or fit_transform() method before attempting to call the inverse_transform() method.");
    }

    auto X_unscaled = (X - (double)feature_min) * (max_vals - min_vals) / (double)(feature_max - feature_min) + min_vals;
    return X_unscaled;

}
