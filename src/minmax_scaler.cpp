#include "NumCpp.hpp"
#include <iostream>
#include "../include/minmax_scaler.hpp"
#include "../include/validation.hpp"




minmax_scaler::minmax_scaler(double feature_min, double feature_max) : feature_min(feature_min), feature_max(feature_max) {
    /*
    Transforms the data by scaling the features to provided range (feature_min, feature_max). Defaults to (0, 1).

    Parameters
    ----------
    feature_min: The desired minimum value for the transformed data. All data will be greater than or equal to this value.
    feature_max: The desired maximum value for the transformed data. All data will be less than or equal to this value. 

    */

    if (feature_min >= feature_max) {
        throw std::runtime_error("Feature minimum must be less than feature maximum");
    }
}

minmax_scaler::~minmax_scaler() {
    // destructor
}


void minmax_scaler::fit(nc::NdArray<double>& X) {
    /*
    Fits the scaler on the provided data X. Fitting in this case is just getting the minimum and maximum values
    for each feature.

    Parameters
    ----------
    X: Array to fit the scaler on. 


    Returns
    ---------
    Nothing, however, sets the is_fit flag to true, allowing for use of .transform(...), .inverse_transform(...), and getters.
    */

    min_vals = nc::min(X, nc::Axis::ROW);
    max_vals = nc::max(X, nc::Axis::ROW);    

    is_fit = true;

}

nc::NdArray<double> minmax_scaler::transform(nc::NdArray<double>& X) {
    /*
    Transforms the data based on feature max and mins calculated from the .fit(...) method.
    Note that this method may not be called until either the .fit(...) or .fit_transform(...) method has been called.

    Parameters
    ----------
    X: Array to transform.


    Returns
    ---------
    X_scaled: Array that has been transformed.
    */

    if (!is_fit) {
        throw std::runtime_error("Scaler has not been fit. Please call the fit() or fit_transform() method before attempting to call the transform() method.");
    }

    auto X_scaled = (double)feature_min + (X - min_vals) * (double)(feature_max - feature_min) / (max_vals - min_vals);

    X_scaled = replace_nan(X_scaled, feature_min);

    return X_scaled;

}

nc::NdArray<double> minmax_scaler::fit_transform(nc::NdArray<double>& X) {
    /*
    Fits the minmax scaler on X, and then transforms X. Equivalent to calling .fit(...) and then .transform(...).

    Parameters
    ----------
    X: Array to fit scaler on and transform.


    Returns
    ---------
    X_scaled: Array that has been transformed.
    */

    fit(X);
    return transform(X);

}

nc::NdArray<double> minmax_scaler::inverse_transform(nc::NdArray<double>& X) {
    /*
    Re-scales the input X based on the feature maxs and mins calculated during the .fit(...) method.
    Cannot be called until either .fit(...) or .fit_transform(...) are first called.

    Parameters
    ----------
    X: Array to inverse transform.


    Returns
    ---------
    X_unscaled: Array that has been inverse transformed.
    */

    if (!is_fit) {
        throw std::runtime_error("Scaler has not been fit. Please call the fit() or fit_transform() method before attempting to call the inverse_transform() method.");
    }

    auto X_unscaled = (X - (double)feature_min) * (max_vals - min_vals) / (double)(feature_max - feature_min) + min_vals;
    return X_unscaled;

}

nc::NdArray<double> minmax_scaler::get_min_vals() {
    /*
    Getter to return the feature mins calculated from the .fit(...) method. Cannot be called until
    either the .fit(...) or .fit_transform(...) methods have been called.

    Returns
    ---------
    min_vals: The minimum values for each feature in the data passed to the .fit(...) method.
    */

    if (!is_fit) {
        throw std::runtime_error("Scaler has not been fit. Please call the fit() or fit_transform() method before attempting to call any getters.");
    }

    return min_vals;
    
}

nc::NdArray<double> minmax_scaler:: get_max_vals() {
    /*
    Getter to return the feature maxs calculated from the .fit(...) method. Cannot be called until
    either the .fit(...) or .fit_transform(...) methods have been called.

    Returns
    ---------
    max_vals: The maximum values for each feature in the data passed to the .fit(...) method.
    */

    if (!is_fit) {
        throw std::runtime_error("Scaler has not been fit. Please call the fit() or fit_transform() method before attempting to call any getters.");
    }

    return max_vals;
}
