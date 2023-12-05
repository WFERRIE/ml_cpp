#include "NumCpp.hpp"
#include <iostream>
#include "../include/standard_scaler.hpp"
#include "../include/validation.hpp"


standard_scaler::standard_scaler(bool with_mean, bool with_std) : with_mean(with_mean), with_std(with_std){
    /*
    Constructor

    Parameters
    ---------
    with_mean: boolean variable. If set to true, transformed data will be scaled to zero mean.
    with_std: boolean variable. If set to true, transformed data will be scaled to unit standard deviation (equivalent to unit variance).
    */
}


standard_scaler::~standard_scaler() {
    // destructor
}


void standard_scaler::fit(nc::NdArray<double>& X) {
    /*
    Fits the standard scaler on the data X.

    Parameters
    ----------
    X: nc::NdArray<double> to fit the scaler on.


    Returns
    ---------
    Nothing, however, sets is_fit flag to true, allowing for use of .transform(...), .inverse_transform(...), and getters.
    */

    means = nc::mean(X, nc::Axis::ROW);
    stds = nc::stdev(X, nc::Axis::ROW);

    is_fit = true;

}


nc::NdArray<double> standard_scaler::transform(nc::NdArray<double>& X) {
    /*
    Transforms the data based on the means and standard deviations calculated from the .fit(...) method.
    Note that this method may not be called until either the .fit(...) or .fit_transform(...) method has been called.

    Parameters
    ----------
    X: Array to transform.


    Returns
    ---------
    X_scaled: Array that has been transformed.
    */

    if (!is_fit) {
        throw std::runtime_error("Scaler has not been fit. Please call the fit() method before attempting to call the transform() method.");
    }

    nc::NdArray<double> X_scaled = X;

    if (with_mean) {
        X_scaled -= means;
    }

    if (with_std) {
        X_scaled /= stds;
    }

    X_scaled = replace_nan(X_scaled, 0.0);

    return X_scaled;

}


nc::NdArray<double> standard_scaler::fit_transform(nc::NdArray<double>& X) {
    /*
    Fits the standard scaler on X, and then transforms X. Equivalent to calling .fit(...) and then .transform(...).

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


nc::NdArray<double> standard_scaler::inverse_transform(nc::NdArray<double>& X) {
    /*
    Re-scales the input X based on the mean and standard deviation calculated during the .fit(...) method.
    Cannot be called until either .fit(...) or .fit_transform(...) are first called.

    Parameters
    ----------
    X: Array to inverse transform.


    Returns
    ---------
    X_unscaled: Array that has been inverse transformed.
    */

    if (!is_fit) {
        throw std::runtime_error("Scaler has not been fit. Please call the fit() method before attempting to call the inverse_transform() method.");
    }

    nc::NdArray<double> X_unscaled = X;

    if (with_std) {
        X_unscaled *= stds;
    }

    if (with_mean) { 
        X_unscaled += means; 
    }

    return X_unscaled;

}


nc::NdArray<double> standard_scaler::get_means() {
    /*
    Getter to return the means attribute calculated from the .fit(...) method. Cannot be called until
    either the .fit(...) or .fit_transform(...) methods have been called.

    Returns
    ---------
    means: The means calcualted during the .fit(...) method.
    */

    if (!is_fit) {
        throw std::runtime_error("Scaler has not been fit. Please call the .fit() method before attempting to use getters.");
    }
    return means;
}


nc::NdArray<double> standard_scaler::get_stds() {
    /*
    Getter to return the standard deviation attribute calculated from the .fit(...) method. Cannot be called until
    either the .fit(...) or .fit_transform(...) methods have been called.

    Returns
    ---------
    stds: The standard deviations calcualted during the .fit(...) method.
    */

    if (!is_fit) {
        throw std::runtime_error("Scaler has not been fit. Please call the .fit() method before attempting to use getters.");
    }
    return stds;
}