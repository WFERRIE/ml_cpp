#include "NumCpp.hpp"
#include <iostream>
#include <string>
#include "../include/utils.hpp"
#include "../include/dataset.hpp"
#include "../include/metrics.hpp"
#include "../include/standard_scaler.hpp"
#include "../include/validation.hpp"
#include "../include/linear_regression.hpp"


int main() {
    // demo

    std::cout << "Runing Regression Example..." << std::endl;

    std::cout << "Importing data" << std::endl;
    // import the data
    nc::NdArray<double> data = read_csv("../data/diabetes_regression.csv", true);

    std::cout << "Loading data into dataset object and splitting into train/test sets" << std::endl;
    dataset DS = dataset(data);
    DS.train_test_split(0.75, true);

    nc::NdArray<double> X_train_unscaled = DS.get_X_train();
    nc::NdArray<double> y_train = DS.get_y_train();

    nc::NdArray<double> X_test_unscaled = DS.get_X_test();
    nc::NdArray<double> y_test = DS.get_y_test();


    std::cout << "Performing data scaling." << std::endl;
    // perform data scaling
    standard_scaler ss = standard_scaler();
    nc::NdArray<double> X_train = ss.fit_transform(X_train_unscaled);
    nc::NdArray<double> X_test = ss.fit_transform(X_test_unscaled);


    // create and train model
    const std::string penalty = "l2";
    const double reg_strength = 0.1;
    const int max_iters = 1000;
    const double lr = 0.01;
    const double tol = 0.0001;
    const int init_mode = 1;
    const bool verbose = false;

    
    linear_regression lin_reg = linear_regression(penalty, reg_strength, max_iters, lr, tol, init_mode);
    std::cout << "Begin training Linear Regressor..." << std::endl;
    lin_reg.fit(X_train, y_train, verbose);

    nc::NdArray<double> y_pred = lin_reg.predict(X_test);


    std::cout << "Evaluating model performance on test set:" << std::endl;
    // evaluate model performance
    double max_err = max_error(y_test, y_pred);
    double mae = mean_absolute_error(y_test, y_pred);
    double mse = mean_squared_error(y_test, y_pred);

    std::cout << "Max Error: " << max_err << std::endl;
    std::cout << "Mean Absolute Error: " << mae << std::endl;
    std::cout << "Mean Squared Error: " << mse << std::endl;

    return 0;

}
