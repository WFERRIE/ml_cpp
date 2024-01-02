#include "../include/dataset.hpp"
#include "NumCpp.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

dataset::dataset(nc::NdArray<double> data) : data(data) {
    // Constructor
    // Stores data in the data attribute.
}
dataset::~dataset() {
    // Destructor
}


nc::NdArray<double> dataset::fisher_yates_shuffle(nc::NdArray<double>& input) {

    /*
    
    shuffle an array row-wise using the fisher - yates algorithm
    
    Parameters
    ----------
    input: nc::NdArray to be shuffled.

    Returns 
    ----------
    input: nc::NdArray - the original input but shuffled in place.

    Example
    ---------
    data = fisher_yates_shuffle(data);
    */
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

    /*
    Splits the array stored in this->data into the X, y, X_train, y_train, X_test, y_test attributes.

    Parameters
    ---------
    train_size: double in the range (0, 1) which which represents the percentage of data that will
    be assigned to the training sets. For example, if train_size = 0.8, 80% of the data will be assigned
    to the train set, and 20% will go to the test set.

    shuffle: boolean which indicates whether or not to shuffle the data before performing the split. Data 
    is shuffled row-wise to preserve data integrity and uses the fisher-yates shuffling algorithm.


    Returns
    ---------
    Does not return anything. Instead, sets the X, y, X_train, y_train, X_test, y_test attributes.
    These can then be accessed using the provided getter methods.


    Example
    ---------
    nc::NdArray<double> some_data = {{1, 1, 1, 1, -1},
                                    {2, 2, 2, 2, -2},
                                    {3, 3, 3, 3, -3},
                                    {4, 4, 4, 4, -4},
                                    {5, 5, 5, 5, -5},
                                    {6, 6, 6, 6, -6},
                                    {7, 7, 7, 7, -7},
                                    {8, 8, 8, 8, -8}};

    dataset DS = dataset(some_data);

    DS.train_test_split(0.8, false);

    nc::NdArray<double> X_train = DS.get_X_train();
    nc::NdArray<double> y_train = DS.get_y_train();
    nc::NdArray<double> X_test = DS.get_X_test();
    nc::NdArray<double> y_test = DS.get_y_test();
    */


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

    y_train = y({0, train_idx_end} , y.cSlice());
    y_test = y({train_idx_end, n_samples}, y.cSlice());

    X_train = X({0, train_idx_end} , X.cSlice());
    X_test = X({train_idx_end, n_samples}, X.cSlice());

    train_test_set = true;
}


nc::NdArray<double> dataset::get_X() {
    // Returns the X attribute.
    if (!train_test_set) {
        std::runtime_error("Error: please call .train_test_split() before attempting to use getters.");
    }
    return X;
}
nc::NdArray<double> dataset::get_y() {
    // Returns the y attribute.
    if (!train_test_set) {
        std::runtime_error("Error: please call .train_test_split() before attempting to use getters.");
    }
    return y;
}

nc::NdArray<double> dataset::get_X_train() {
    // Returns the X_train attribute.
    if (!train_test_set) {
        std::runtime_error("Error: please call .train_test_split() before attempting to use getters.");
    }
    return X_train;
}
nc::NdArray<double> dataset::get_y_train() {
    // Returns the y_train attribute.
    if (!train_test_set) {
        std::runtime_error("Error: please call .train_test_split() before attempting to use getters.");
    }
    return y_train;
}


nc::NdArray<double> dataset::get_X_test() {
    // Returns the X_test attribute.
    if (!train_test_set) {
        std::runtime_error("Error: please call .train_test_split() before attempting to use getters.");
    }
    return X_test;
}
nc::NdArray<double> dataset::get_y_test() {
    // Returns the y_test attribute.
    if (!train_test_set) {
        std::runtime_error("Error: please call .train_test_split() before attempting to use getters.");
    }
    return y_test;
}