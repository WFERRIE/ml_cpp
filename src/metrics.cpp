#include "NumCpp.hpp"
#include "../include/metrics.hpp"
#include "../include/validation.hpp"


// Classification 

double accuracy_score(nc::NdArray<double> y_true, nc::NdArray<double> y_pred) {
    /*
    Returns the accuracy of the predictions, defined as the number of 
    correct predictions / total number of predictions.

    Returned score is in the range [0.0, 1.0] where 1.0 is a perfect score.


    Parameters
    ----------
    y_true: 1d numcpp array. Contains the ground-truth labels
    y_pred: 1d numcpp array. Contains the predicted labels as returned by a classifier.


    Returns 
    ----------
    score: double. Represents number of correct classifications / total number of labels
    
    */
    

    check_consistent_shapes(y_true, y_pred);

    nc::NdArray<bool> _equal = nc::equal(y_true, y_pred);
    
    int true_positive = nc::sum<int>(_equal.astype<int>())(0, 0);
    
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
    

    double score = (double)true_positive / (double)n_predictions;
    return score;
}


nc::NdArray<double> confusion_matrix(nc::NdArray<double> y_true, nc::NdArray<double> y_pred) {
    /*
    Returns a n_classes x n_classes matrix
    Rows are true labels
    Columns are predicted labels
    In other words, the for confusion matrix C, the value at the i-th row and j-th
    column, C(i, j) is the number of observations in group i, predicted to be in group j.
    */

    int n_predictions = y_true.shape().rows;
    auto classes = nc::unique(y_true);
    int n_classes = classes.shape().cols;

    nc::NdArray<double> confusion_matrix = nc::zeros<double>(n_classes, n_classes);


    for (int i = 0; i < n_predictions; i++) {
        
        auto true_val = y_true(i, 0);
        auto predicted_val = y_pred(i, 0);
        
        int count = confusion_matrix(true_val, predicted_val);
        count++;

        confusion_matrix.put(true_val, predicted_val, count);

    }

    return confusion_matrix;

}


nc::NdArray<double> f1_score(nc::NdArray<double> y_true, nc::NdArray<double> y_pred) {

    int n_predictions = y_true.shape().rows;
    auto classes = nc::unique(y_true);
    int n_classes = classes.shape().cols;

    nc::NdArray<double> c_matrix = confusion_matrix(y_true, y_pred);
    // it may be better and more robust to manually calculate the precisions
    // and recalls, but for now I'm just going to base them off of the confusion
    // matrix. 

    nc::NdArray<double> f1_output = nc::zeros<double>(1, n_classes);

    for (int class_idx = 0; class_idx < n_classes; class_idx++) {

        double true_positive = c_matrix(class_idx, class_idx);

        double precision_denom = nc::sum<double>(c_matrix(c_matrix.rSlice(), class_idx))(0, 0);
        double recall_denom = nc::sum<double>(c_matrix(class_idx, c_matrix.cSlice()))(0, 0);

        double precision = true_positive / precision_denom;
        double recall = true_positive / recall_denom;
        
        double f1 = (2 * precision * recall) / (precision + recall);

        f1_output.put(0, class_idx, f1);
    }

    return f1_output;

}

nc::NdArray<double> precision_score(nc::NdArray<double> y_true, nc::NdArray<double> y_pred) {

    int n_predictions = y_true.shape().rows;
    auto classes = nc::unique(y_true);
    int n_classes = classes.shape().cols;

    nc::NdArray<double> c_matrix = confusion_matrix(y_true, y_pred);
    // it may be better and more robust to manually calculate the precisions
    // and recalls, but for now I'm just going to base them off of the confusion
    // matrix. 

    nc::NdArray<double> precision_output = nc::zeros<double>(1, n_classes);

    for (int class_idx = 0; class_idx < n_classes; class_idx++) {

        double true_positive = c_matrix(class_idx, class_idx);

        double precision_denom = nc::sum<double>(c_matrix(c_matrix.rSlice(), class_idx))(0, 0);

        double precision = true_positive / precision_denom;
        

        precision_output.put(0, class_idx, precision);
    }

    return precision_output;

}


nc::NdArray<double> recall_score(nc::NdArray<double> y_true, nc::NdArray<double> y_pred) {

    int n_predictions = y_true.shape().rows;
    auto classes = nc::unique(y_true);
    int n_classes = classes.shape().cols;

    nc::NdArray<double> c_matrix = confusion_matrix(y_true, y_pred);
    // it may be better and more robust to manually calculate the precisions
    // and recalls, but for now I'm just going to base them off of the confusion
    // matrix. 

    nc::NdArray<double> recall_output = nc::zeros<double>(1, n_classes);

    for (int class_idx = 0; class_idx < n_classes; class_idx++) {

        double true_positive = c_matrix(class_idx, class_idx);

        double recall_denom = nc::sum<double>(c_matrix(class_idx, c_matrix.cSlice()))(0, 0);

        double recall = true_positive / recall_denom;

        recall_output.put(0, class_idx, recall);
    }

    return recall_output;

}


// Regression

nc::NdArray<double> max_error(nc::NdArray<double> y_true, nc::NdArray<double> y_pred) {

    nc::NdArray<double> error = y_true - y_pred;

    return nc::max(error);
}

nc::NdArray<double> mean_absolute_error(nc::NdArray<double> y_true, nc::NdArray<double> y_pred) {
    
    nc::NdArray<double> error = y_true - y_pred;

    return nc::mean(nc::abs(error));

}

nc::NdArray<double> mean_squared_error(nc::NdArray<double> y_true, nc::NdArray<double> y_pred) {

    nc::NdArray<double> error = y_true - y_pred;

    return nc::mean(nc::power<double>(error, 2));
}