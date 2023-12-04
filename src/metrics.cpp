#include "NumCpp.hpp"
#include "../include/metrics.hpp"
#include "../include/validation.hpp"
#include <algorithm> // std::max


// Classification 

double accuracy_score(nc::NdArray<double>& y_true, nc::NdArray<double>& y_pred) {
    /*
    Returns the accuracy of the predictions, defined as the number of 
    correct predictions / total number of predictions.

    Returned score is in the range [0.0, 1.0] where 1.0 is a perfect score.


    Parameters
    ----------
    y_true: nc::NdArray<double> of shape (1 x n_samples) or (n_samples x 1). Contains the ground-truth labels
    y_pred: nc::NdArray<double> of shape (1 x n_samples) or (n_samples x 1). Contains the predicted labels as returned by a classifier.


    Returns 
    ----------
    score: double. Represents number of correct classifications / total number of labels
    
    */
    
    y_true.reshape(-1, 1);
    y_pred.reshape(-1, 1);

    check_consistent_shapes(y_true, y_pred);

    int n_samples = y_true.shape().rows;

    nc::NdArray<bool> _equal = nc::equal(y_true, y_pred);
    
    int true_positive = nc::sum<int>(_equal.astype<int>())(0, 0);
    

    double score = (double)true_positive / (double)n_samples;
    return score;
}


nc::NdArray<double> confusion_matrix(nc::NdArray<double>& y_true, nc::NdArray<double>& y_pred) {
    /*
    Returns a n_classes x n_classes matrix
    Rows are true labels
    Columns are predicted labels
    In other words, the for confusion matrix C, the value at the i-th row and j-th
    column, C(i, j) is the number of observations in group i, predicted to be in group j.
    */

    /*
    Returns an n_classes x n_classes confusion matrix of the predictions. 
    Rows represent the true labels, and columns the predicted labels.

    For confusion matrix C, the value at the i-th row and j-th column, C(i, j) is the
    number of observations in group i, predicted to be in group j.

    i < n_classes, j < n_classes

    Parameters
    ----------
    y_true: nc::NdArray<double> of shape (1 x n_samples) or (n_samples x 1). Contains the ground-truth labels
    y_pred: nc::NdArray<double> of shape (1 x n_samples) or (n_samples x 1). Contains the predicted labels as returned by a classifier.


    Returns 
    ----------
    c_matrix: nc::NdArray<double> of shape (n_classes x n_classes). 

    
    Issues / to do
    ----------
    this function only works when the classes are continuous starting at zero. For example, [0, 1, 2] is a valid
    set of classes, however, [1, 88, 100] is going to cause issues. This needs to be fixed by normalizing the classes
    before processing them.
    
    */
    y_true.reshape(-1, 1);
    y_pred.reshape(-1, 1);

    check_consistent_shapes(y_true, y_pred);

    int n_samples = y_true.shape().rows;
    auto classes = nc::unique(nc::vstack({y_true, y_pred}));
    int n_classes = classes.shape().cols;

    nc::NdArray<double> c_matrix = nc::zeros<double>(n_classes, n_classes);

    for (int i = 0; i < n_samples; i++) {
        auto true_val = y_true(i, 0);
        auto predicted_val = y_pred(i, 0);
        int count = c_matrix(true_val, predicted_val); // get current value at C(i, j)
        count++; // incrememnt that value by one
        c_matrix.put(true_val, predicted_val, count); // put the incremented value back at C(i, j)
    }

    return c_matrix;

}


nc::NdArray<double> f1_score(nc::NdArray<double>& y_true, nc::NdArray<double>& y_pred) {

    y_true.reshape(-1, 1);
    y_pred.reshape(-1, 1);

    check_consistent_shapes(y_true, y_pred);

    auto classes = nc::unique(nc::vstack({y_true, y_pred}));
    int n_classes = classes.shape().cols;

    nc::NdArray<double> c_matrix = confusion_matrix(y_true, y_pred);
    // it may be better and more robust to manually calculate the precisions
    // and recalls, but for now I'm just going to base them off of the confusion
    // matrix. 

    nc::NdArray<double> f1_output = nc::zeros<double>(1, n_classes);

    for (int class_idx = 0; class_idx < n_classes; class_idx++) {

        double precision;
        double recall;
        double f1;

        double true_positive = c_matrix(class_idx, class_idx);

        double precision_denom = nc::sum<double>(c_matrix(c_matrix.rSlice(), class_idx))(0, 0);
        double recall_denom = nc::sum<double>(c_matrix(class_idx, c_matrix.cSlice()))(0, 0);


        // the following is a bit messy and should be cleaned up. The idea is just to handle for
        // 0s in the denominators, otherwise it will set the f1 as -nan, which is okay, but then
        // causes issues during testing. Instead we will just set them to 0 in this case to make
        // testing easier.
        if (precision_denom == 0.0) {
            precision = 0.0;
        }
        else {
            precision = true_positive / precision_denom; 
        }

        if (recall_denom == 0.0) {
            recall = 0.0;
        }

        else {
            recall = true_positive / recall_denom;
        }

        if ((precision + recall) == 0) {
            f1 = 0;
        }
        else {
            f1 = (2 * precision * recall) / (precision + recall);
        }
        

        f1_output.put(0, class_idx, f1);
    }

    return f1_output;

}

nc::NdArray<double> precision_score(nc::NdArray<double>& y_true, nc::NdArray<double>& y_pred) {
    y_true.reshape(-1, 1);
    y_pred.reshape(-1, 1);

    check_consistent_shapes(y_true, y_pred);

    auto classes = nc::unique(nc::vstack({y_true, y_pred}));
    int n_classes = classes.shape().cols;

    nc::NdArray<double> c_matrix = confusion_matrix(y_true, y_pred);

    nc::NdArray<double> precision_output = nc::zeros<double>(1, n_classes);

    for (int class_idx = 0; class_idx < n_classes; class_idx++) {
        double precision;

        double true_positive = c_matrix(class_idx, class_idx);

        double precision_denom = nc::sum<double>(c_matrix(c_matrix.rSlice(), class_idx))(0, 0);

        if (precision_denom == 0) {
            precision = 0;
        }
        else {
            precision = true_positive / precision_denom;
        }

        precision_output.put(0, class_idx, precision);
    }

    return precision_output;

}


nc::NdArray<double> recall_score(nc::NdArray<double>& y_true, nc::NdArray<double>& y_pred) {

    y_true.reshape(-1, 1);
    y_pred.reshape(-1, 1);

    check_consistent_shapes(y_true, y_pred);

    auto classes = nc::unique(nc::vstack({y_true, y_pred}));
    int n_classes = classes.shape().cols;

    nc::NdArray<double> c_matrix = confusion_matrix(y_true, y_pred);

    nc::NdArray<double> recall_output = nc::zeros<double>(1, n_classes);

    for (int class_idx = 0; class_idx < n_classes; class_idx++) {
        double recall;

        double true_positive = c_matrix(class_idx, class_idx);

        double recall_denom = nc::sum<double>(c_matrix(class_idx, c_matrix.cSlice()))(0, 0);

        if (recall_denom == 0) {
            recall = 0;
        }
        else {
            recall = true_positive / recall_denom;
        }

        recall_output.put(0, class_idx, recall);
    }

    return recall_output;

}


// Regression

double max_error(nc::NdArray<double>& y_true, nc::NdArray<double>& y_pred) {

    nc::NdArray<double> error = y_true - y_pred;

    return nc::max(error)(0, 0);
}

double mean_absolute_error(nc::NdArray<double>& y_true, nc::NdArray<double>& y_pred) {
    
    nc::NdArray<double> error = y_true - y_pred;

    return nc::mean(nc::abs(error))(0, 0);

}

double mean_squared_error(nc::NdArray<double>& y_true, nc::NdArray<double>& y_pred) {

    nc::NdArray<double> error = y_true - y_pred;

    return nc::mean(nc::power<double>(error, 2))(0, 0);
}