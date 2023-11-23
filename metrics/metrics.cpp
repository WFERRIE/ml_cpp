#include "NumCpp.hpp"
#include "metrics.h"

double accuracy_score(nc::NdArray<double> y_true, nc::NdArray<double> y_pred) {
    /*
    Returns the accuracy of the predictions, defined as the number of 
    correct predictions / total number of predictions.

    Returned score is in the range [0.0, 1.0] where 1.0 is a perfect score.
    */
    int n_predictions = y_pred.shape().rows;
    int true_positive = 0;

    for (int i = 0; i < n_predictions; i++) {
        if (y_true(i, 0) == y_pred(i, 0)) {
            true_positive++;
        }
    }

    return (double)true_positive / (double)n_predictions;
}


nc::NdArray<double> confusion_matrix(nc::NdArray<double> y_true, nc::NdArray<double> y_pred) {
    /*
    Returns a n_classes x n_classes matrix
    Rows are true labels
    Columns are predicted labels
    Top left element is the number of true label = 0 which were correctly predicted as 0
    Top right element is the number of true label = 0 which were incorrectly predicted as n_classes
    Bottom left element is the number of true label = n_classes which were incorrectly predicted as 0
    Bottom right element is the number of true label = n_classes which were correctly predicted as 0
    */

    int n_predictions = y_true.shape().rows;
    auto classes = nc::unique(y_true);
    int n_classes = classes.shape().cols;

    auto confusion_matrix = nc::zeros<double>(n_classes, n_classes);


    for (int i = 0; i < n_predictions; i++) {
        
        auto true_val = y_true(i, 0);
        auto predicted_val = y_pred(i, 0);
        
        auto count = confusion_matrix(true_val, predicted_val);
        count++;

        confusion_matrix.put(true_val, predicted_val, count);

    }

    return confusion_matrix;

}