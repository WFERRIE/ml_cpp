#ifndef METRICS_METRICS_H
#define METRICS_METRICS_H

#include "NumCpp.hpp"

// classification

double accuracy_score(nc::NdArray<double> y_true, nc::NdArray<double> y_pred);

nc::NdArray<double> confusion_matrix(nc::NdArray<double> y_true, nc::NdArray<double> y_pred);

nc::NdArray<double> f1_score(nc::NdArray<double> y_true, nc::NdArray<double> y_pred);

nc::NdArray<double> precision_score(nc::NdArray<double> y_true, nc::NdArray<double> y_pred);

nc::NdArray<double> recall_score(nc::NdArray<double> y_true, nc::NdArray<double> y_pred);

// regression

nc::NdArray<double> max_error(nc::NdArray<double> y_true, nc::NdArray<double> y_pred);

nc::NdArray<double> mean_absolute_error(nc::NdArray<double> y_true, nc::NdArray<double> y_pred);

nc::NdArray<double> mean_squared_error(nc::NdArray<double> y_true, nc::NdArray<double> y_pred);

#endif