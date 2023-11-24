#ifndef METRICS_METRICS_H
#define METRICS_METRICS_H

#include "NumCpp.hpp"

double accuracy_score(nc::NdArray<double> y_true, nc::NdArray<double> y_pred);

nc::NdArray<double> confusion_matrix(nc::NdArray<double> y_true, nc::NdArray<double> y_pred);

nc::NdArray<double> f1_score(nc::NdArray<double> y_true, nc::NdArray<double> y_pred);

#endif