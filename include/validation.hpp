#ifndef VALIDATION_HPP
#define VALIDATION_HPP
#include "NumCpp.hpp"

void check_consistent_shapes(nc::NdArray<double>& a, nc::NdArray<double>& b);

nc::NdArray<double> replace_nan(nc::NdArray<double>& X, double val);

#endif
