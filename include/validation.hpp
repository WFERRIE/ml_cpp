#ifndef VALIDATION_HPP
#define VALIDATION_HPP
#include "NumCpp.hpp"

void check_consistent_shapes(nc::NdArray<double>& a, nc::NdArray<double>& b);


int get_n_samples(nc::NdArray<double>& y_true);


#endif
