#ifndef UTILS_UTILS_H
#define UTILS_UTILS_H

#include <vector>
#include <string>
#include "NumCpp.hpp"

std::vector<std::vector<double>> read_csv(const std::string& filename);

nc::NdArray<double> fisher_yates_shuffle(nc::NdArray<double> input, const int& n_samples);

#endif
