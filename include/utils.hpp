#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <string>
#include "NumCpp.hpp"

nc::NdArray<double> read_csv(const std::string& filename);

template<typename T>
T get_most_frequent_element(std::vector<T>& vec);

#endif
