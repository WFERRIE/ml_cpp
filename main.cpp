#include "NumCpp.hpp"
#include <iostream>
#include "utils/utils.h"
#include "models/logistic_regression.h"
#include "metrics/metrics.h"
#include "models/kmeans.h"


double calculate_distance(const nc::NdArray<double>& point1, const nc::NdArray<double>& point2) {
    // calculate the euclidean distance between two points
    return nc::norm<double>(point1 - point2)(0, 0);

}


int main() {

    std::vector<std::vector<double>> data = read_csv("data/iris.csv");

    const nc::uint32 n_samples = 150;
    const nc::uint32 n_features = 5; // this counts the labels as a feature, so n_features-1 input features, 1 output feature

    auto matrix = nc::NdArray<double>(n_samples, n_features);

    for(nc::uint32 row = 0; row < n_samples; ++row) {
        for (nc::uint32 col = 0; col < n_features; ++col) {
            matrix(row, col) = data[row][col];
        }
    }

    matrix = fisher_yates_shuffle(matrix, n_samples);


    auto y = matrix(matrix.rSlice(), n_features - 1);
    y--; // set labels to be 0, 1, 2 instead of 1, 2, 3
    auto X = matrix(matrix.rSlice(), {0, n_features - 1});


    const int n_clusters = 3;
    const int max_iter = 100; 
    const double tol = 0.001;

    kmeans k = kmeans(n_clusters, max_iter, tol);

    auto final_labels = k.fit(X, false);

    std::cout << final_labels << std::endl;

    std::cout << y << std::endl;

    // logistic_regression logit_reg(100000, 0.01, 1);
    // logit_reg.fit(X, y, false);
    // std::cout << logit_reg.get_bias() << std::endl;
    // std::cout << logit_reg.get_weights() << std::endl;

    // auto y_pred = logit_reg.predict(X);

    // std::cout << confusion_matrix(y, y_pred) << std::endl;
    // std::cout << accuracy_score(y, y_pred) << std::endl;

    // std::cout << "f1 score:" << std::endl;
    // std::cout << f1_score(y, y_pred) << std::endl;

    // std::cout << precision_score(y, y_pred) << std::endl;
    // std::cout << recall_score(y, y_pred) << std::endl;

    // nc::NdArray<int> a1 = { { 1, 2 }, { 3, 4 }, { 5, 6 } };

    // std::cout << a1 << std::endl;

    // std::cout << a1(a1.rSlice(), 0) << std::endl;

    // std::cout << nc::sum<int>(a1(a1.rSlice(), 0)) << std::endl;
    // std::cout << nc::sum<int>(a1(a1.rSlice(), 1)) << std::endl;

    // std::cout << nc::sum<int>(a1(0, a1.cSlice()))(0, 0) << std::endl;
    // std::cout << nc::sum<int>(a1(1, a1.cSlice()))(0, 0) << std::endl;
    // std::cout << nc::sum<int>(a1(2, a1.cSlice()))(0, 0) << std::endl;
    

    return 0;

}
