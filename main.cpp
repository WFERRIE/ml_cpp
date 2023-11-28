#include "NumCpp.hpp"
#include <iostream>
#include <string>
#include "utils/utils.h"
#include "models/logistic_regression.h"
#include "metrics/metrics.h"
#include "models/kmeans.h"
#include "preprocessing/standard_scaler.h"
#include "preprocessing/minmax_scaler.h"


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





    // // testing out minmax scaler
    // nc::NdArray<double> a1 = { { 1, 2 }, { 3, 4 }, { 5, 6 } };


    // minmax_scaler mm = minmax_scaler();
    // mm.fit(a1);
    // auto a1_t = mm.transform(a1);
    // std::cout << a1_t << std::endl;
    // std::cout << mm.inverse_transform(a1_t) << std::endl;

    // std::cout << "========" << std::endl;

    // minmax_scaler mm2 = minmax_scaler(-5, 11);
    // auto a1_t2 = mm2.fit_transform(a1);
    // std::cout << a1_t2 << std::endl;
    // std::cout << mm2.inverse_transform(a1_t2) << std::endl;




    // // testing out standard scaler
    // nc::NdArray<double> a1 = { { 1, 2 }, { 3, 4 }, { 5, 6 } };
    // standard_scaler ss = standard_scaler(false, false);
    // auto a1_scaled = ss.fit_transform(a1);
    // std::cout << a1_scaled << std::endl;
    // a1_scaled = ss.inverse_transform(a1_scaled);
    // std::cout << a1_scaled << std::endl;

    // standard_scaler ssf = standard_scaler(true, false);
    // a1_scaled = ssf.fit_transform(a1);
    // std::cout << a1_scaled << std::endl;
    // a1_scaled = ssf.inverse_transform(a1_scaled);
    // std::cout << a1_scaled << std::endl;

    // std::cout << ss.get_means() << std::endl;
    // std::cout << ss.get_stds() << std::endl;
    // std::cout << ssf.get_means() << std::endl;
    // std::cout << ssf.get_stds() << std::endl;








    // // testing out kmeans
    // const int n_clusters = 3;
    // const int max_iter = 100; 
    // const double tol = 0.001;

    // kmeans k = kmeans(n_clusters, max_iter, tol);

    // auto final_labels = k.fit(X, false);

    // std::cout << final_labels << std::endl;

    // std::cout << y << std::endl;



    // // testing out logistic regression

    // const std::string penalty = "l1";
    // const double reg_strength = 0.1;
    // const int max_iters = 10000;
    // const double lr = 0.01;
    // const double tol = 0.01;
    // const int init_mode = 1;
    
    // logistic_regression logit_reg(penalty, reg_strength, max_iters, lr, tol, init_mode);
    // logit_reg.fit(X, y, false);

    // auto y_pred = logit_reg.predict(X);


    // // testing out evaluation metrics
    // std::cout << confusion_matrix(y, y_pred) << std::endl;
    // std::cout << accuracy_score(y, y_pred) << std::endl;

    // std::cout << "f1 score:" << std::endl;
    // std::cout << f1_score(y, y_pred) << std::endl;

    // std::cout << precision_score(y, y_pred) << std::endl;
    // std::cout << recall_score(y, y_pred) << std::endl;



    // // misc code
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
