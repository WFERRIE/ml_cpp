#include "NumCpp.hpp"
#include <iostream>
#include <string>
#include "include/utils.hpp"
#include "include/dataset.hpp"
#include "include/linear_regression.hpp"
#include "include/logistic_regression.hpp"
#include "include/metrics.hpp"
#include "include/kmeans.hpp"
#include "include/standard_scaler.hpp"
#include "include/minmax_scaler.hpp"
#include "include/validation.hpp"


int main() {


    nc::NdArray<double> y_true = {1.0, 1.0, 0.0, 1.0, 0.0};
    nc::NdArray<double> y_pred = {1.0, 1.0, 1.0, 0.0, 0.0};
    
    std::cout << accuracy_score(y_true, y_pred) << std::endl;
    std::cout << accuracy_score(y_true.transpose(), y_pred.transpose()) << std::endl;
    // std::cout << nc::sum(y_true2 == y_pred2) << std::endl;

    // // testing out linear regression
    // const std::string penalty = "l1";
    // const double reg_strength = 1.0;
    // const int max_iters = 1000000;
    // const double lr = 0.01;
    // const double tol = 0.00001;
    // const int init_mode = 1;



    // standard_scaler ss = standard_scaler();
    // auto X_scaled = ss.fit_transform(X);
    // linear_regression lin_reg = linear_regression(penalty, reg_strength, max_iters, lr, tol, init_mode);
    // lin_reg.fit(X_scaled, y, true);
    // std::cout << lin_reg.get_bias() << std::endl;
    // std::cout << lin_reg.get_weights() << std::endl;

    // nc::NdArray<double> y_pred = lin_reg.predict(X_scaled);

    // std::cout << "max error: " << max_error(y, y_pred) << std::endl;

    // std::cout << "MAE: " << mean_absolute_error(y, y_pred) << std::endl;

    // std::cout << "MSE: " << mean_squared_error(y, y_pred) << std::endl;



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
