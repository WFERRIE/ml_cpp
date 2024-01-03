#include "NumCpp.hpp"
#include <iostream>
#include <string>
#include "../include/utils.hpp"
#include "../include/dataset.hpp"
#include "../include/metrics.hpp"
#include "../include/standard_scaler.hpp"
#include "../include/validation.hpp"
#include "../include/randomforest_classifier.hpp"


int main() {
    // demo

    std::cout << "Runing Classification Example..." << std::endl;

    std::cout << "Importing data." << std::endl;
    // import the data
    nc::NdArray<double> data = read_csv("../data/tree_data_fixed.csv", false);
    
    
    std::cout << "Loading data into dataset object and splitting into train/test sets." << std::endl;
    dataset DS = dataset(data);
    DS.train_test_split(0.05, true); // only need 5% of data for this example or else itll take too long to train.

    nc::NdArray<double> X_train_unscaled = DS.get_X_train();
    nc::NdArray<double> y_train = DS.get_y_train();

    nc::NdArray<double> X_test_unscaled = DS.get_X_test();
    nc::NdArray<double> y_test = DS.get_y_test();



    std::cout << "Performing data scaling." << std::endl;
    // perform data scaling
    standard_scaler ss = standard_scaler();
    nc::NdArray<double> X_train = ss.fit_transform(X_train_unscaled);
    nc::NdArray<double> X_test = ss.fit_transform(X_test_unscaled);


    // create and train model
    int n_estimators = 4;
    int max_depth = 5;
    int min_samples_split = 10;
    int max_features = 5;
    bool verbose = false;

    randomforest_classifier rfc = randomforest_classifier(n_estimators, max_depth, min_samples_split, max_features);

    std::cout << "Begin training Random Forest Classifier..." << std::endl;
    rfc.fit(X_train, y_train, verbose);

    auto y_pred = rfc.predict(X_test);


    std::cout << "Evaluating model performance on test set:" << std::endl;
    // evaluate model performance
    double acc = accuracy_score(y_test, y_pred);
    nc::NdArray<double> f1 = f1_score(y_test, y_pred);
    nc::NdArray<double> c_matrix = confusion_matrix(y_test, y_pred);

    std::cout << "Accuracy: " << acc << std::endl;
    std::cout << "F1: " << f1 << std::endl;
    std::cout << "Confusion Matrix: " << std::endl;
    std::cout <<  c_matrix << std::endl;

    return 0;

}
