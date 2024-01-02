#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../include/kmeans.hpp"
#include "NumCpp.hpp"
#include <iostream>
#include <tuple>
#include <set>
#include <map>
#include <vector>
#include "../include/utils.hpp"
#include "../include/dataset.hpp"
#include "../include/metrics.hpp"

TEST_CASE("kmeans Test", "[K Means Clustering]") {

    SECTION (".get_centroids()") {

        // test model is able to fit to some data

        nc::NdArray<double> X_train = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {10, 10, 10}, {10, 10, 10}, {10, 10, 10}};


        // testing out kmeans
        const int n_clusters = 2;
        const int max_iter = 10000; 
        const double tol = 0.000001;
        const bool verbose = false;

        kmeans k = kmeans(n_clusters, max_iter, tol);

        k.fit(X_train, verbose);

        nc::NdArray<double> centroids = k.get_centroids();

        nc::NdArray<double> expected_centroids = {{10, 10, 10}, {1, 1, 1}};

        int elements_correct = nc::sum<int>(nc::equal(expected_centroids, centroids).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected_centroids.shape().cols * expected_centroids.shape().rows);



    }


    SECTION (".fit()") {

        // test model is able to fit to some data

        nc::NdArray<double> data = read_csv("../data/iris_binary.csv", true);

        dataset DS = dataset(data);

        DS.train_test_split(0.90, true);

        nc::NdArray<double> X_train = DS.get_X_train();
        nc::NdArray<double> y_train = DS.get_y_train();

        nc::NdArray<double> X_test = DS.get_X_test();
        nc::NdArray<double> y_test = DS.get_y_test();




        // testing out kmeans
        const int n_clusters = 2;
        const int max_iter = 10000; 
        const double tol = 0.000001;
        const bool verbose = false;

        kmeans k = kmeans(n_clusters, max_iter, tol);

        k.fit(X_train, verbose);

        auto y_pred = k.predict(X_train);

        nc::NdArray<double> y_pred_double = y_pred.astype<double>(); // cast to doubles so we can use acc_score

        /*
        We can't really assess the accuracy of k means using the metrics i've implemented so far, as the labels assigned
        are not deterministic. The model could cluster perfectly but use different values for the cluster labels than that
        of the true data, thus resulting in 0% accuracy. This is on my todo list to fix, but in the meantime I'm going
        to use the following workaround:

        we take every true label, multiply it by 10, and then add the predicted label. This will form a "prediction ID" 
        where the 10s column is the true label value and the 1s column is the predicted value (this doesnt work if we 
        have more than 10 classes but for now its ok). We then add all of these to a vector which we then iterate through,
        counting each prediction ID occurence and putting them in a map. Then we can just print out the map to see the
        preidctions. For example, from my testing we get the following results from our map:

        Prediction ID: 10, Count: 2
        Prediction ID: 11, Count: 87
        Prediction ID: 20, Count: 46

        This implies that the model clustered 87 + 46 = 133 samples correctly, and 2 incorrectly, 
        for an accuracy of 133 / 135 = 98.5%
        */

        std::vector<int> vec;

        int n_samples = y_pred.shape().rows;

        for (int i = 0; i < n_samples; i++) {
            int v = (y_train(i, 0) + 1) * 10 + y_pred(i, 0);
            vec.push_back(v);
        }

        std::map<int, int> label_counts;

        // // Iterate through the values and update the counts in the map
        for (int value : vec) {
            // If the value is not in the map, insert it with a count of 1
            if (label_counts.find(value) == label_counts.end()) {
                label_counts[value] = 1;
            } else {
                // If the value is already in the map, increment the count
                label_counts[value]++;
            }
        }

        // // Print the counts
        // for (const auto& pair : label_counts) {
        //     std::cout << "Prediction ID: " << pair.first << ", Count: " << pair.second << std::endl;
        // }

    }
}
