#include "NumCpp.hpp"
#include <iostream>
#include "kmeans.h"
#include <vector>


kmeans::kmeans(const int n_clusters, const int max_iter, const double tol) : n_clusters(n_clusters), max_iter(max_iter), tol(tol){
    // constructor
}

kmeans::~kmeans() {
    // destructor
}


std::vector<nc::NdArray<double>> kmeans::initialize_centroids(nc::NdArray<double>& X, const int n_clusters) {

    // randomly initialize n_clusters points as the centroids
    std::vector<nc::NdArray<double>> centroids;

    int n_samples = X.shape().rows;

    for (int i = 0; i < n_clusters; i++) {
        int randInt = nc::random::randInt<int>(0, n_samples); // pick a integer between 0 and the number of data points in X
        auto c = X(randInt, X.cSlice()); // pick a random point in our dataset
        centroids.push_back(c);
    }

    for (auto i = centroids.begin(); i != centroids.end(); i++) {
        std::cout << *i << std::endl;

    }

    return centroids;    
}


double kmeans::calculate_distance(const nc::NdArray<double>& point1, const nc::NdArray<double>& point2) {
    // calculate the euclidean distance between two points
    return nc::norm<double>(point1 - point2)(0, 0);

}

void kmeans::assign_labels() {
    // assign all points to their nearest centroid
}

void kmeans::update_centroids() {
    // move centroids to their new location
}

void kmeans::fit(nc::NdArray<double>& X, bool verbose) {
    // fit data X on all centroids     

    initialize_centroids(X, n_clusters); 

    int n_samples = X.shape().rows;

    // create array of size n_samples x n_clusters where each element is the distance
    // between that sample and that cluster

    auto distances = nc::zeros<double>(n_samples, n_clusters);

    // then create an array called assignment of size n_samples x 1 where each element is the
    // int assignment of that sample to a cluster, so itll be in the range [0, n_clusters]

    // then start the training loop. In each iteration do the following:
    // 1. move the centroids to the average location of all points assigned to it
    //      update_centroid_locations()
    // 2. recalculate the assignments
    //      assign_labels()
    // 3. move the centroids to their new average location
    //      move_centroids()
    // 4. check stopping criterion, if not, repeat
    //      check_stopping_criterion()


    // for (int i = 0; i < max_iter; i++) {
    //     // assign_labels()
    //     // calculate all distances
    //     // update centroids
    // }

}

void kmeans::predict(nc::NdArray<double>& X) {

    // change return type
    
}
