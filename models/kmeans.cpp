#include "NumCpp.hpp"
#include <iostream>
#include "kmeans.h"


kmeans::kmeans(const int n_clusters, const int max_iter, const double tol) : n_clusters(n_clusters), max_iter(max_iter), tol(tol){
    // constructor
}

kmeans::~kmeans() {
    // destructor
}


nc::NdArray<double> kmeans::initialize_centroids(nc::NdArray<double>& X, const int n_clusters) {

    // randomly initialize n_clusters points as the centroids

    int n_samples = X.shape().rows;
    int n_features = X.shape().cols;
    nc::NdArray<double> centroids = nc::zeros<double>(n_clusters, n_features);

    for (int i = 0; i < n_clusters; i++) {
        int randInt = nc::random::randInt<int>(0, n_samples); // pick a integer between 0 and the number of data points in X
        auto c = X(randInt, X.cSlice()); // pick a random point in our dataset
        centroids.put(i, centroids.cSlice(), c); // put it in our centroids matrix
    }

    return centroids;    
}


double kmeans::calculate_distance(const nc::NdArray<double>& point1, const nc::NdArray<double>& point2) {
    // calculate the euclidean distance between two points
    return nc::norm<double>(point1 - point2)(0, 0);

}

nc::NdArray<nc::uint32> kmeans::assign_labels(nc::NdArray<double>& X, nc::NdArray<double>& clusters) {
    // assign all points to their nearest centroid

    int n_samples = X.shape().rows;
    int n_clusters = clusters.shape().rows;

    nc::NdArray<double> distances = nc::zeros<double>(n_samples, n_clusters); // (# points, # centroids)

    for (int i = 0; i < n_samples; i++) {
        // for each point
        for (int j = 0; j < n_clusters; j++) {
            // for each centroid

            double dist = calculate_distance(clusters(j, clusters.cSlice()), X(i, X.cSlice()));

            distances.put(i, j, dist);

        }
    }

    auto labels = nc::argmin(distances, nc::Axis::COL).transpose();

    return labels;
}

nc::NdArray<double> kmeans::update_centroids(nc::NdArray<double>& X, const int n_clusters, nc::NdArray<nc::uint32> labels) {
    // move centroids to their new location

    int n_samples = X.shape().rows;
    int n_features = X.shape().cols;

    nc::NdArray<double> new_centroids = nc::zeros<double>(n_clusters, n_features);

    nc::NdArray<int> label_counts = nc::zeros<int>(n_clusters, 1);

    for (int i = 0; i < n_samples; i++) {
        // for each point

        auto point = X(i, X.cSlice());

        int l = labels(i, 0);

        label_counts.put(l, 0, label_counts(l, 0) + 1); // incrememnt the number of times we've seen label l

        auto _new_centroid = new_centroids(l, new_centroids.cSlice()); // grab the centroid corresponding to the label

        _new_centroid += point;

        new_centroids.put(l, new_centroids.cSlice(), _new_centroid);

    }

    new_centroids = new_centroids / label_counts.astype<double>();

    return new_centroids;
}

nc::NdArray<nc::uint32> kmeans::fit(nc::NdArray<double>& X, bool verbose) {
    // fit data X on all centroids     

    nc::NdArray<nc::uint32> labels;
    nc::NdArray<double> centroids;

    for (int i = 0; i < max_iter; i++) {
        centroids = initialize_centroids(X, n_clusters);

        labels = assign_labels(X, centroids);

        centroids = update_centroids(X, n_clusters, labels);

    }

    std::cout << "FINAL CENTROIDS:" << centroids << std::endl;

    return labels;
    

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

}

void kmeans::predict(nc::NdArray<double>& X) {

    // change return type
    
}
