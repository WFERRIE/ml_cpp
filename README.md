# ML_CPP

Welcome to ml_cpp, a personal project aimed at creating a simple but powerful machine learning library in c++. Drawing design inspiration from [scikit-learn](https://scikit-learn.org/stable/), I'm working on this project to better understand the math underlying various statistical and machine learning algorithms, and to keep up on my c++ abilities.

Note that this project is still very much a work in progress and therefore has no release yet, however, if you'd like to see the features implemented so far, check out [the documentation](https://github.com/WFERRIE/ml_cpp/tree/main/docs).

 If you're interested in playing around with this project, run the following:

    git clone git@github.com:WFERRIE/ml_cpp.git
    cd ml_cpp
    mkdir build
    cd build
    cmake ..
    make

At this point, you can run either of the following to execute the [regression](https://github.com/WFERRIE/ml_cpp/blob/main/examples/regression.cpp) and [classification](https://github.com/WFERRIE/ml_cpp/blob/main/examples/classification.cpp) examples.

    ./regression
    ./classification

Note that this project requires the installation of both [NumCpp](https://github.com/dpilger26/NumCpp/) and [Catch2](https://github.com/catchorg/Catch2).
