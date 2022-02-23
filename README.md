# ATCS-Neural-Networks
This repository contains work I've done for my ATCS Neural Networks class.

## Input File Structures
Each input file for the executable of Network.cpp is expected to be in the same directory as the
executable.

The input configuration parameter file is expected to contain numbers separated by whitespace in the
following order: the number of connectivity layers in the network, the number of input activation
nodes, the number of nodes in the hidden layer, the number of nodes in the output layer, the number
of test cases in the truth table, and a boolean that is 1 if the network is in training mode or 0
if the network is in running mode. If the network is in training mode, the following numbers should
be included in the following order: the value of the learning rate, the value of the error
threshold to use where the network will cease training if the largest error in a given iteration is
less than that error threshold, the maximum number of iterations for training, and a boolean that is
1 if the network should use random values for the initial weight values or 0 if the network should
pull initial weight values from a file.

The 
