# ATCS-Neural-Networks
This repository contains work I've done for my ATCS Neural Networks class.


## Input File Structures
Each input file for the executable of Network.cpp is expected to be in the same directory as the
executable. Note that examples of each file can be found in the src/ folder in this repository.


### Configuration Parameters File
The input configuration parameter file is expected to contain the following values separated by
whitespace in the following order:
   the number of connectivity layers in the network,
   the number of input activation nodes,
   the number of nodes in the hidden layer,
   the number of nodes in the output layer,
   the number of test cases in the truth table,
   the name of the truth table file,
   and a boolean that is 1 if the network is in training mode or 0 if the network is in running
   mode.

If the network is in training mode, the following values should be included after the above in the
following order:
   the name of the file to which the weights should be written while training,
   the value of the learning rate,
   the value of the error threshold to use where the network will cease training if the largest test
   case error in a given iteration is less than that error threshold,
   the maximum number of iterations for training,
   and a boolean that is 1 if the network should use random values for the initial weight values or
   0 if the network should pull initial weight values from a file.

If the network is either not in training mode or should pull initial weight values from a file,
   the name of the weights file
should be included after the above values.

If the network is in running mode,
   the name of the input activation layer file
should be included after the above values.

Finally, if the network is in training mode and should use random values for the initial weight
values, the following numbers should be included after the above in the following order:
   the minimum bound of the random range to use for weights and
   the maximum bound of the random range to use for weights.


### Truth Table File
The input truth table file is expected to contain numbers separated by whitespace. For each test
case, the file should contain the values of the input activation nodes where the first value maps
to the first input activation node, the second value maps to the second input activation node, and
so on and so forth. The file should contain the values of the output layer nodes following these
input activation node values that represent the expected output for the input activation values
immediately before the output layer values. Note that the first output layer value maps to the
first output layer node, the second output layer value maps to the second output layer node, etc.
This pattern should then be repeated in the truth table file for each truth table test case.


### Weights File
The input weights file is expected to contain numbers separated by whitespace. To verify that the
correct weights file is being used, the first numbers should be the following in the following
order:
   the number of connectivity layers in the network,
   the number of input activation nodes,
   the number of nodes in the hidden layer,
   and the number of nodes in the output layer.

The following values should represent the weight values of the connections in the connectivity
layers of the network. Each connectivity layer weight values should be in the order of the
connections from the first activation node in the left layer to the nodes in the right layer, in
order from the first to last nodes in the right layer, followed by the connections between the
second activation node in the left layer to the nodes in the right layer, in the aforementioned
order, and so on and so forth.


### Input Activation Layer File
The input activation layer file is expected to contain the values of the input activation layer to
use for running the network in order (where the first value corresponds to the first input
activation node, the second value corresponds to the second input activation node, and so on and so
forth) and separated by whitespace.
