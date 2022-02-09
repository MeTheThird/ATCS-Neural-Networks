/**
* Rohan Thakur
* Date Created: 2/7/22
*
* This file allows for the training and running of an A-B-1 2-connectivity-layer neural network
* based upon input configuration parameters
*/

#include <bits/stdc++.h>

using namespace std;

bool training;                            // training represents whether the network is in training
                                          // mode
int A, B, F;                              // A is the number of activation nodes in the input
                                          // activation layer, B is the number of nodes in the
                                          // hidden layer, and F is the number of output nodes
int numLayers;                            // the number of connectivity layers in the network
vector<vector<vector<double> > > weights; // weights is the weights array, where the first index
                                          // represents the connectivity layer of an edge, and the
                                          // second and third indices represent the node from which
                                          // the edge originates and the node to which the edge
                                          // goes, respectively. The value contained in a given
                                          // position in weights is the weight of the corresponding
                                          // edge
vector<vector<double> > T;                // T is the truth table, where the first index represents
                                          // the test case index, and the second index represents
                                          // the index of the node. Note that the values contained
                                          // in all but the last entry of any given test case are
                                          // the input activation layer values, and the last entry
                                          // in any given test case is the expected output value
double lambda;                            // the learning factor value
double errorThreshold;                    // the error threshold for terminating training
int maxIterations;                        // the maximum number of iterations after which training
                                          // will terminate
bool useRandWeights;                      // useRandWeights represents whether the network
                                          // should be using random weights or set weights for
                                          // training
double minRandVal, maxRandVal;            // the minimum and maximum random values to use for random
                                          // weight values

/**
* Prompts the user with the input message string for the name of the file from which to read and
* then sets that file as the standard input stream
*/
void setupFileInputWithMessage(string message)
{
   string filename;
   cout << message << "\n";
   cin >> filename;
   cout << "\n"; // add a new line after the user's input to make the program output easier to read
   freopen(filename.c_str(), "r", stdin);
} // void setupFileInput(string filename)

/**
* Takes in the input configuration parameters and sets their corresponding global variable values
* accordingly
*/
void config()
{
   setupFileInputWithMessage("What is the name of your input parameter file?");
   cin >> numLayers >> A >> B >> F;
   cin >> training;
   if (training) cin >> lambda >> errorThreshold >> maxIterations >> useRandWeights;
   if (useRandWeights) cin >> minRandVal >> maxRandVal;
} // void config()

/**
* Prints out the configuration parameters used
*/
void printOutConfigVals()
{
   cout << "numLayers: " << numLayers << ", A: " << A << ", B: " << B << ", F: " << F << "\n";
   cout << "training: " << training << "\n";

   if (training)
   {
      cout << "lambda: " << lambda << ", errorThreshold: " << errorThreshold << ", maxIterations: ";
      cout << maxIterations << ", useRandWeights: " << useRandWeights << "\n";
   } // if (training)

   if (useRandWeights)
   {
      cout << "minRandVal: " << minRandVal << ", maxRandVal: " << maxRandVal << "\n";
   } // if (useRandWeights)
} // void printOutConfigVals()

/**
* The main method which either trains or executes the network, depending upon the configuration
* parameters
*/
int main()
{
   config();
   // TODO: make everything their own functions
   // TODO: write down the expected structure of the input parameter file in the README file
   // i.e. numLayers, A, B, F, training, lambda, errorThreshold, maxIterations, useRandWeights,
   // minRandVal, maxRandVal

   // TODO: make the iterator variables k, j, and i as set in the design doc based on which layer
   // TODO: I'm iterating through
   printOutConfigVals();

   // TODO: make the weights file a separate file from the config file

} // int main()
























