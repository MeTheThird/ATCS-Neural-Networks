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
int numTruthTableCases;                   // the number of test cases in the truth table
vector<vector<vector<double> > > weights; // weights is the weights array, where the first index
                                          // represents the connectivity layer of an edge, and the
                                          // second and third indices represent the node from which
                                          // the edge originates and the node to which the edge
                                          // goes, respectively. The value contained in a given
                                          // position in weights is the weight of the corresponding
                                          // edge
vector<vector<double> > truth;            // truth is the truth table, where the first index
                                          // represents the test case index, and the second index
                                          // represents the index of the node. Note that the values
                                          // contained in all but the last entry of any given test
                                          // case are the input activation layer values, and the
                                          // last entry in any given test case is the expected
                                          // output value
vector<double> activations;               // the input activation layer values to use to run the
                                          // network if the network is in running mode
double lambda;                            // the learning factor value
double errorThreshold;                    // the error threshold for terminating training
int maxIterations;                        // the maximum number of iterations after which training
                                          // will terminate
bool useRandWeights;                      // useRandWeights represents whether the network
                                          // should be using random weights or set weights for
                                          // training
double minRandVal, maxRandVal;            // the minimum and maximum random values to use for random
                                          // weight values
string previousFilename;                  // the name of the most recently opened user input file

/**
* Prompts the user with the input message string for the name of the file from which to read and
* then sets that file as the standard input stream
*/
void setupFileInputWithMessage(string message)
{
   string filename;
   cout << message << "\n";
// TODO: fix file IO i.e. copy stdin before closing it to then re-open it to get the filename
// from stdin
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
   cin >> numLayers >> A >> B >> F >> numTruthTableCases;
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
* Allocates the necessary memory for the weights array
*/
void allocateWeightsArray()
{
   weights.resize(numLayers);
   for (int n = 0; n < numLayers; n++)
   {
      if (n)
      {
         weights[n].resize(B); // OR weights[n].resize(weights[n-1][0].size());
         for (int j = 0; j < B; j++)
         {
            weights[n][j].resize(F);
         } // for (int j = 0; j < B; j++)
      } // if (n)
      else
      {
         weights[n].resize(A);
         for (int k = 0; k < A; k++)
         {
            weights[n][k].resize(B);
         } // for (int k = 0; k < A; k++)
      } // else
   } // for (int n = 0; n < numLayers; n++)
} // void allocateWeightsArray()

/**
* Allocates the necessary memory for the truth table array
*/
void allocateTruthTableArray()
{
   truth.resize(numTruthTableCases);
   for (int i = 0; i < numTruthTableCases; i++)
   {
      truth.resize(A + 1); // needs A + 1 spaces because each test case needs A space for the input
                           // activation values plus an additional space for the expected output
                           // value
   } // for (int i = 0; i < numTruthTableCases; i++)
} // void allocateTruthTableArray()

/**
* Allocates the necessary memory for the global variables based upon the configuration parameters
*
* Precondition: the configuration parameters must have been already set
*/
void allocateMemory()
{
   allocateWeightsArray();
   allocateTruthTableArray();
   // allocates memory for the activations array only if the network is running
   if (!training) activations.resize(A);
} // void allocateMemory()

/**
* Loads in the truth table values from a file the user has created
*/
void loadTruthTableValues()
{
   setupFileInputWithMessage("What is the name of the file contining the truth table?");
   for (int i = 0; i < numTruthTableCases; i++)
   {
      for (int k = 0; k < A; k++)
      {
         cin >> truth[i][k];
      } // for (int k = 0; k < A; k++)
      cin >> truth[i][A]; // take in the expected output value for the current truth table test case
   } // for (int i = 0; i < numTruthTableCases; i++)
} // void loadTruthTableValues()

/**
* Loads in the weight values from a file the user has created
*/
void loadWeightValues()
{
   setupFileInputWithMessage("What is the name of the file containing the weights?");
   for (int n = 0; n < numLayers; n++)
   {
      for (int i = 0; i < weights[n].size(); i++)
      {
         for (int j = 0; j < weights[n][i].size(); j++)
         {
            cin >> weights[n][i][j];
         } // for (int j = 0; j < weights[n][i].size(); j++)
      } // for (int i = 0; i < weights[n].size(); i++)
   } // for (int n = 0; n < numLayers; n++)
} // void loadWeightValues()

/**
* Returns a random number in between minValue and maxValue
*
* Precondition: maxValue >= minValue
*/
double getRandomNumberBetween(double minValue, double maxValue)
{
   double randBetweenZeroAndOne = ((double) rand()) / ((double) RAND_MAX);
   return randBetweenZeroAndOne * (maxValue - minValue) + minValue;
}

/**
* Generates random weight values based upon the configuration parameters
*/
void generateRandomWeightValues()
{
   for (int n = 0; n < numLayers; n++)
   {
      for (int i = 0; i < weights[n].size(); i++)
      {
         for (int j = 0; j < weights[n][i].size(); j++)
         {
            weights[n][i][j] = getRandomNumberBetween(minRandVal, maxRandVal);
         } // for (int j = 0; j < weights[n][i].size(); j++)
      } // for (int i = 0; i < weights[n].size(); i++)
   } // for (int n = 0; n < numLayers; n++)
} // void generateRandomWeightValues()

/**
* Loads in the activation values to use for running the network from a file the user has created
*/
void loadActivationValues()
{
   setupFileInputWithMessage("What is the name of the file containing the input activation "
                             "values?");
   for (int k = 0; k < A; k++) cin >> activations[k];
} // void loadActivationValues()

/**
* Loads the appropriate values into the correponding global variable arrays depending upon the
* network's configuration parameters
*/
void loadValues()
{
   loadTruthTableValues();
   if (!training || !useRandWeights) loadWeightValues();
   else generateRandomWeightValues();
   if (!training) loadActivationValues();
} // void loadValues()

/**
* Trains the network
*/
void train()
{

}

/**
* Runs the network
*/
void run()
{

}

/**
* The main method which either trains or executes the network, depending upon the configuration
* parameters
*/
int main()
{
   config();
   printOutConfigVals();
   allocateMemory();
   loadValues();
   if (training) train();
   else run();
   // TODO: make everything their own functions
   // TODO: write down the expected structure of the input parameter file in the README file
   // i.e. numLayers, A, B, F, numTruthTableCases, training, lambda, errorThreshold, maxIterations,
   // useRandWeights,
   // minRandVal, maxRandVal

   // TODO: write down the expected structure of each and every input file

   // TODO: write down the fact that each file is expected to be in the same directory as the
   // executable file

   // TODO: make the iterator variables k, j, and i as set in the design doc based on which layer
   // TODO: I'm iterating through i.e. k for activation layer, j for hidden layer, i for output

   // TODO: make the weights file a separate file from the config file

} // int main()
























