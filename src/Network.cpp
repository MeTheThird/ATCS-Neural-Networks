/**
* Rohan Thakur
* Date Created: 2/7/22
*
* This file allows for the training and running of an A-B-1 2-connectivity-layer neural network
* based upon input configuration parameters
*/

#include <bits/stdc++.h>

using namespace std;

#define f first
#define s second

typedef pair<vector<double>, vector<double> > pvd;

bool training;                                 // training represents whether the network is in
                                               // training mode
int A, B, F;                                   // A is the number of activation nodes in the input
                                               // activation layer, B is the number of nodes in the
                                               // hidden layer, and F is the number of output nodes
int numLayers;                                 // the number of connectivity layers in the network
int numTruthTableCases;                        // the number of test cases in the truth table
vector<vector<vector<double> > > weights;      // weights stores the weights, where the first index
                                               // represents the connectivity layer of an edge, and
                                               // the second and third indices represent the node
                                               // from which the edge originates and the node to
                                               // which the edge goes, respectively
vector<vector<vector<double> > > deltaWeights; // deltaWeights stores the change in weights between
                                               // each training iteration, and the indices represent
                                               // the same characteristics as those for the weights
                                               // array
vector<vector<double> > nodes;                 // nodes stores the values of the nodes in the
                                               // network, where the first index represents the node
                                               // layer of a node, and the second index represents
                                               // the index of that node in that node layer
vector<pvd> truth;                             // truth is the truth table, where the index
                                               // represents the test caseNum index, the first entry
                                               // in the pair of vectors of doubles contains the
                                               // values for the input activation nodes, and the
                                               // second entry in the pair contains the
                                               // corresponding expected values for the output nodes
double lambda;                                 // the learning factor value
double errorThreshold;                         // the error threshold for terminating training
int maxIterations;                             // the maximum number of iterations after which
                                               // training will terminate
bool useRandWeights;                           // useRandWeights represents whether the network
                                               // should be using random weights or set weights for
                                               // training
double minRandVal, maxRandVal;                 // the minimum and maximum random values to use for
                                               // generating random weight values
ifstream inputFile;                            // the current input file stream to read from

/**
* Prompts the user with the input message string for the name of the file from which to read, closes
* the old input file stream if open, and sets the global variable inputFile to a new input file
* stream associated with the file the user inputted
*/
void setupFileInputWithMessage(string message)
{
   string filename;
   cout << message << "\n";
   cin >> filename;
   cout << "\n"; // add a new line after the user's input to make the program output easier to read

   if (inputFile.is_open()) inputFile.close();
   inputFile.open(filename);
} // void setupFileInput(string filename)

/**
* Takes in the input configuration parameters and sets their corresponding global variable values
* accordingly
*/
void config()
{
   setupFileInputWithMessage("What is the full name of your input parameter file?");
   inputFile >> numLayers >> A >> B >> F >> numTruthTableCases;
   inputFile >> training;
   if (training) inputFile >> lambda >> errorThreshold >> maxIterations >> useRandWeights;
   if (useRandWeights) inputFile >> minRandVal >> maxRandVal;
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
   for (int caseNum = 0; caseNum < numTruthTableCases; caseNum++)
   {
      truth[caseNum].f.resize(A);
      truth[caseNum].s.resize(F);
   } // for (int caseNum = 0; caseNum < numTruthTableCases; caseNum++)
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
   nodes.resize(numLayers + 1); // the number of node layers is always one greater than the number
                                // of connectivity layers
   for (int nodeLayer = 0; nodeLayer < numLayers + 1; nodeLayer++)
   {
      if (nodeLayer == numLayers) nodes[nodeLayer].resize(F);
      else if (!nodeLayer) nodes[nodeLayer].resize(A);
      else nodes[nodeLayer].resize(B);
   } // for (int nodeLayer = 0; nodeLayer < numLayers + 1; nodeLayer++)
} // void allocateMemory()

/**
* Loads in the truth table values from a file the user has created
*/
void loadTruthTableValues()
{
   setupFileInputWithMessage("What is the full name of the file containing the truth table?");
   for (int caseNum = 0; caseNum < numTruthTableCases; caseNum++)
   {
      for (int k = 0; k < A; k++) inputFile >> truth[caseNum].f[k];
      for (int i = 0; i < F; i++) inputFile >> truth[caseNum].s[i];
   } // for (int caseNum = 0; caseNum < numTruthTableCases; caseNum++)
} // void loadTruthTableValues()

/**
* Loads in the weight values from a file the user has created
*/
void loadWeightValues()
{
   setupFileInputWithMessage("What is the full name of the file containing the weights?");
   for (int n = 0; n < numLayers; n++)
      for (int i = 0; i < weights[n].size(); i++)
         for (int j = 0; j < weights[n][i].size(); j++)
            inputFile >> weights[n][i][j];
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
      for (int i = 0; i < weights[n].size(); i++)
         for (int j = 0; j < weights[n][i].size(); j++)
            weights[n][i][j] = getRandomNumberBetween(minRandVal, maxRandVal);
} // void generateRandomWeightValues()

/**
* Loads in the activation values to use for running the network from a file the user has created
*/
void loadActivationValues()
{
   setupFileInputWithMessage("What is the full name of the file containing the input activation "
                             "values?");
   for (int k = 0; k < A; k++) inputFile >> nodes[0][k];
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
* Returns the activation function evaluated at the input value. The current activation function is
* the sigmoid function 1 / (1 + e^(-x))
*/
double activationFunction(double value)
{
   return 1 / (1 + exp(-value));
} // double activationFunction(double value)

/**
* Returns the dervivative of the activation function defined above
*/
double activationFunctionDerivative(double value)
{
   return activationFunction(value) * (1 - activationFunction(value));
}

/**
* Calculates and returns the value of the node in the given node layer at the given index, where
* both the node layer and index are zero-indexed
*
* Precondition: nodeLayer > 0
*/
double calculateNode(int nodeLayer, int index)
{
   double ret;
   for (int prev = 0; prev < weights[nodeLayer-1].size(); prev++)
      ret += nodes[nodeLayer-1][prev] * weights[nodeLayer-1][prev][index];
   return activationFunction(ret);
} // void calculateNode(int nodeLayer, int index)

/**
* Runs the network
*/
void run()
{
   // nodeLayer begins at 1 because the input activation layer was given by the user, and nodeLayer
   // goes to numLayers + 1 because there exists one more node layer than the number of connectivity
   // layers
   for (int nodeLayer = 1; nodeLayer < numLayers + 1; nodeLayer++)
   {
      if (nodeLayer == numLayers)
         for (int i = 0; i < F; i++) nodes[nodeLayer][i] = calculateNode(nodeLayer, i);
      else
         for (int j = 0; j < B; j++) nodes[nodeLayer][j] = calculateNode(nodeLayer, j);
   } // for (int nodeLayer = 1; nodeLayer < numLayers + 1; nodeLayer++)
} // void run()

/**
* Trains the network, stopping when either the maximum number of iterations has been reached or
* the maximum error across all test cases is lower than the error threshold
*/
void train()
{
   int numIterations = 0;
   bool maxIterationsReached = false;
   bool errorThresholdReached = false;

   while (!maxIterationsReached && !errorThresholdReached)
   {
      double maxError = -DBL_MAX; // initialized to the lowest possible double value
      for (int testCaseNum = 0; testCaseNum < numTruthTableCases; testCaseNum++)
      {
         for (int k = 0; k < A; k++) nodes[0][k] = truth[testCaseNum].f[k];
         run();

      } // for (int testCaseNum = 0; testCaseNum < numTruthTableCases; testCaseNum++)
      numIterations++;
      if (numIterations >= maxIterations) maxIterationsReached = true;
      if (maxError < errorThreshold) errorThresholdReached = true;
   } // while (!maxIterationsReached && !errorThresholdReached)

   cout << "Training was terminiated because of the following reason(s):\n";
   if (maxIterationsReached) cout << "The maximum number of iterations was reached\n";
   if (errorThresholdReached) cout << "The error threshold was reached\n";
} // void train()

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
   if (!training) run();
   else train();
   // TODO: printOutOutput()



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

   // TODO: can we #include <bits/stdc++.h>? or should we only include specific modules?

   // TODO: can we use typedef and/or #define (check top of file for specifics)?

   // TODO: can we have nested for loops without braces that have one line of code -- see
   // generateRandomWeightValues() and loadWeightValues() and calculateNode(params)

} // int main()
























