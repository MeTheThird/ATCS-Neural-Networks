/**
* Rohan Thakur
* Date Created: 2/7/22
*
* This file allows for the training and running of an A-B-1 2-connectivity-layer neural network
* based upon input configuration parameters.
*
* Throughout this file, the index 0 is used to refer to the input activation layer, 1 refers to the
* hidden layer, and 2 refers to the output layer. Note that those indices apply when the values of
* nodes are being fetched or memory is being allocated. For weight values, the index 0 refers to the
* weights between the input activation layer and the hidden layer, and the index 1 refers to the
* weights between the hidden layer and the output layer. Moreover, since this network is defined to
* only have one node in the output layer, the index 0 is used when appropriate to fetch its
* associated values. In short, the 0s, 1s, and 2s present throughout this code are used because of
* the design doc and are not magic numbers.
*/

#include <bits/stdc++.h>
#include <random>

using namespace std;

#define f first // these #define statements are for shorthand convenience
#define s second

/**
* training represents whether the network is in training mode
*/
bool training;
/**
* A is the number of activation nodes in the input activation layer, B is the number of nodes in the
* hidden layer, and F is the number of output nodes
*/
int A, B, F;
/**
* the number of connectivity layers in the network
*/
int numLayers;
/**
* the number of test cases in the truth table
*/
int numTruthTableCases;
/**
* weights stores the weights, where the first index represents the connectivity layer of an edge,
* and the second and third indices represent the node from which the edge originates and the node to
* which the edge goes, respectively
*/
vector<vector<vector<double> > > weights;
/**
* deltaWeights stores the change in weights between each training iteration, and the indices
* represent the same characteristics as those for the weights array
*/
vector<vector<vector<double> > > deltaWeights;
/**
* nodes stores the values of the nodes in the network, where the first index represents the node
* layer of a node, and the second index represents the index of that node in that node layer
*/
vector<vector<double> > nodes;
/**
* sums stores the values of the nodes in the network before the activation function is applied i.e.
* for each node, sums stores the sum over the values of the nodes in the previous node layer
* multiplied by their associated weight value. Note that the first node layer of sums is just the
* input activation layer, and the later node layers of sums follows the aforementioned definition.
* Also note that the indices represent the same characteristics as those in the nodes array above.
*/
vector<vector<double> > sums;
/**
* truth is the truth table, where the index represents the test caseNum index, the first entry in
* the pair of vectors of doubles contains the values for the input activation nodes, and the second
* entry in the pair contains the corresponding expected values for the output nodes
*/
vector<pair<vector<double>, vector<double> > > truth;
/**
* the learning factor value
*/
double lambda;
/**
* the error threshold for terminating training
*/
double errorThreshold;
/**
* the maximum number of iterations after which training will terminate
*/
int maxIterations;
/**
* useRandWeights represents whether the network should be using random weights or set weights for
* training
*/
bool useRandWeights;
/**
* the minimum and maximum random values to use for generating random weight values
*/
double minRandVal, maxRandVal;
/**
* the current input file stream to read from
*/
ifstream inputFile;
/**
* integer representing the number of iterations completed while training
*/
int numIterations = 0;
/**
* boolean representing whether the maximum number of iterations was reached while training
*/
bool maxIterationsReached = false;
/**
* boolean representing whether the error threshold was reached while training i.e. whether the
* maximum test case error in a given training iteration was less than the input error threshold
*/
bool errorThresholdReached = false;
/**
* the error reached while training
*/
double errorReached;

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
   cout << "Configuration parameters:\n";
   cout << "\tnumLayers: " << numLayers << ", A: " << A << ", B: " << B << ", F: " << F;
   cout << ", numTruthTableCases: " << numTruthTableCases << "\n";
   cout << "\ttraining: " << training << "\n";

   if (training)
   {
      cout << "\tlambda: " << lambda << ", errorThreshold: " << errorThreshold << ", maxIterations: ";
      cout << maxIterations << ", useRandWeights: " << useRandWeights << "\n";
   } // if (training)

   if (useRandWeights)
      cout << "\tminRandVal: " << minRandVal << ", maxRandVal: " << maxRandVal << "\n";

   cout << "\n";
} // void printOutConfigVals()

/**
* Allocates the necessary memory for the weights array and the deltaWeights array, only allocating
* memory for deltaWeights if the network is in training mode
*
* Precondition: the configuration parameters have been set
*/
void allocateWeightsArrays()
{
   weights.resize(numLayers);
   if (training) deltaWeights.resize(numLayers);

   weights[0].resize(A); // allocates memory for the first connectivity layer
   if (training) deltaWeights[0].resize(A);
   for (int k = 0; k < A; k++)
   {
      weights[0][k].resize(B);
      if (training) deltaWeights[0][k].resize(B);
   } // for (int k = 0; k < A; k++)

   weights[1].resize(B); // allocates memory for the second connectivity layer
   if (training) deltaWeights[1].resize(B);
   for (int j = 0; j < B; j++)
   {
      weights[1][j].resize(F);
      if (training) deltaWeights[1][j].resize(F);
   } // for (int j = 0; j < B; j++)
} // void allocateWeightsArray()

/**
* Allocates the necessary memory for the truth table array
*
* Precondition: the configuration parameters have been set
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
* Precondition: the configuration parameters have been set
*/
void allocateMemory()
{
   allocateWeightsArrays();
   allocateTruthTableArray();

   // the number of node layers is always one greater than the number of connectivity layers
   nodes.resize(numLayers + 1);
   sums.resize(numLayers + 1);

   nodes[0].resize(A); // the first node layer is the activation layer with A nodes
   sums[0].resize(A);
   nodes[1].resize(B); // the second node layer is the hidden layer with B nodes
   sums[1].resize(B);
   nodes[2].resize(F); // the third and final node layer is the output layer with F nodes
   sums[2].resize(F);
} // void allocateMemory()

/**
* Loads in the truth table values from a file the user has created
*
* Precondition: the memory for the truth table values has been carved out
*/
void loadTruthTableValues()
{
   setupFileInputWithMessage("What is the full name of the file containing the truth table?");
   for (int caseNum = 0; caseNum < numTruthTableCases; caseNum++)
   {
      for (int k = 0; k < A; k++) inputFile >> truth[caseNum].f[k];
      inputFile >> truth[caseNum].s[0];
   } // for (int caseNum = 0; caseNum < numTruthTableCases; caseNum++)
} // void loadTruthTableValues()

/**
* Loads in the weight values from a file the user has created
*
* Precondition: the memory for the network's weight values has been carved out
*/
void loadWeightValues()
{
   setupFileInputWithMessage("What is the full name of the file containing the weights?");

   for (int k = 0; k < A; k++) // load the first connectivity layer of the weights array
      for (int j = 0; j < B; j++)
         inputFile >> weights[0][k][j];

   for (int j = 0; j < B; j++) // load the second connectivity layer of the weights array
      inputFile >> weights[1][j][0];
} // void loadWeightValues()

/**
* Returns a random number in between minValue and maxValue
*
* Precondition: maxValue >= minValue
*/
double getRandomNumberBetween(double minValue, double maxValue)
{
   random_device rd;
   mt19937 generator(rd());
   uniform_real_distribution<> distr(minValue, maxValue);
   return distr(generator);
} // double getRandomNumberBetween(double minValue, double maxValue)

/**
* Generates random weight values based upon the configuration parameters
*
* Precondition: the memory for the network's weight values has been carved out
*/
void generateRandomWeightValues()
{
   // randomly generate the first connectivity layer of the weights array
   for (int k = 0; k < A; k++)
      for (int j = 0; j < B; j++)
         weights[0][k][j] = getRandomNumberBetween(minRandVal, maxRandVal);

   // randomly generate the second connectivity layer of the weights array
   for (int j = 0; j < B; j++)
      weights[1][j][0] = getRandomNumberBetween(minRandVal, maxRandVal);
} // void generateRandomWeightValues()

/**
* Loads in the activation values to use for running the network from a file the user has created
*
* Precondition: the memory for the input activation layer has been carved out
*/
void loadActivationValues()
{
   setupFileInputWithMessage("What is the full name of the file containing the input activation "
                             "values?");
   for (int k = 0; k < A; k++) inputFile >> nodes[0][k];
} // void loadActivationValues()

/**
* Loads the appropriate values into the corresponding global variable arrays depending upon the
* network's configuration parameters
*
* Precondition: the memory for each necessary array has been carved out
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
   return 1.0 / (1.0 + exp(-value));
} // double activationFunction(double value)

/**
* Returns the derivative, evaluated at the given value, of the activation function defined above
*
* Precondition: the activation function is differentiable
*/
double activationFunctionDerivative(double value)
{
   return activationFunction(value) * (1.0 - activationFunction(value));
} // double activationFunctionDerivative(double value)

/**
* Runs the network
*
* Precondition: the input activation layer values and the network's weight values have been set
*/
void run()
{
   double nodeVal;
   for (int j = 0; j < B; j++) // calculates and populates the hidden layer
   {
      nodeVal = 0.0;
      for (int k = 0; k < A; k++) nodeVal += nodes[0][k] * weights[0][k][j];
      nodes[1][j] = activationFunction(nodeVal);
      sums[1][j] = nodeVal;
   } // for (int j = 0; j < B; j++)

   nodeVal = 0.0;              // calculates and populates the output layer
   for (int j = 0; j < B; j++) nodeVal += nodes[1][j] * weights[1][j][0];
   nodes[numLayers][0] = activationFunction(nodeVal);
   sums[numLayers][0] = nodeVal;
} // void run()

/**
* Trains the network, stopping when either the maximum number of iterations has been reached or
* the maximum error across all test cases is lower than the error threshold
*
* Precondition: the truth table values and the network's weight values have been set
*/
void train()
{
   while (!maxIterationsReached && !errorThresholdReached)
   {
      errorReached = -DBL_MAX; // initialized to the lowest possible double value
      for (int testCaseNum = 0; testCaseNum < numTruthTableCases; testCaseNum++)
      {
         for (int k = 0; k < A; k++)
         {
            nodes[0][k] = truth[testCaseNum].f[k];
            sums[0][k] = truth[testCaseNum].f[k];
         } // for (int k = 0; k < A; k++)
         run();

         double omega_0 = truth[testCaseNum].s[0] - nodes[numLayers][0];
         double totalError = 0.0;

         totalError += omega_0 * omega_0;
         totalError *= 1.0 / 2.0;
         if (totalError > errorReached) errorReached = totalError;

         // calculate and populate the delta weights array for the second connectivity layer
         double psi_0 = 0.0;
         psi_0 = omega_0 * activationFunctionDerivative(sums[numLayers][0]);
         for (int j = 0; j < B; j++)
         {
            double partialDeriv = -nodes[1][j] * psi_0;
            deltaWeights[1][j][0] = -lambda * partialDeriv;
         } // for (int j = 0; j < B; j++)

         // calculate and populate the delta weights array for the first connectivity layer
         for (int j = 0; j < B; j++)
         {
            double capitalOmega_j = psi_0 * weights[1][j][0];
            double capitalPsi_j = capitalOmega_j * activationFunctionDerivative(sums[1][j]);
            for (int k = 0; k < A; k++)
            {
               double partialDeriv = -nodes[0][k] * capitalPsi_j;
               deltaWeights[0][k][j] = -lambda * partialDeriv;
            } // for (int k = 0; k < A; k++)
         } // for (int j = 0; j < B; j++)

         for (int k = 0; k < A; k++) // update the weights array for the first connectivity layer
            for (int j = 0; j < B; j++)
               weights[0][k][j] += deltaWeights[0][k][j];

         for (int j = 0; j < B; j++) // update the weights array for the second connectivity layer
            weights[1][j][0] += deltaWeights[1][j][0];
      } // for (int testCaseNum = 0; testCaseNum < numTruthTableCases; testCaseNum++)

      numIterations++;
      if (numIterations >= maxIterations) maxIterationsReached = true;
      if (errorReached < errorThreshold) errorThresholdReached = true;
   } // while (!maxIterationsReached && !errorThresholdReached)
} // void train()

/**
* Reports on the network's behavior after training or running, where the boolean parameter
* doInitialReport represents whether this method should print out the non-truth table portion of the
* report or print out the truth table portion of the report, and the integer parameter testCaseNum
* represents the index of the test case on which to report if doInitialReport is false
*
* Precondition: the network has either trained or run
*/
void report(bool doInitialReport, int testCaseNum)
{
   if (doInitialReport)
   {
      printOutConfigVals();
      if (training)
      {
         cout << "Training was terminated because of the following reason(s):\n";
         if (maxIterationsReached) cout << "\tThe maximum number of iterations was reached\n";
         if (errorThresholdReached) cout << "\tThe error threshold was reached\n";
         cout << "\n";

         cout << "Relevant values at the end of training:\n";
         cout << "\tThe number of iterations reached was " << numIterations << "\n";
         cout << "\tThe error reached was " << errorReached << "\n";
         cout << "\n";
      } // if (training)
   } // if (doInitialReport)
   else
   {
      cout << "Test Case " << testCaseNum + 1 << ":\n";
      cout << "\tInput activation value(s):";
      for (int k = 0; k < A; k++) cout << " " << truth[testCaseNum].f[k];
      cout << "\n\tExpected output value(s):";
      cout << " " << truth[testCaseNum].s[0];
      cout << "\n\tNetwork generated output value(s):";
      cout << " " << nodes[numLayers][0];
      cout << "\n";
   } // else
} // void report(bool doInitialReport, int testCaseNum)

/**
* The main method which either trains or executes the network, depending upon the configuration
* parameters, and then outputs a report
*/
int main()
{
   config();
   printOutConfigVals();
   allocateMemory();
   loadValues();
   if (!training) run();
   else train();
   report(true, 0);
   for (int testCaseNum = 0; testCaseNum < numTruthTableCases; testCaseNum++)
   {
      for (int k = 0; k < A; k++) nodes[0][k] = truth[testCaseNum].f[k];
      run();
      report(false, testCaseNum);
   } // for (int testCaseNum = 0; testCaseNum < numTruthTableCases; testCaseNum++)
} // int main()
