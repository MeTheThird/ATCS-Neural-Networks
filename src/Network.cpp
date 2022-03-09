/**
 * Rohan Thakur
 * Date Created: 2/7/22
 *
 * This file allows for the training (using backpropagation) and running of an A-B-C
 * 2-connectivity-layer neural network based upon input configuration parameters.
 *
 * Throughout this file, the index 0 is used to refer to the input activation layer, 1 refers to the
 * hidden layer, and 2 refers to the output layer. Note that those indices apply when the values of
 * nodes are being fetched or memory is being allocated. For weight values, the index 0 refers to
 * the weights between the input activation layer and the hidden layer, and the index 1 refers to
 * the weights between the hidden layer and the output layer. In short, the 0s, 1s, and 2s present
 * throughout this code are used because of the design doc and are not magic numbers.
 */

#include <bits/stdc++.h>
#include <random>

using namespace std;

#define f first // these #define statements are for shorthand convenience
#define s second

bool training;  // Represents whether the network is in training mode

/**
 * The number of activation nodes in the input activation layer, B is the number of nodes in the
 * hidden layer, and F is the number of output nodes
 */
int A, B, F;

int numLayers;          // The number of connectivity layers in the network

int numTruthTableCases; // The number of test cases in the truth table

// the default filename to use for the configuration parameters file
string DEFAULT_CONFIG_FILENAME = "test.conf";

/**
 * These variables represent the default filenames to use for taking in the truth table, weight
 * values, and input activation values, respectively
 */
string defaultTruthTableFilename, defaultWeightsFilename, defaultInputActivationsFilename;

/**
 * Stores the weights, where the first index represents the connectivity layer of an edge, and the
 * second and third indices represent the node from which the edge originates and the node to which
 * the edge goes, respectively
 */
vector<vector<vector<double> > > weights;

/**
 * Stores the values of the nodes in the network after the activation function is applied, where the
 * first index represents the node layer of a node, and the second index represents the index of
 * that node in that node layer
 */
vector<vector<double> > nodes;

/**
 * Stores the values of the nodes in the network before the activation function is applied i.e. for
 * each node, sums stores the sum over the values of the nodes in the previous node layer multiplied
 * by their associated weight value. Note that the first node layer of sums is just the input
 * activation layer, and the later node layers of sums follows the aforementioned definition. Also
 * note that the indices represent the same characteristics as those in the nodes array above.
 */
vector<vector<double> > sums;

/**
 * The truth table, where the index represents the test caseNum index, the first entry in the pair
 * of vectors of doubles contains the values for the input activation nodes, and the second entry in
 * the pair contains the corresponding expected values for the output nodes
 */
vector<pair<vector<double>, vector<double> > > truth;

double lambda;         // The learning factor value

double errorThreshold; // The error threshold for terminating training

int maxIterations;     // The maximum number of iterations after which training will terminate

/**
 * useRandWeights represents whether the network should be using random weights or set weights for
 * training
 */
bool useRandWeights;

// The minimum and maximum random values to use for generating random weight values
double minRandVal, maxRandVal;

ifstream inputFile;                    // The current input file stream to read from

vector<double> psi;                    // Stores the psi_i values calculated during training

// Integer representing the number of iterations completed while training
int numIterations = 0;

double errorReached;                   // The error reached while training

chrono::duration<double> trainingTime; // the amount of time training took

// Boolean representing whether the maximum number of iterations was reached while training
bool maxIterationsReached = false;

/**
 * Boolean representing whether the error threshold was reached while training i.e. whether the
 * maximum test case error in a given training iteration was less than the input error threshold
 */
bool errorThresholdReached = false;

/**
 * Closes the old input file stream if open and sets the global variable inputFile to a new input
 * file stream associated with the file named defaultFilename
 */
void setupFileInput(string defaultFilename)
{
   if (inputFile.is_open()) inputFile.close();

   inputFile.open(defaultFilename);
} // void setupFileInput(string defaultFilename)

/**
 * Processes the input configuration parameters and sets their corresponding global variable values
 * accordingly, using the standard C++ command line input arguments as parameters for this method to
 * determine whether the user has overriden the default name of the configuration parameters file
 */
void config(int argc, char* argv[])
{
   // sets up file input for the configuration parameters file
   if (argc > 1) setupFileInput(argv[1]);
   else setupFileInput(DEFAULT_CONFIG_FILENAME);

   inputFile >> numLayers >> A >> B >> F >> numTruthTableCases >> defaultTruthTableFilename;
   inputFile >> training;

   if (training) inputFile >> lambda >> errorThreshold >> maxIterations >> useRandWeights;

   if (!training || !useRandWeights) inputFile >> defaultWeightsFilename;

   if (!training) inputFile >> defaultInputActivationsFilename;

   if (useRandWeights) inputFile >> minRandVal >> maxRandVal;
} // void config(int argc, char* argv[])

/**
 * Prints out the configuration parameters used
 */
void printOutConfigVals()
{
   cout << "Names of files taken from the configuration parameters file:\n";
   cout << "\tTruth table filename: " << defaultTruthTableFilename << "\n";
   if (!training || !useRandWeights)
      cout << "\tWeight values filename: " << defaultWeightsFilename << "\n";
   if (!training)
      cout << "\tActivation values filename: " << defaultInputActivationsFilename << "\n";
   cout << "\n";

   cout << "Configuration parameters:\n";
   cout << "\tnumLayers: " << numLayers << ", A: " << A << ", B: " << B << ", F: " << F;
   cout << ", numTruthTableCases: " << numTruthTableCases << "\n";
   cout << "\ttraining: " << training << "\n";

   if (training)
   {
      cout << "\tlambda: " << lambda << ", errorThreshold: " << errorThreshold;
      cout << ", maxIterations: " << maxIterations << ", useRandWeights: " << useRandWeights;
      cout << "\n";
   } // if (training)

   if (useRandWeights)
      cout << "\tminRandVal: " << minRandVal << ", maxRandVal: " << maxRandVal << "\n";

   cout << "\n";
} // void printOutConfigVals()

/**
 * Allocates the necessary memory for the weights array
 *
 * Precondition: the configuration parameters have been set
 */
void allocateWeightsArrays()
{
   weights.resize(numLayers);

   weights[0].resize(A); // allocates memory for the first connectivity layer
   for (int k = 0; k < A; k++) weights[0][k].resize(B);

   weights[1].resize(B); // allocates memory for the second connectivity layer
   for (int j = 0; j < B; j++) weights[1][j].resize(F);
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

   if (training) psi.resize(F);

   // the number of node layers is always one greater than the number of connectivity layers
   nodes.resize(numLayers + 1);
   if (training) sums.resize(numLayers + 1);

   nodes[0].resize(A); // the first node layer is the activation layer with A nodes

   nodes[1].resize(B); // the second node layer is the hidden layer with B nodes
   if (training) sums[1].resize(B);

   nodes[2].resize(F); // the third and final node layer is the output layer with F nodes
   if (training) sums[2].resize(F);
} // void allocateMemory()

/**
 * Loads in the truth table values from a file the user has created
 *
 * Precondition: the memory for the truth table values has been carved out
 */
void loadTruthTableValues()
{
   setupFileInput(defaultTruthTableFilename);

   for (int caseNum = 0; caseNum < numTruthTableCases; caseNum++)
   {
      for (int k = 0; k < A; k++) inputFile >> truth[caseNum].f[k];
      for (int i = 0; i < F; i++) inputFile >> truth[caseNum].s[i];
   } // for (int caseNum = 0; caseNum < numTruthTableCases; caseNum++)
} // void loadTruthTableValues()

/**
 * Loads in the weight values from a file the user has created
 *
 * Precondition: the memory for the network's weight values has been carved out
 */
void loadWeightValues()
{
   setupFileInput(defaultWeightsFilename);

   for (int k = 0; k < A; k++) // load the first connectivity layer of the weights array
      for (int j = 0; j < B; j++)
         inputFile >> weights[0][k][j];

   for (int i = 0; i < F; i++) // load the second connectivity layer of the weights array
      for (int j = 0; j < B; j++)
         inputFile >> weights[1][j][i];
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
   for (int i = 0; i < F; i++)
      for (int j = 0; j < B; j++)
         weights[1][j][i] = getRandomNumberBetween(minRandVal, maxRandVal);
} // void generateRandomWeightValues()

/**
 * Loads in the activation values to use for running the network from a file the user has created
 *
 * Precondition: the memory for the input activation layer has been carved out
 */
void loadActivationValues()
{
   setupFileInput(defaultInputActivationsFilename);

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
 * Runs the network, storing the values of the nodes in the nodes array
 *
 * Precondition: the input activation layer values and the network's weight values have been set,
 * and the network is in running mode
 */
void runRunning()
{
   double nodeVal;

   for (int j = 0; j < B; j++) // calculates and populates the hidden layer
   {
      nodeVal = 0.0;
      for (int k = 0; k < A; k++) nodeVal += nodes[0][k] * weights[0][k][j];

      nodes[1][j] = activationFunction(nodeVal);
   } // for (int j = 0; j < B; j++)

   for (int i = 0; i < F; i++) // calculates and populates the output layer
   {
      nodeVal = 0.0;
      for (int j = 0; j < B; j++) nodeVal += nodes[1][j] * weights[1][j][i];

      nodes[numLayers][i] = activationFunction(nodeVal);
   } // for (int i = 0; i < F; i++)
} // void runRunning()

/**
 * Runs the network, storing the values of the nodes in the nodes array, the values of the weighted
 * sums in the sums array, and the psi_i values in the psi array. Returns the total error for the
 * test case at the input index represented by the parameter testCaseNum.
 *
 * Precondition: the input activation layer values and the network's weight values have been set,
 * and the network is in training mode
 */
double runTraining(int testCaseNum)
{
   double totalError = 0.0;
   double omega_i;
   double nodeVal;

   for (int j = 0; j < B; j++) // calculates and populates the hidden layer
   {
      nodeVal = 0.0;
      for (int k = 0; k < A; k++) nodeVal += nodes[0][k] * weights[0][k][j];

      sums[1][j] = nodeVal;
      nodes[1][j] = activationFunction(nodeVal);
   } // for (int j = 0; j < B; j++)

   // calculates and populates the output layer and relevant training values
   for (int i = 0; i < F; i++)
   {
      nodeVal = 0.0;
      for (int j = 0; j < B; j++) nodeVal += nodes[1][j] * weights[1][j][i];

      sums[numLayers][i] = nodeVal;
      nodes[numLayers][i] = activationFunction(nodeVal);

      omega_i = truth[testCaseNum].s[i] - nodes[numLayers][i];
      totalError += omega_i * omega_i;

      psi[i] = omega_i * activationFunctionDerivative(sums[numLayers][i]);
   } // for (int i = 0; i < F; i++)

   return totalError * 1.0 / 2.0;
} // double runTraining(int testCaseNum)

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
         for (int k = 0; k < A; k++) nodes[0][k] = truth[testCaseNum].f[k];

         double totalError = runTraining(testCaseNum);

         // update the maximum error for the current iteration if needed
         if (totalError > errorReached) errorReached = totalError;

         double capitalOmega_j;
         double capitalPsi_j;

         // calculates by how much we must change all weights and updates all weights accordingly
         for (int j = 0; j < B; j++)
         {
            capitalOmega_j = 0.0;
            for (int i = 0; i < F; i++)
            {
               capitalOmega_j += psi[i] * weights[1][j][i];

               // updates the ji weight because we're done with it
               weights[1][j][i] += lambda * nodes[1][j] * psi[i];
            } // for (int i = 0; i < F; i++)

            // updates the kj weights
            capitalPsi_j = capitalOmega_j * activationFunctionDerivative(sums[1][j]);
            for (int k = 0; k < A; k++) weights[0][k][j] += lambda * nodes[0][k] * capitalPsi_j;
         } // for (int j = 0; j < B; j++)
      } // for (int testCaseNum = 0; testCaseNum < numTruthTableCases; testCaseNum++)

      numIterations++;
      if (numIterations >= maxIterations) maxIterationsReached = true;
      if (errorReached < errorThreshold) errorThresholdReached = true;
   } // while (!maxIterationsReached && !errorThresholdReached)
} // void train()

/**
 * Reports on the network's behavior after training or running, where the boolean parameter
 * doInitialReport represents whether this method should print out the non-truth table portion of
 * the report or print out the truth table portion of the report, and the integer parameter
 * testCaseNum represents the index of the test case on which to report if doInitialReport is false
 *
 * Precondition: the network has either trained or run
 */
void report(bool doInitialReport, int testCaseNum)
{
   if (doInitialReport && training)
   {
      cout << "Training was terminated because of the following reason(s):\n";
      if (maxIterationsReached) cout << "\tThe maximum number of iterations was reached\n";
      if (errorThresholdReached) cout << "\tThe error threshold was reached\n";
      cout << "\n";

      cout << "Relevant values at the end of training:\n";
      cout << "\tThe number of iterations reached was " << numIterations << "\n";
      cout << "\tThe error reached was " << errorReached << "\n";
      cout << "\tThe amount of time training took was " << trainingTime.count() << " seconds\n";
      cout << "\n";
   } // if (doInitialReport && training)
   else
   {
      cout << "Test Case " << testCaseNum + 1 << ":\n";

      cout << "\tInput activation value(s):";
      for (int k = 0; k < A; k++) cout << " " << truth[testCaseNum].f[k];

      cout << "\n\tExpected output value(s):";
      for (int i = 0; i < F; i++) cout << " " << truth[testCaseNum].s[i];

      cout << "\n\tNetwork generated output value(s):";
      for (int i = 0; i < F; i++) cout << " " << nodes[numLayers][i];
      cout << "\n";
   } // else
} // void report(bool doInitialReport, int testCaseNum)

/**
 * The main method either trains or executes the network, depending upon the configuration
 * parameters, and then outputs a report. This method takes the standard C++ command line input
 * arguments as parameters
 */
int main(int argc, char* argv[])
{
   config(argc, argv);

   printOutConfigVals();

   allocateMemory();

   loadValues();

   if (!training) runRunning();
   else
   {
      chrono::time_point<chrono::high_resolution_clock> start =
         chrono::high_resolution_clock::now();

      train();

      chrono::time_point<chrono::high_resolution_clock> end = chrono::high_resolution_clock::now();
      trainingTime = end - start;
   }

   report(true, 0);
   for (int testCaseNum = 0; testCaseNum < numTruthTableCases; testCaseNum++)
   {
      for (int k = 0; k < A; k++) nodes[0][k] = truth[testCaseNum].f[k];
      runRunning();

      report(false, testCaseNum);
   } // for (int testCaseNum = 0; testCaseNum < numTruthTableCases; testCaseNum++)
} // int main(int argc, char* argv[])
