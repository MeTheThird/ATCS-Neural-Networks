/**
* Rohan Thakur
* Date Created: 2/7/22
*
* This file allows for the training and running of an A-B-1 2-connectivity-layer neural network
* based upon input configuration parameters
*/

#include <bits/stdc++.h>
#include <random>

using namespace std;

#define f first
#define s second

typedef pair<double, double> pdd;
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
vector<vector<double> > sums;                  // sums stores the values of the nodes in the network
                                               // before the activation function is applied i.e. for
                                               // each node, sums stores the sum over the values of
                                               // the nodes in the previous node layer multiplied by
                                               // their associated weight value. Note that the first
                                               // node layer of sums is just the input activation
                                               // layer, and the later node layers of sums follows
                                               // the aforementioned definition. Also note that the
                                               // indices represent the same characteristics as
                                               // those in the nodes array above.
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
* Allocates the necessary memory for the weights array and the deltaWeights array, only allocating
* memory for deltaWeights if the network is in training mode
*/
void allocateWeightsArrays()
{
   weights.resize(2);
   if (training) deltaWeights.resize(2);

   // allocates memory for the first connectivity layer
   weights[0].resize(B);
   if (training) deltaWeights[0].resize(B);
   for (int j = 0; j < B; j++)
   {
      weights[0][j].resize(F);
      if (training) deltaWeights[0][j].resize(F);
   } // for (int j = 0; j < B; j++)

   // allocates memory for the second connectivity layer
   weights[1].resize(A);
   if (training) deltaWeights[1].resize(A);
   for (int k = 0; k < A; k++)
   {
      weights[1][k].resize(B);
      if (training) deltaWeights[1][k].resize(B);
   } // for (int k = 0; k < A; k++)
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
   allocateWeightsArrays();
   allocateTruthTableArray();
   nodes.resize(3);    // the number of node layers is always one greater than the number of
                       // connectivity layers
   sums.resize(3);
   nodes[0].resize(A); // the first node layer is the activation layer with A nodes
   sums[0].resize(A);
   nodes[1].resize(B); // the second node layer is the hidden layer with B nodes
   sums[1].resize(B);
   nodes[2].resize(F); // the third and final node layer is the output layer with F nodes
   sums[2].resize(F);
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
      inputFile >> truth[caseNum].s[0];
   } // for (int caseNum = 0; caseNum < numTruthTableCases; caseNum++)
} // void loadTruthTableValues()

/**
* Loads in the weight values from a file the user has created
*/
void loadWeightValues()
{
   setupFileInputWithMessage("What is the full name of the file containing the weights?");

   // load the first connectivity layer of the weights array
   for (int k = 0; k < A; k++)
      for (int j = 0; j < B; j++)
         inputFile >> weights[0][k][j];

   // load the second connectivity layer of the weights array
   for (int j = 0; j < B; j++)
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
   return ((double) 1) / (((double) 1) + exp(-value));
} // double activationFunction(double value)

/**
* Returns the dervivative, evaluated at the given value, of the activation function defined above
*
* Precondition: the activation function is differentiable
*/
double activationFunctionDerivative(double value)
{
   return activationFunction(value) * (((double) 1) - activationFunction(value));
}

/**
* Calculates and returns the value of the node in the given node layer at the given index, where
* both the node layer and index are zero-indexed
*
* Precondition: nodeLayer > 0
*/
pdd calculateNode(int nodeLayer, int index)
{
   double ret = 0;

   if (nodeLayer == 1) // if the current node is in the hidden layer
   {
      for (int prev = 0; prev < A; prev++)
         ret += nodes[0][prev] * weights[0][prev][index];
   } // if (nodeLayer == 1)
   else // if the current node is in the output layer
   {
      for (int prev = 0; prev < B; prev++)
         ret += nodes[1][prev] * weights[1][prev][index];
   } // else
   return make_pair(activationFunction(ret), ret);
} // void calculateNode(int nodeLayer, int index)

/**
* Runs the network
*
* Precondition: the input activation layer has already been set
*/
void run()
{
   // calculates and populates the hidden layer
   for (int j = 0; j < B; j++)
   {
      pdd hiddenNodeVals = calculateNode(1, j);
      nodes[1][j] = hiddenNodeVals.f;
      sums[1][j] = hiddenNodeVals.s;
   } // for (int j = 0; j < B; j++)

   // calculates and populates the output layer
   pdd outputNodeVals = calculateNode(2, 0);
   nodes[2][0] = outputNodeVals.f;
   sums[2][0] = outputNodeVals.s;
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
         for (int k = 0; k < A; k++)
         {
            nodes[0][k] = truth[testCaseNum].f[k];
            sums[0][k] = truth[testCaseNum].f[k];
         } // for (int k = 0; k < A; k++)
         run();

         double omega_0 = truth[testCaseNum].s[0] - nodes[2][0];
         double totalError = 0;

         totalError += omega_0 * omega_0;
         totalError *= ((double) 1) / ((double) 2);
         if (totalError > maxError) maxError = totalError;

         // calculate and populate the delta weights array for the second connectivity layer
         double psi_0 = 0;
         psi_0 = omega_0 * activationFunctionDerivative(sums[2][0]);
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

         // update the weights array for the first connectivity layer
         for (int k = 0; k < A; k++)
            for (int j = 0; j < B; j++)
               weights[0][k][j] += deltaWeights[0][k][j];

         // update the weights array for the second connectivity layer
         for (int j = 0; j < B; j++)
            weights[1][j][0] += deltaWeights[1][j][0];
      } // for (int testCaseNum = 0; testCaseNum < numTruthTableCases; testCaseNum++)

      if (numIterations % 10000 == 0) cout << maxError << "\n"; // testing print statement
      numIterations++;
      if (numIterations >= maxIterations) maxIterationsReached = true;
      if (maxError < errorThreshold) errorThresholdReached = true;
   } // while (!maxIterationsReached && !errorThresholdReached)

   cout << "Training was terminated because of the following reason(s):\n";
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

   for (int caseNum = 0; caseNum < numTruthTableCases; caseNum++)
   {
      for (int k = 0; k < A; k++) nodes[0][k] = truth[caseNum].f[k];
      run();
      cout << "testCaseNum: " << caseNum << ", output: " << nodes[2][0] << "\n";
   } // for (int caseNum = 0; caseNum < numTruthTableCases; caseNum++)

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

   // TODO: can we have nested for loops without braces that have one line of code? See
   // generateRandomWeightValues() and loadWeightValues() and calculateNode(params)

   // TODO: is the 0 in the line "double capitalOmega_j = psi_0 * weights[n][j][0];" a magic number?
   // See train() at or around line 332

   // TODO: is the spillover indentation on line 334 (or around there) correct? It's the line where
   // I define capitalPsi_j

} // int main()
























