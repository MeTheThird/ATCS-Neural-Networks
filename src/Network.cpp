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
* Allocates the necessary memory for the weights array and the deltaWeights array, only allocating
* memory for deltaWeights if the network is in training mode
*/
void allocateWeightsArrays()
{
   weights.resize(numLayers);
   if (training) deltaWeights.resize(numLayers);
   for (int n = 0; n < numLayers; n++)
   {
      if (n == numLayers - 1)
      {
         weights[n].resize(B);
         if (training) deltaWeights[n].resize(B);
         for (int j = 0; j < B; j++)
         {
            weights[n][j].resize(F);
            if (training) deltaWeights[n][j].resize(F);
         } // for (int j = 0; j < B; j++)
      } // if (n == numLayers - 1)
      else
      {
         weights[n].resize(A);
         if (training) deltaWeights[n].resize(A);
         for (int k = 0; k < A; k++)
         {
            weights[n][k].resize(B);
            if (training) deltaWeights[n][k].resize(B);
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
   allocateWeightsArrays();
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
      for (int left = 0; left < weights[n].size(); left++)
         for (int right = 0; right < weights[n][left].size(); right++)
            inputFile >> weights[n][left][right];
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
   for (int n = 0; n < numLayers; n++)
      for (int left = 0; left < weights[n].size(); left++)
         for (int right = 0; right < weights[n][left].size(); right++)
            weights[n][left][right] = getRandomNumberBetween(minRandVal, maxRandVal);
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
*/
double activationFunctionDerivative(double value)
{
   return activationFunction(value) * (((double) 1) - activationFunction(value));
}

/**
* Returns the inverse, evaluated at the given value, of the activation function defined above
*/
double inverseActivationFunction(double value)
{
   return -log(((double) 1) / value - ((double) 1));
}

/**
* Calculates and returns the value of the node in the given node layer at the given index, where
* both the node layer and index are zero-indexed
*
* Precondition: nodeLayer > 0
*/
double calculateNode(int nodeLayer, int index)
{
   double ret = 0;
   for (int prev = 0; prev < weights[nodeLayer - 1].size(); prev++)
      ret += nodes[nodeLayer - 1][prev] * weights[nodeLayer - 1][prev][index];
   return activationFunction(ret);
} // void calculateNode(int nodeLayer, int index)

/**
* Runs the network
*
* Precondition: the input activation layer has already been set
*/
void run()
{
   // nodeLayer begins at 1 because the input activation layer has already been set
   for (int nodeLayer = 1; nodeLayer <= numLayers; nodeLayer++)
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

         double omega_0 = 0;
         double totalError = 0;
         for (int i = 0; i < F; i++)
         {
            double omega = truth[testCaseNum].s[i] - nodes[numLayers][i];

            if (!i) omega_0 = omega;
            totalError += omega * omega;
         } // for (int i = 0; i < F; i++)

         totalError *= ((double) 1) / ((double) 2);
         if (totalError > maxError) maxError = totalError;

         // calculate and populate the delta weights array
         double psi_0 = 0;
         for (int n = numLayers - 1; n >= 0; n--)
         {
            if (n == numLayers - 1)
            {
               for (int i = 0; i < F; i++)
               {
                  psi_0 = omega_0 *
                     activationFunctionDerivative(inverseActivationFunction(nodes[n + 1][i]));
                  for (int j = 0; j < B; j++) deltaWeights[n][j][i] = lambda * nodes[n][j] * psi_0;
               } // for (int i = 0; i < F; i++)
            } // if (n == numLayers - 1)
            else
            {
               for (int j = 0; j < B; j++)
               {
                  double capitalOmega_j = psi_0 * weights[n][j][0];
                  double capitalPsi_j = capitalOmega_j *
                     activationFunctionDerivative(inverseActivationFunction(nodes[n + 1][j]));
                  for (int k = 0; k < A; k++)
                     deltaWeights[n][k][j] = lambda * nodes[n][k] * capitalPsi_j;
               } // for (int j = 0; j < B; j++)
            } // else
         } // for (int n = 0; n < numLayers; n++)

         // update the weights array
         for (int n = 0; n < numLayers; n++)
            for (int left = 0; left < weights[n].size(); left++)
               for (int right = 0; right < weights[n][left].size(); right++)
                  weights[n][left][right] += deltaWeights[n][left][right];
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
      cout << "testCaseNum: " << caseNum << ", output: " << nodes[numLayers][0] << "\n";
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
























