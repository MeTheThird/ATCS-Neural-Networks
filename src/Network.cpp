/**
* Rohan Thakur
* Date Created: 2/7/22
*
* This file allows for the training and running of an A-B-1 2-connectivity-layer neural network
* based upon input configuration parameters
*/

// TODO: include list of functions in the module in the header comment

#include <bits/stdc++.h>

using namespace std;

boolean training;           // training represents whether the network is training or not
// TODO: boolean that represents whether we're loading in random weights for training or set weights
// TODO: for training
int N, A, B, F;             // N is the number of connectivity layers in the network, A is the
                            // number of activation nodes in the input activation layer, B is the
                            // number of nodes in the hidden layer, and F is the number of output
                            // nodes
vector<double> weights[][]; // the weights array where the first index represents the connectivity
                            // layer, and the second and third indices represent the node TODO FINISH THIS COMMENT

/**
* Prompts the user for the name of the input configuration parameter file to use and then sets that
* file as the standard input stream
*/
void setupFileInput()
{
   string filename;
   cout << "What is the name of your input parameter file?\n";
   cin >> filename;
   freopen(filename.c_str(), "r", stdin);
} // void setupFileInput(string filename)

/**
* Takes in the input configuration parameters and sets their corresponding global variable values
* accordingly
*/
// takes in input parameters
void config()
{

}

// prints out input parameters

/**
* The main method which either trains or executes the network, depending upon the input parameters
*/
int main()
{
   setupFileInput();
   // TODO: make everything their own functions
   // TODO: write down the expected outline of the input parameter file in the README file
   cin >> N >> A >> B >> F;

   cout << "N: " << N << ", A: " << A << ", B: " << B << ", F: " << F << "\n";

   // TODO: make the weights file a separate file from the config file

} // int main()
























