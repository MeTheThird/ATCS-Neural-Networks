/**
* Rohan Thakur
* Date Created: 2/7/22
*
* This file allows for the training and running of an A-B-1 neural network based upon input
* parameters
*/

// TODO: include list of functions in the module in the header comment

#include <bits/stdc++.h>

using namespace std;

int A, B, F;

/**
* Prompts the user for the name of the input parameter file to use and then sets that file as the
* standard input stream
*/
void setupFileInput()
{
   string filename;
   cout << "What is the name of your input parameter file?\n";
   cin >> filename;
   freopen(filename.c_str(), "r", stdin);
} // void setupFileInput(string filename)


/**
* The main method which either trains or executes the network, depending upon the input parameters
*/
int main()
{
   setupFileInput();
   // TODO: make everything their own functions
   cin >> A >> B >> F;

   cout << "A: " << A << ", B: " << B << ", F: " << F << "\n";

} // int main()
























