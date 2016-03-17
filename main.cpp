
#include "Header.h"

int main()
{

    //-------XOR EXAMPLE--------//
    FFN *XORnetwork = new FFN();
    XORnetwork->initFFN(3, 5, 1);
    //XORnetwork->about();
    
    vector<vector<float>> XOR_APP = {{0,0,1},{0,1,1},{1,0,1},{1,1,1}};
    vector<vector<float>> XOR_TAR = {{0},{1},{1},{0}};
    
    XORnetwork->train_by_error(XOR_APP,XOR_TAR,0.0001);
    //cout << "Calculation done ... Drawing graph" << endl;
    //drawFront(XORnetwork, 1000); // requires SFML lib
    
    
    
    //-------DIGIT RECOGNITION EXAMPLE--------//
    FFN *DIGITnetwork = new FFN();
    DIGITnetwork->initFFN(55, 20, 10);
    //DIGITnetwork->about();
    
    vector<vector<float>> Xapp = readMatFromFile("/Users/Alexis/Documents/SFML/Neural Networks/sfmlMac/samples/Xapp.txt");
    vector<vector<float>> Ta = readMatFromFile("/Users/Alexis/Documents/SFML/Neural Networks/sfmlMac/samples/TA.txt");
    vector<vector<float>> Xtest = readMatFromFile("/Users/Alexis/Documents/SFML/Neural Networks/sfmlMac/samples/Xtest.txt");
    vector<vector<float>> Tt = readMatFromFile("/Users/Alexis/Documents/SFML/Neural Networks/sfmlMac/samples/TT.txt");
    
    DIGITnetwork->train_by_iteration(Xapp,Ta,1000);
    DIGITnetwork->test(Xtest,Tt);
    
}
