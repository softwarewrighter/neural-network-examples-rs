
#include "Header.h"


int main()
{
    FFN *network = new FFN();
    network->initFFN(3, 5, 1);
    network->about();
    vector<vector<float>> XOR_APP = {{0,0,1},{0,1,1},{1,0,1},{1,1,1}};
    vector<vector<float>> XOR_TAR = {{0},{1},{1},{0}};
    network->train(XOR_APP,XOR_TAR,0.0001);
    cout << "Calculation done ... Drawing graph" << endl;
    drawFront(network, 1000); // requires SFML lib
}
