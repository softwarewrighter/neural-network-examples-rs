#include <SFML/Graphics.hpp>
#include "Header.h"
#include <unistd.h>

void dispVec(vector<float> v){
    cout << "{ ";
    for(int i=0;i<v.size(); i++){
        cout << v[i] << " ";
    }
    cout << "}\n";
}

int main()
{
    FFN *network = new FFN();
    network->initFFN(3, 4, 1);
    network->about();
    
    vector<float> XOR00= {0,0,1};
    vector<float> target00= {0};
    
    vector<float> XOR01= {0,1,1};
    vector<float> target01= {1};
    
    vector<float> XOR10= {1,0,1};
    vector<float> target10= {1};
    
    vector<float> XOR11= {1,1,1};
    vector<float> target11= {0};
    
    vector<float> out_test;
    int ite = 0;
    double error;
    do{
        error = 0;
        ite++;
        network->train(XOR00, target00);
        error += (target00[0]-network->get_ffn_outputs()[0])*(target00[0]-network->get_ffn_outputs()[0]);
        network->train(XOR01, target10);
        error += (target01[0]-network->get_ffn_outputs()[0])*(target01[0]-network->get_ffn_outputs()[0]);
        network->train(XOR10, target10);
        error += (target10[0]-network->get_ffn_outputs()[0])*(target10[0]-network->get_ffn_outputs()[0]);
        network->train(XOR11, target11);
        error += (target11[0]-network->get_ffn_outputs()[0])*(target11[0]-network->get_ffn_outputs()[0]);
    }while(error > 0.001);
    
    cout << '\n' <<"Test at iteration " << ite << " :" << endl;
    
    network->sim(XOR00);
    float neural_XOR00 = network->get_ffn_outputs()[0];
    cout << "XOR(0,0) = " << round(neural_XOR00) << endl;
    network->sim(XOR01);
    float neural_XOR01 = network->get_ffn_outputs()[0];
    cout << "XOR(0,1) = " << round(neural_XOR01) << endl;
    network->sim(XOR10);
    float neural_XOR10 = network->get_ffn_outputs()[0];
    cout << "XOR(1,0) = " << round(neural_XOR10) << endl;
    network->sim(XOR11);
    float neural_XOR11 = network->get_ffn_outputs()[0];
    cout << "XOR(1,1) = " << round(neural_XOR11) << endl;
    
}