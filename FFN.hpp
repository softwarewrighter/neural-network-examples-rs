//
//  FFN.hpp
//  Neural Networks
//
//  Created by Alexis Louis on 15/03/2016.
//  Copyright © 2016 Alexis Louis. All rights reserved.
//



#ifndef FFN_hpp
#define FFN_hpp

#include "Header.h"

using namespace std;
class Layer;

class FFN{
    
public:
    FFN();
    
    void initFFN(int nb_inputs, int nb_hidden_neurons, int nb_outputs);
    void sim(vector<float> inputs);
    void test(vector<vector<float>> Xtest, vector<vector<float>> Tt);
    void train_by_iteration(vector<vector<float>> inputs, vector<vector<float>> targets, int target_iteration);
    void train_by_error(vector<vector<float>> inputs, vector<vector<float>> targets, float target_error);
    
    void about();
    
    vector<float> get_ffn_outputs();
    void set_targets(vector<float> tar);
    vector<float> get_targets(void){return targets;};
    int get_nb_layers(){return layers.size();};
    Layer* get_layer_at(int indice){return layers[indice];};
    

private:
    vector<Layer*> layers;
    vector<float> targets;
    void add_layer(Layer* l);
};

#endif /* FFN_hpp */
