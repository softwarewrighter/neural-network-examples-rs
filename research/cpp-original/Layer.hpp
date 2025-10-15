//
//  Layer.hpp
//  Neural Networks
//
//  Created by Alexis Louis on 15/03/2016.
//  Copyright © 2016 Alexis Louis. All rights reserved.
//

#ifndef Layer_hpp
#define Layer_hpp

#include "Header.h"

using namespace std;
class FFN;

class Layer
{
public:
    Layer(int indice, int nb_neurons, FFN *net);
    void forward_propagate(void);
    void back_propagate(void);

    vector<float> get_outputs(){return outputs;};
    vector<float> get_inputs(){return inputs;};
    vector<float> get_deltas(){return deltas;};
    vector<vector<float>> get_weights(){return weights;};
    int get_nb_neurons(){return nb_neurons;};
    int get_indice(){return indice;};
    
    void set_outputs(vector<float> out);
    void set_inputs(vector<float> in);

    void show_weights_matrix();
    void calc_deltas();
    void calc_new_weights();

private:
    FFN *network;
    int indice;
    int nb_neurons;
    vector<vector<float>> weights;

//ForwardProp
    vector<float> inputs;
    vector<float> outputs;
    void calc_inputs();
    void calc_outputs();

//Back Propagation
    vector<float> deltas;


};

#endif /* Layer_hpp */
