//
//  HiddenLayer.cpp
//  Neural Networks
//
//  Created by Alexis Louis on 15/03/2016.
//  Copyright © 2016 Alexis Louis. All rights reserved.
//

#include "Layer.hpp"

Layer::Layer(int indice, int nb_neurons, FFN *net){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1, 1);
    this->indice = indice;
    this->nb_neurons = nb_neurons;
    this->network = net;
    //Init weights
    vector<float> temp_weights;
    if(indice){ // Input layer has no weights
        for(int i=0; i < net->get_layer_at(indice-1)->get_nb_neurons(); i++){
            for(int j=0; j < nb_neurons; j++){
                temp_weights.push_back(dis(gen));
            }
            weights.push_back(temp_weights);
            temp_weights.clear();
        }
    }
}

void Layer::show_weights_matrix(){
    cout << "Matrix Size : " << weights.size() << "x" << weights[0].size() << endl;
    for(int i=0; i<weights.size(); i++){
        for(int j=0; j<weights[i].size(); j++){
            cout << " " << weights[i][j];
        }
        cout << '\n';
    }
}

void Layer::calc_inputs(){
    if(indice>0){
        inputs.clear();
        vector<float> prev_outputs = network->get_layer_at(indice-1)->get_outputs();
        float temp_weight = 0;
        for(int col=0; col<weights[0].size(); col++){
            for (int li=0; li<weights.size(); li++) {
                temp_weight += prev_outputs[li]*weights[li][col];
            }
            inputs.push_back(temp_weight);
            temp_weight=0;
        }
    }
}

void Layer::calc_outputs(){
    outputs.clear();
    if(indice==0 || indice==network->get_nb_layers()-1){
        outputs = inputs;
    }else{
        for(int i=0; i<inputs.size(); i++){
            outputs.push_back(1/(1+exp(-inputs[i])));
        }
    }
}

void Layer::forward_propagate(){
    calc_inputs();
    calc_outputs();
}

void Layer::back_propagate(){
    calc_deltas();
    calc_new_weights();
}

void Layer::calc_deltas(){
    if(indice>0){
        deltas.clear();
        float deltas_sum = 0;
        vector<float> targets = network->get_targets();
        if(indice==network->get_nb_layers()-1){
            for(int i=0; i<this->get_nb_neurons(); i++){
                deltas.push_back((targets[i]-outputs[i]));
            }
        }else{
            vector<vector<float>> next_weights = network->get_layer_at(indice+1)->get_weights();
            vector<float> next_deltas = network->get_layer_at(indice+1)->get_deltas();
            for(int i=0; i<this->get_nb_neurons(); i++){
                for(int j=0; j<next_weights[i].size(); j++){
                    deltas_sum += next_weights[i][j] * next_deltas[j];
                }
                deltas.push_back(deltas_sum * outputs[i]*(1-outputs[i]));
                deltas_sum = 0;
            }
        }
    }
}

void Layer::calc_new_weights(){
    if(indice>0){
        float eta = 0.01;
        vector<float> prev_outputs = network->get_layer_at(indice-1)->get_outputs();
        for(int i=0;i<weights.size();i++){
            for(int j=0;j<weights[i].size();j++){
                weights[i][j] += eta * deltas[j] * prev_outputs[i];
            }
        }
    }
}

void Layer::set_inputs(vector<float> in){
    inputs = in;
}

void Layer::set_outputs(vector<float> out){
    outputs = out;
}