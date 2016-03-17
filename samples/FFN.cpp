//
//  FFN.cpp
//  Neural Networks
//
//  Created by Alexis Louis on 15/03/2016.
//  Copyright © 2016 Alexis Louis. All rights reserved.
//

#include "FFN.hpp"

FFN::FFN(){
}

void FFN::add_layer(Layer *l){
    layers.push_back(l);
}

void FFN::initFFN(int nb_inputs_neurons, int nb_hidden_neurons, int nb_outputs_neurons){
    this->add_layer(new Layer(0, nb_inputs_neurons, this));
    this->add_layer(new Layer(1, nb_hidden_neurons, this));
    this->add_layer(new Layer(2, nb_outputs_neurons, this));
}

void FFN::sim(vector<float> inputs){
    layers[0]->set_inputs(inputs);
    for(int indice = 0; indice < this->get_nb_layers(); indice++){
        this->get_layer_at(indice)->forward_propagate();
    }
}

void FFN::test(vector<vector<float>> Xtest, vector<vector<float>> Tt){
    vector<float> current_input;
    vector<float> current_output;
    int target_max_indice;
    int output_max_indice;
    int true_guess = 0;
    int false_guess = 0;
    float ratio;
    for(int i=0; i<Xtest.size(); i++){
        current_input = Xtest[i];
        this->sim(current_input);
        current_output = this->get_ffn_outputs();
        target_max_indice = distance(Tt[i].begin(), max_element(Tt[i].begin(),Tt[i].end()));
        output_max_indice = distance(current_output.begin(), max_element(current_output.begin(),current_output.end()));
        if(target_max_indice==output_max_indice){
            true_guess++;
        }else{
            false_guess++;
        }
    }
    cout << "Correct : " << true_guess << endl;
    cout << "Incorrect : " << false_guess << endl;
    ratio = true_guess*100.0/(false_guess+true_guess);
    cout << "Ratio : " << ratio << endl;
}

void FFN::train_by_error(vector<vector<float>> inputs, vector<vector<float>> targets, float target_error){
    float error;
    do{
        error = 0;
        for(int i=0; i<inputs.size(); i++){
            
            layers[0]->set_inputs(inputs[i]);
            this->set_targets(targets[i]);
            for(int indice = 0; indice < this->get_nb_layers(); indice++){
                this->get_layer_at(indice)->forward_propagate();
                
            }
            for(int j=0; j<targets[i].size();j++){
                error += (targets[i][j]-this->get_ffn_outputs()[j])*(targets[i][j]-this->get_ffn_outputs()[j]);
            }
            
            for(int indice = this->get_nb_layers()-1; indice >=0; indice--){
                this->get_layer_at(indice)->calc_deltas();
                this->get_layer_at(indice)->calc_new_weights();
            }
        }
        
        
    }while(error>target_error);
    cout << "Stopping at error : " << error << endl;
}

void FFN::train_by_iteration(vector<vector<float>> inputs, vector<vector<float>> targets, int target_iteration){
    int ite = 0;
    do{
        ite++;
        //error = 0;
        for(int i=0; i<inputs.size(); i++){
            
            layers[0]->set_inputs(inputs[i]);
            this->set_targets(targets[i]);
            for(int indice = 0; indice < this->get_nb_layers(); indice++){
                this->get_layer_at(indice)->forward_propagate();
                
            }
            for(int j=0; j<targets[i].size();j++){
                //error += (targets[i][j]-this->get_ffn_outputs()[j])*(targets[i][j]-this->get_ffn_outputs()[j]);
            }
            
            for(int indice = this->get_nb_layers()-1; indice >=0; indice--){
                this->get_layer_at(indice)->calc_deltas();
                this->get_layer_at(indice)->calc_new_weights();
            }
        }
        
    }while(ite<target_iteration);
    cout << "Stopping at iteration : " << ite << endl;
}

void FFN::about(){
    cout << "Number of layers : "<< layers.size() << endl;
    cout << "Number of input neurons : "<< layers[0]->get_nb_neurons() << endl;
    cout << "Number of hidden neurons : "<< layers[1]->get_nb_neurons() << endl;
    cout << "Number of output neurons : "<< layers[2]->get_nb_neurons() << endl;
}

vector<float> FFN::get_ffn_outputs(){
    return this->get_layer_at(this->get_nb_layers()-1)->get_outputs();
}

void FFN::set_targets(vector<float> tar){
    targets = tar;
}