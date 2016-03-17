//
//  graphic_f.cpp
//  Neural Networks
//
//  Created by Alexis Louis on 16/03/2016.
//  Copyright © 2016 Alexis Louis. All rights reserved.
//

#include "graphic_f.hpp"

void drawFront(FFN *network, int window_size){
    sf::RenderWindow window(sf::VideoMode(window_size, window_size), "FFNN");
    sf::Texture graph_tex;
    sf::Sprite graph;
    sf::Uint8 *pixels = new sf::Uint8[window_size*window_size*4];
    graph_tex.create(window_size, window_size);
    graph.setTexture(graph_tex);
    bool draw = false;
    window.setVerticalSyncEnabled(true);
    while (window.isOpen())
    {
        window.clear();
        sf::Event event;
        float grid_value, x_norm, y_norm;
        vector<float> grid_input;
        int r,g,b;
        
        
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }
        
        if(draw==false){
            draw = true;
            
            for(int x = 0; x < window_size; x++)
            {
                for(int y = 0; y < window_size; y++)
                {
                    x_norm = x/(float)window_size;
                    y_norm = y/(float)window_size;
                    grid_input = {x_norm,y_norm,1};
                    network->sim(grid_input);
                    grid_value = network->get_ffn_outputs()[0];
                    if(grid_value>0.5){
                        r = 255;
                        g = 0;
                        b = 0;
                    }else{
                        r = 0;
                        g = 0;
                        b = 255;
                    }
                    pixels[(window_size * y + x)*4] = r; // R
                    pixels[(window_size * y + x)*4 + 1] = g; // G
                    pixels[(window_size * y + x)*4+ 2] = b; // B
                    pixels[(window_size * y + x)*4 + 3] = 255; // A
                }
            }
        }
        
        graph_tex.update(pixels);
        window.draw(graph);
        window.display();
    }
}