# Feed-Forward-Neural-Network


c++ implementation of multi-layer feed forward neural networks with back propagation algorithm.
Optimization + features to be added soon.

### Examples of main.cpp
main.cpp contains an example of XOR function learning and a more advanced application case: digit recognition.

#### XOR
Console output for XOR network :
```
Number of layers : 3
Number of input neurons : 2
Number of hidden neurons : 4
Number of output neurons : 1

Test at iteration 3475 :
XOR(0,0) = 0
XOR(0,1) = 1
XOR(1,0) = 1
XOR(1,1) = 0
```
Visualization of non-linear discrimination for XOR :
![alt tag](http://i.imgur.com/UkW1Qsc.png)

#### Digit recognition
Samples used can be found in _samples/*.txt_.
Learning and test variables are already processed from original histogram pool through normalization and PCA projection.
```
X* : experiences * variables
T* : targets
```
Dicrimination result of a neural network trained with 967 experiences in 55 dimensions tested with 967 unlearned inputs :
```
Stopping learning at iteration : 1000
Correct : 902
Incorrect : 65
Ratio : 93.2782
```
Note that performances are strongly dependent of representation choices in variable space. 


#### References
- http://www.cs.bham.ac.uk/~jxb/NN/l7.pdf
- http://www.di.unito.it/~cancelli/retineu06_07/FNN.pdf
- ESEO Data Mining Course - M. Feuilloy
