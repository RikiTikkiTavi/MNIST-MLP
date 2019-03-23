# MNIST-MLP
Project to learn Multi Layer Perceptron model and Stochastic Gradient Descent with some heuristics as learning rate decay, momentum and Nesterov

### From scratch:
Was created MLP with 2 hidden layers. Error metric: Mean Squared Error. Optimisation: SGD. 
Could be found at https://github.com/RikiTikkiTavi/MNIST-MLP/tree/master/src/self_realisation


### Using keras tensorflow:
Was created MLP with 2 hidden layers. Error metric: Mean Squared Error. Optimisation: SGD with decay, momentum and Nesterov.
Parameters where tuned using GridSearch implementation from Talos library.
Accuracy 95.69% is achieved on test data.
Code could be found at https://github.com/RikiTikkiTavi/MNIST-MLP/tree/master/src/tf_realisation
Trained model: https://github.com/RikiTikkiTavi/MNIST-MLP/blob/master/src/tf_realisation/mnist_mlp_sgd_mse_model.h5
