#ifndef _NEURAL_NET_HEADER_
#define _NEURAL_NET_HEADER_

#include <stdlib.h>
#include <time.h>
#include "activation_functions.c"

#define RANDMAX 0x7ff

// Default arguments aren't allowed in C

typedef struct {
	unsigned int input_neurons_size, hidden_neurons_size, hidden_layer_size, output_neurons_size;
	double* inputs;
	double* outputs;
	double** hiddens; // If more than one hidden layer otherwise treated as double* hiddens
	double* weights; unsigned int weight_size;
	double* biases; unsigned int bias_size;
	double* deltas; 
	unsigned int total_neurons;
} ANN;

// Initialize Artificial Neural Network
void ANNNew(ANN* ann, unsigned int input_neurons_size, unsigned int hidden_neurons_size,
	unsigned int hidden_layer_size, unsigned int output_neurons_size);

// Generate Random Weights from lower to upper inclusive
void GenerateRandomWeights(ANN* ann, double lower, double upper);

/* Return output neuron values after Forward Propogating once, changes delta value after output
For PReLU and ELU Alpha must be provided [0, 1]. And for others Activations Functions 0 must be provided for alpha */
void ForwardPropagate(ANN* ann, double const *inputs, double const *outputs, 
	char* activation_func, double alpha, double* ans);

// Return weights list after Back Propagating once
double* BackwardPropagate(ANN* ann, double const *inputs, double const *outputs, double learning_rate); 

// Disposes of all allocated memory for ANN
void ANNDispose(ANN* ann);

#endif