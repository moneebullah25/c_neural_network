#ifndef _NEURAL_NET_HEADER_
#define _NEURAL_NET_HEADER_

#include <stdlib.h>
#include "activation_functions.c"

// Default arguments aren't allowed in C
double neuron(double* x, double* w, unsigned int input_size, double b, char* actFunc, float alpha);


#endif