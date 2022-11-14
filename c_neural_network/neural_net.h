#ifndef _NEURAL_NET_HEADER_
#define _NEURAL_NET_HEADER_

double neuron(double* x, double* w, unsigned int input_size, double b, double(*ActFunc)(double x));

#endif