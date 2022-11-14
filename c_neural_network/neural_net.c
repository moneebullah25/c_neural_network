#include "neural_net.h"

double neuron(double* x, double* w, unsigned int input_size, double b, double(*ActFunc)(double x))
{
	double in = b;
	for (unsigned int i = 0; i < input_size; i++)
		in += x[i] * w[i];
	double out = ActFunc(in);
	return out;
}


