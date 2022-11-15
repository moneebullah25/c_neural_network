#include "neural_net.h"

double neuron(double* x, double* w, unsigned int input_size, double b, char* actFunc, float alpha)
{
	double in = b;
	for (unsigned int i = 0; i < input_size; i++)
		in += x[i] * w[i];

	if (actFunc = "SIGMOID")
		return SIGMOID(in);
	else if (actFunc = "TANH")
		return TANH(in);
	else if (actFunc = "ReLU")
		return ReLU(in);
	else if (actFunc = "DReLU")
		return DReLU(in);
	else if (actFunc = "LReLU")
		return LReLU(in);
	else if (actFunc = "PReLU")
		return PReLU(alpha, in);
	else if (actFunc = "ELU")
		return ELU(alpha, in);
	else if (actFunc = "SWISH")
		return SWISH(in);
	else if (actFunc = "GELU")
		return GELU(in);
	else
		return -DOUBLE_MAX;
}

void initializeRandomWeights(double* x, unsigned int input_size, int lower, int upper)
{
	time_t t;
	srand((unsigned)time(&t));
	for (unsigned int i = 0; i < input_size; i++)
		x[i] = (rand() % upper) + lower;
}

