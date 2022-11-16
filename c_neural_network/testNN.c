#include "neural_net.h"




int main()
{
	double w[2] = {.15, .2};
	double x[2] = {.05, .1};

	double result = neuron(x, w, 2, .35, "SIGMOID", 0);
	printf("%.7f ", result);
}