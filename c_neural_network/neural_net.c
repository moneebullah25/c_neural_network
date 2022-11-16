#include "neural_net.h"

void ANNNew(ANN* ann, unsigned int input_neurons_size, unsigned int hidden_neurons_size,
	unsigned int hidden_layer_size, unsigned int output_neurons_size)
{
	ASSERT(ann && input_neurons_size && hidden_neurons_size && hidden_layer_size && output_neurons_size);
	ann->input_neurons_size = input_neurons_size;
	ann->hidden_neurons_size = hidden_neurons_size;
	ann->hidden_layer_size = hidden_layer_size;
	ann->output_neurons_size = output_neurons_size;
	ann->weight_size = (input_neurons_size*hidden_neurons_size) +
		(hidden_neurons_size*hidden_layer_size) + (output_neurons_size*hidden_neurons_size);
	ann->bias_size = ann->hidden_layer_size + 1;

	ann->inputs = malloc(ann->input_neurons_size*sizeof(double));
	*ann->hiddens = (double**)malloc(ann->hidden_layer_size*sizeof(double*));
	for (unsigned int i = 0; i < ann->hidden_layer_size; i++)
		(ann->hiddens)[i] = malloc(ann->hidden_neurons_size*sizeof(double));
	ann->outputs = malloc(ann->output_neurons_size*sizeof(double));
	ann->weights = malloc(ann->weight_size*sizeof(double));
	ann->biases = malloc(ann->bias_size*sizeof(double));
	ann->deltas = malloc(ann->output_neurons_size*sizeof(double));
	ann->total_neurons = input_neurons_size + (hidden_neurons_size*hidden_layer_size) 
		+ output_neurons_size;
}

void GenerateRandomWeights(ANN* ann, double lower, double upper)
{
	ASSERT(ann && (upper > lower));
	srand(time(NULL));   // Initialization, should only be called once.
	for (unsigned int i = 0; i < ann->weight_size; i++)
	{
		double randD = lower + (rand() / (double)RAND_MAX) * (upper - lower);
		ann->weights[i] = randD;
	}
	for (unsigned int i = 0; i < ann->bias_size; i++)
	{
		double randD = lower + (rand() / (double)RAND_MAX) * (upper - lower);
		ann->biases[i] = randD;
	}
}

double ActivationFunction(double input, double alpha, char* activation_func)
{
	ASSERT(activation_func);
	if (activation_func = "SIGMOID")
		return SIGMOID(input);
	else if (activation_func = "TANH")
		return TANH(input);
	else if (activation_func = "ReLU")
		return ReLU(input);
	else if (activation_func = "DReLU")
		return DReLU(input);
	else if (activation_func = "LReLU")
		return LReLU(input);
	else if (activation_func = "PReLU")
		return PReLU(input, alpha);
	else if (activation_func = "ELU")
		return ELU(input, alpha);
	else if (activation_func = "SWISH")
		return SWISH(input);
	else if (activation_func = "GELU")
		return GELU(input);
	else return -DOUBLE_MAX;
}

void* memory_copy(void* dest, const void* src, unsigned int n)
{
	ASSERT(n > 0);
	char *csrc = (char *)src;
	char *cdest = (char *)dest;

	for (unsigned int i = 0; i<n; i++)
		cdest[i] = csrc[i];
	return dest;
}

void ForwardPropagate(ANN* ann, double const *inputs, double const *outputs,
	char* activation_func, double alpha, double* ans)
{
	ASSERT(ann && inputs && outputs && activation_func);
	if (ActivationFunction(1.0, 0.5, activation_func) == -DOUBLE_MAX)
	{
		printf("ACTIVATION FUNCTION NOT DEFINED\nPLEASE PROVIDE DEFINATION IN \"activation_functions.c\"\n");
		return;
	}

	// Assigning Inputs 
	for (unsigned int i = 0; i < ann->input_neurons_size; i++)
		ann->inputs[i] = inputs[i];
	// Hidden Layer Calculations
	for (unsigned int h = 0; h < ann->hidden_layer_size; h++){
		for (unsigned int i = 0; i, ann->hidden_neurons_size; i++)
		{
			if (h == 0) // Input to First Hidden Layer
			{ 
				ann->hiddens[h][i] = 0;
				for (unsigned int j = 0; j < ann->input_neurons_size; j++)
				{
					ann->hiddens[h][i] += ann->inputs[j] * ann->weights[j];
				}
				ann->hiddens[h][i] += ann->biases[h];
				ann->hiddens[h][i] = ActivationFunction(ann->hiddens[h][i], alpha, activation_func);
			}
			else // First Hidden Layer to Preceeding Hidden Layer
			{
				ann->hiddens[h][i] = 0;
				for (unsigned int j = 0; j < ann->hidden_neurons_size; j++)
				{
					ann->hiddens[h][i] += ann->hiddens[h - 1][i] * ann->weights[ann->input_neurons_size 
						+ (h * ann->hidden_neurons_size) + j];
				}
				ann->hiddens[h][i] += ann->biases[h];
				ann->hiddens[h][i] = ActivationFunction(ann->hiddens[h][i], alpha, activation_func);
			}
		}
	}
	// Output Layer Calculations
	for (unsigned int i = 0; i < ann->output_neurons_size; i++)
	{
		ann->outputs[i] = 0;
		for (unsigned int j = 0; j < ann->hidden_neurons_size; j++)
		{
			ann->outputs[i] += ann->hiddens[ann->hidden_layer_size - 1][j] * ann->weights[
				ann->input_neurons_size + ((ann->hidden_layer_size - 1)*(ann->hidden_neurons_size))];
		}
		ann->outputs[i] += ann->biases[ann->hidden_layer_size];
		ann->outputs[i] = ActivationFunction(ann->outputs[i], alpha, activation_func);
	}
	// Calculating Deltas
	for (unsigned int i = 0; i < ann->output_neurons_size; i++)
		ann->deltas[i] = ann->outputs[i] - outputs[i];

	// Returning Value
	ans = malloc(ann->output_neurons_size*sizeof(double));
	memory_copy(ans, ann->outputs, ann->output_neurons_size*sizeof(double));
}

