from evoman.controller import Controller
import numpy as np
import time
from itertools import product
import random
from functools import reduce
from math import floor

class controller_automata(Controller):

    def __init__(self):
        self.rulebook = {}

    def create_random_rulebook(self):
        # list of keys : 
        keys = list(product([0,1], repeat=20))
        # possible outputs : 
        outputs = list(product([0,1], repeat=5))

		# convert to lists :
        keys = [reduce(lambda x,y:str(x) + str(y),k) for k in keys]
        outputs = [list(o) for o in outputs]
        # iteratively create dict with random outputs: 
        for key in keys:
            ind = random.randint(0,24)
            self.rulebook[key] = outputs[ind]

    def control(self,inputs):
        # binarize inputs : 
        inputs_x = [int(round(x/736.)) for x in inputs[0::2]]
        inputs_y = [int(round(y/512.))  for y in inputs[1::2]]

        binned_inputs = []
        for i in range(10):
            binned_inputs.append(inputs_x[i])
            binned_inputs.append(inputs_y[i])
        
		# reduce to string :
        print(binned_inputs)
        binned_inputs = reduce(lambda x,y:str(x) + str(y),binned_inputs)

        return self.rulebook[binned_inputs]

# implements controller structure for enemy
class enemy_controller(Controller):
	def __init__(self):
		# Number of hidden neurons
		self.n_hidden = [10]

	def control(self, inputs,controller):
		# Normalises the input using min-max scaling
		inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))

		if self.n_hidden[0]>0:
			# Preparing the weights and biases from the controller of layer 1

			# Biases for the n hidden neurons
			bias1 = controller[:self.n_hidden[0]].reshape(1,self.n_hidden[0])
			# Weights for the connections from the inputs to the hidden nodes
			weights1_slice = len(inputs)*self.n_hidden[0] + self.n_hidden[0]
			weights1 = controller[self.n_hidden[0]:weights1_slice].reshape((len(inputs),self.n_hidden[0]))

			# Outputs activation first layer.
			output1 = sigmoid_activation(inputs.dot(weights1) + bias1)

			# Preparing the weights and biases from the controller of layer 2
			bias2 = controller[weights1_slice:weights1_slice + 5].reshape(1,5)
			weights2 = controller[weights1_slice + 5:].reshape((self.n_hidden[0],5))

			# Outputting activated second layer. Each entry in the output is an action
			output = sigmoid_activation(output1.dot(weights2)+ bias2)[0]
		else:
			bias = controller[:5].reshape(1, 5)
			weights = controller[5:].reshape((len(inputs), 5))

			output = sigmoid_activation(inputs.dot(weights) + bias)[0]

		# takes decisions about sprite actions
		if output[0] > 0.5:
			attack1 = 1
		else:
			attack1 = 0

		if output[1] > 0.5:
			attack2 = 1
		else:
			attack2 = 0

		if output[2] > 0.5:
			attack3 = 1
		else:
			attack3 = 0

		if output[3] > 0.5:
			attack4 = 1
		else:
			attack4 = 0

		return [attack1, attack2, attack3, attack4]

x = controller_automata()
x.create_random_rulebook()
print(len(x.rulebook.keys()))
out = x.control([321.,12.,512.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,1.,0.,0.,0.,0.,0.])
print(out)