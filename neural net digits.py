import sys
import numpy as np
import math

SIZE = 28 #number of pixels for input
EPOCH = 30 
NBATCHES = 20
RATE = 3
HIDDEN_LAYERS = 1

MEAN = 0
VARIANCE = 1


class NeuralNetLayer:
	def __init__(self, nInputs, nNeurons):
		#randomly init weights for every neuron
		self.hiddenWeights = 0.1 *np.random.randn(nInputs, nNeurons)
		#randomly init the bias for each neuron
		self.bias = 0.1 * np.random.randn(1, nNeurons)

	#Calculates 1 forward pass through a layer
	def forward(self, curLayer):
		self.output = np.dot(curLayer, self.hiddenWeights) + self.bias
		for num in self.output:
			for i in num:
				print(i, end=' ')
				i = self.sigmoid(i)
				print(i)



	def sigmoid(self, x):
		return 1/(1+math.exp(-x))


	#prints the NN's guess, and the answer.
	def answer(self, train2):
		#for each numbers output in the training set
		for num, index in zip(self.output, train2):
			m = 0
			#for each output in that specific num's output.
			for i in range(len(num)):
				if num[i] > num[m]:
					m = i
			print("Guess: ", m, end=' ')
			print("Actual: ", index)


	def cost(self, train2):
		cost = 0
		for num, index in zip(self.output, train2):
			#for each output in that specific num's output.
			for i in range(len(num)):
				if i == index:
					cost += (num[i] - 1)**2
				else:
					cost += (num[i] - 0)**2
		return cost/len(self.output)


def main():
	nInput = int(sys.argv[1]) #number of neurons in the input layer
	nHidden = int(sys.argv[2])#number of neurons in the hidden layer
	nOutput = int(sys.argv[3]) #number of neurons in the output layer 
	#test #the test set
	#predict #Predicted labels for the test set

	#training set
	train1 = np.loadtxt('TrainDigitX.csv.gz', dtype=float, delimiter=',') 
	#labels associated with the training set
	train2 = np.loadtxt('TrainDigitY.csv.gz', dtype=float)


	#init first hidden layer with size of input (28*28), and number of neurons specified in arguments.
	hiddenLayer1 = NeuralNetLayer(nInput, nHidden)
	#init output layer with n neurons has input, and 10 outputs (0,9)
	outputLayer = NeuralNetLayer(nHidden, nOutput)
	
	#input training set 1 into first hidden layer.
	hiddenLayer1.forward(train1)
	#input the output of the first hidden layer into the output layer.
	outputLayer.forward(hiddenLayer1.output)
	print(outputLayer.cost(train2))

	"""
	for i in outputLayer.output:
		for j in i:
			print(round(j,2), end=' ')
		print("\n")
	
	"""

if __name__ == "__main__":
	main()