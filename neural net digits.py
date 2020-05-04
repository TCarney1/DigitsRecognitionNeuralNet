import sys
import numpy as np
import math

SIZE = 28 #number of pixels for input
EPOCHS = 30 
BATCHSIZE = 20
RATE = 3
HIDDEN_LAYERS = 1

MEAN = 0
VARIANCE = 1
np.random.seed(0)


class NeuralNet:
	def __init__(self, sizes):
		#randomly init weights for every neuron
		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
		#randomly init the bias for each neuron
		self.biases = [np.random.randn(y,) for y in sizes[1:]]

	# returns the cost and activations of a single number through the NN
	def forward(self, numInformation, answer):
		for bias, weight in zip(self.biases, self.weights):
			numInformation = self.sigmoid(np.dot(numInformation, weight.T) + bias)
		return (self.cost(numInformation, answer), numInformation)


	def backward(self, x, y):
		pass


	def cost_derivative(self, output_activations, y):
		return (output_activations-y)
		

	def sigmoid_prime(self, z):
		return self.sigmoid(z)*(1-self.sigmoid(z))

        
	def sigmoid(self, x):
		return 1.0/(1.0+np.exp(-x))


	#Returns the cost for one number (10 activations)
	def cost(self, activations, answer):
		cost = 0
		for i in range(len(activations)):
			if i == answer:
				cost += (activations[i] - 1)**2
			else:
				cost += (activations[i] - 0)**2
		return cost


	#runs forwards and backwards for n number of epochs. This is done it batches.
	def trainNet(self, data, answers):
		batchCounter = 0
		numBatches = round(len(data)/BATCHSIZE)

		
		for epoch in range(EPOCHS):
			for i in range(numBatches):

				batchEnd = batchCounter + BATCHSIZE
				
				currentBatch = data[batchCounter:batchEnd]
				currentAnswers = answers[batchCounter:batchEnd]

				for (x, y) in zip(currentBatch, answers):
					#pass the information of 1 input through the NN
					cost, activations = self.forward(x, y)
					gradW_x = self.backward(x, y)
					sumGrad = sumGrad + gradW_x

				batchCounter += BATCHSIZE

			gradw_batch = sumGrad / BATCHESIZE

			w = w - learningRate * gradw_batch
		

	def test(self, data, answers):
		totalright = 0
		for (x, y) in data, answers:
			cost, activations = forward(x, y)
			maxIndex = 0
			for i in range(len(activations)):
				if activations[i] > activations[maxIndex]:
					maxIndex = i
			print("Guess: ", maxIndex, "Answer: ", y)
			if maxIndex == y:
				totalright += 1
		print(totalright, "/", len(answers), "\nAccuracy: ", totalright/len(answers)*100)





def main():
	nInput = int(sys.argv[1]) #number of neurons in the input layer
	nHidden = int(sys.argv[2])#number of neurons in the hidden layer
	nOutput = int(sys.argv[3]) #number of neurons in the output layer 
	#test #the test set
	#predict #Predicted labels for the test set

	sizes = [nInput, nHidden, nOutput]

	#training set
	train1 = np.loadtxt('TrainDigitX.csv.gz', dtype=float, delimiter=',') 
	#labels associated with the training set
	trainAns = np.loadtxt('TrainDigitY.csv.gz', dtype=float)

	test1 = np.loadtxt('TestDigitX.csv.gz', dtype=float, delimiter=',')
	testAns = np.loadtxt('TestDigitY.csv.gz', dtype=float, delimiter=',')

	test = [[i, j] for i, j in zip(test1, testAns)]
	train = [(i, j) for i, j in zip(train1, trainAns)]



	#init first hidden layer with size of input (28*28), and number of neurons specified in arguments.
	myNet = NeuralNet(sizes)

	myNet.trainNet(train1, trainAns)
	myNet.test(test1, testAns)




if __name__ == "__main__":
	main()