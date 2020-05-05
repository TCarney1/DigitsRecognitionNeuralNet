import sys
import numpy as np
import random


EPOCHS = 30
BATCH_SIZE = 20
LEARNING_RATE = 2
HIDDEN_LAYERS = 1

np.random.seed(0)


class NeuralNet:
	def __init__(self, sizes):
		#randomly init weights for every neuron
		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
		#randomly init the bias for each neuron
		self.biases = [np.random.randn(y,) for y in sizes[1:]]
		self.sizes = sizes

	# returns the cost and activations of a single number through the NN
	def forward(self, numInformation, answer):
		for bias, weight in zip(self.biases, self.weights):
			numInformation = self.sigmoid(np.dot(weight, numInformation) + bias)
		return (self.cost(numInformation, answer), numInformation)

	def backward(self, x, y):

		newBiases = [np.zeros(b.shape) for b in self.biases]
		newWeights = [np.zeros(w.shape) for w in self.weights]

		activation = x
		### or x
		activations = [x] 
		zs = [] 
		for bias, weight in zip(self.biases, self.weights):
			######## this z
			z = self.sigmoid(np.dot(weight, activation)+bias)
			zs.append(z)
			activation = z
			activations.append(activation)


		# how far off we where * derivative of sigmoid of the last
		delta = self.cost(activations[-1], y) * self.sigmoidPrime(zs[-1])

		newBiases[-1] = delta
		newWeights[-1] = np.dot(delta[:,None], activations[-2][None,:])


		
		
		for layer in range(2, len(self.sizes)):
			z = zs[-layer]
			sp = self.sigmoidPrime(z)
			delta = np.dot(self.weights[-layer+1].transpose(), delta) * sp
			newBiases[-layer] = delta
			newWeights[-layer] = np.dot(delta[:,None], activations[-layer-1][None,:])
		

		return (newBiases, newWeights)

	

	def sigmoidPrime(self, x):
		return x * (1 - x)
        
	def sigmoid(self, x):
		return 1.0/(1.0+np.exp(-x))

	def cost(self, activations, answer):
		cost = []
		for i in range(len(activations)):
			if i == answer:
				cost.append(activations[i] - 1)
			else:
				cost.append(activations[i] - 0)
		return cost

	"""
	#Returns list of costs for one number (10 activations)
	def cost(self, activations, answer):
		
		cost = []
		for i in range(len(activations)):
			if i == answer:
				cost.append((activations[i] - 1)**2)
			else:
				cost.append((activations[i] - 0)**2)
		print("Answer: ", answer)	
		for i, j in zip(activations, cost):
			
			print("Activations: ", i, "Cost: ", j)
		return cost
	"""

	#runs forwards and backwards for n number of epochs. This is done it batches.
	def trainNet(self, data):

		nBatches = len(data)/BATCH_SIZE
		batches = [data[i:i+BATCH_SIZE] for i in range(0, len(data), BATCH_SIZE)]
		
		for epoch in range(EPOCHS):
			random.shuffle(data)
			for batch in batches:

				totalBiasChange = np.array([np.zeros(b.shape) for b in self.biases])
				totalWeightChange = np.array([np.zeros(w.shape) for w in self.weights])

				for (x, y) in batch:
					deltaBias, deltaWeight = self.backward(x, y)

					totalBiasChange = [b + db for b, db in zip(totalBiasChange, deltaBias)]
					totalWeightChange = [w + dw for w, dw in zip(totalWeightChange, deltaWeight)]

				self.weights = [w-(LEARNING_RATE/BATCH_SIZE)*dw for w, dw in zip(self.weights, totalWeightChange)]
				self.biases = [b-(LEARNING_RATE/BATCH_SIZE)*db for b, db in zip(self.biases, totalBiasChange)]

	


		
	def test(self, data, answers):
		totalright = 0
		for (x, y) in zip(data, answers):
			cost, activations = self.forward(x, y)
			maxIndex = 0
			for i in range(len(activations)):
				if activations[i] > activations[maxIndex]:
					maxIndex = i
			print("Guess: ", maxIndex, "Answer: ", y)
			if maxIndex == y:
				totalright += 1
		print(totalright, "/", len(answers), "\nAccuracy(%): ", totalright/len(answers)*100)



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
	testAns = np.loadtxt('TestDigitY.csv.gz', dtype=float)


	testData = [(i, j) for i, j in zip(test1, testAns)]
	trainData = [(i, j) for i, j in zip(train1, trainAns)]



	#init first hidden layer with size of input (28*28), and number of neurons specified in arguments.
	myNet = NeuralNet(sizes)

	myNet.trainNet(trainData)
	myNet.test(test1, testAns)




if __name__ == "__main__":
	main()