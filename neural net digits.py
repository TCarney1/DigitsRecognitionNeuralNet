import sys
import numpy as np
import random
import matplotlib.pyplot as plt





np.random.seed(0)


class NeuralNet:
	def __init__(self, sizes, learningRate, batchSize, epochs):
		#randomly init weights for every neuron (Sizes is used to inti the right shape.)
		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
		#randomly init the bias for each neuron (Sizes is used to inti the right shape.)
		self.biases = [np.random.randn(y,) for y in sizes[1:]]
		self.sizes = sizes
		self.learningRate = learningRate
		self.batchSize = batchSize
		self.epochs = epochs


	# returns the prediction from 1 forward pass through the NN
	def forward(self, nodeValues):
		for bias, weight in zip(self.biases, self.weights):
			nodeValues = self.sigmoid(np.dot(weight, nodeValues) + bias)

		maxIndex = 0
		for i in range(len(nodeValues)):
			if nodeValues[i] > nodeValues[maxIndex]:
				maxIndex = i

		return maxIndex


	#Backwards propagation
	def backward(self, inputValues, answer):

		# init a lisdirt of np arrays that is the same shape as self.weights and self.biases
		# this will be used to store the slight changes we want to make for the current
		# number.
		changeInBiases = [np.zeros(b.shape) for b in self.biases]
		changeInWeights = [np.zeros(w.shape) for w in self.weights]

		# layerValues will store the values at each layer. 
		layerValues = [inputValues] 


		### This is a standard forward pass ###
		### but we are storing the values on the way through ###
		currentLayerValues = inputValues
		for bias, weight in zip(self.biases, self.weights):
			# find the output of each Neuron
			output = self.sigmoid(np.dot(weight, currentLayerValues)+bias)
			currentLayerValues = output
			# store the outputs at each layer
			layerValues.append(currentLayerValues)

		### End of forward pass ###

		# find how far off the last layer is and how much we wanna change it
		correction = self.cost(layerValues[-1], answer) * self.sigmoidPrime(layerValues[-1])

		# apply the corrections to the last layer's biases and weights
		changeInBiases[-1] = correction
		changeInWeights[-1] = np.dot(correction[:,None], layerValues[-2][None,:])

		# apply the corrections to the rest of the NN. (loops backwards)
		for layer in range(2, len(self.sizes)):
			correction = np.dot(self.weights[-layer+1].transpose(), correction) * self.sigmoidPrime(layerValues[-layer])
			changeInBiases[-layer] = correction
			changeInWeights[-layer] = np.dot(correction[:,None], layerValues[-layer-1][None,:])
		

		return (changeInBiases, changeInWeights)

	

	# This the x's here dont have to be 'sigmoided' as the x that is passed in 
	# will already be 'sigmoided'.
	def sigmoidPrime(self, x):
		return x * (1.0 - x)

       
    # standard sigmoid function
	def sigmoid(self, x):
		return 1.0/(1.0+np.exp(-x))

	#Returns list of costs for one number (10 activations)
	def cost(self, activations, answer):
		cost = [ j - i for i, j in zip(answer, activations)]
		return cost

	

	#runs forwards and backwards for n number of epochs. This is done it batches.
	def trainNet(self, data):
		#Breaking up training data into batches of side self.batchSize
		batches = [data[i:i+self.batchSize] for i in range(0, len(data), self.batchSize)]
		accuracy = []
		
		for epoch in range(self.epochs):
			random.shuffle(data)
			for batch in batches:

				# copy the shape of bias and weight  as we will store the change in 
				# bias and weight here. This will make it easy to add the change in
				# bias and weight down the road.
				totalBiasChange = np.array([np.zeros(b.shape) for b in self.biases])
				totalWeightChange = np.array([np.zeros(w.shape) for w in self.weights])

				#Individualy pass each number in the data set through the backward propergation.
				# x is a input data. y is answer.
				for (x, y) in batch:

					# we move forwards through the layers inside of backwards.
					# This way we can store values throughout the pass forwards.
					deltaBias, deltaWeight = self.backward(x, y)

					# add the change in biases and weights for that one number, to the total changes.
					totalBiasChange = [b + db for b, db in zip(totalBiasChange, deltaBias)]
					totalWeightChange = [w + dw for w, dw in zip(totalWeightChange, deltaWeight)]

				# update the weights and biases slightly. (Gradient decent)
				self.weights = [w-(self.learningRate/self.batchSize)*dw for w, dw in zip(self.weights, totalWeightChange)]
				self.biases = [b-(self.learningRate/self.batchSize)*db for b, db in zip(self.biases, totalBiasChange)]
			accuracy.append(self.test(data))

		return accuracy

	
	# Performs 1 forward pass on NN with test data. Prints: "Correct guesses/Total guesses, Accuracy: %"
	def test(self, data):
		totalright = 0
		for (x, y) in data:
			prediction = self.forward(x)
			if y[prediction] == 1:
				totalright += 1
		return totalright/len(data)*100


	def output(self, data, outFile):
		predictions = [self.forward(d) for d in data]
		np.savetxt(outFile, predictions, delimiter=', ')




def main():
	nInput = int(sys.argv[1]) # Number of neurons in the input layer
	nHidden = int(sys.argv[2]) # Number of neurons in the hidden layer
	nOutput = int(sys.argv[3]) # Number of neurons in the output layer 
	trainFile = sys.argv[4] # Name of the file with the training data
	trainLabel = sys.argv[5] # Name of the file with the answers to the training data
	testFile = sys.argv[6] # Name of the file with the test data
	predictionFile = sys.argv[7] # output file for the predictions.


	# Structure of NN. Index's are layers, values are nNodes in a layer.
	sizes = [nInput, nHidden, nOutput]

	# Load training information and answers
	trainingData = np.loadtxt(trainFile, dtype=float, delimiter=',') 
	trainingAnswers = np.loadtxt(trainLabel, dtype=float)
	trainingAnswers = formatAnswers(trainingAnswers)
	# Load testing information and answers
	testData = np.loadtxt(testFile, dtype=float, delimiter=',')
	testAns = np.loadtxt('TestDigitY.csv.gz', dtype=float)
	testAns = formatAnswers(testAns)

	# Form information and answers into training and testing sets.
	testingSet = [(i, j) for i, j in zip(testData, testAns)]
	trainingSet = [(i, j) for i, j in zip(trainingData, trainingAnswers)]

	#  Args: sizes, learningRate, batchSize, epochs
	NN1 = NeuralNet(sizes, 3, 20, 50)
	NN2 = NeuralNet(sizes, 1, 10, 50)
	NN3 = NeuralNet(sizes, 3, 20, 50)
	NN4 = NeuralNet(sizes, 10, 50, 50)
	NN5 = NeuralNet(sizes, 1, 50, 50)

	print(NN1.test(testingSet))


	allAcc = []
	
	# trains NN with back propergation
	allAcc.append(NN1.trainNet(trainingSet))
	allAcc.append(NN2.trainNet(trainingSet))
	allAcc.append(NN3.trainNet(trainingSet))
	allAcc.append(NN4.trainNet(trainingSet))
	allAcc.append(NN5.trainNet(trainingSet))

	displayAccuracy(allAcc)


	#myNet.output(testData, predictionFile)

def displayAccuracy(accuracy):
	e = np.arange(0, 50, 1)

	fig, ax = plt.subplots()

	ax.plot(e, accuracy[0], label='High LR low BS' )
	ax.plot(e, accuracy[0], label='High LR Low BS' )
	ax.plot(e, accuracy[1], label='Low LR Low BS' )
	ax.plot(e, accuracy[2], label='Medium Everything')
	ax.plot(e, accuracy[3], label='High LR High BS' )
	ax.plot(e, accuracy[4], label='Low LR High BS' )

	ax.set(xlabel="Epoch (n)", ylabel="Accuracy (%)", title="Various")
	plt.legend()
	ax.grid()
	fig.savefig("Various.png")
	plt.show()


def formatAnswers(trainingAnswers):
	fAnswers = []

	for a in trainingAnswers:
		answer = []
		for i in range(10):
			if i == a:
				answer.append(1)
			else:
				answer.append(0)
		fAnswers.append(answer)
	return fAnswers


if __name__ == "__main__":
	main()