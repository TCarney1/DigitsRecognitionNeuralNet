import sys
import numpy

SIZE = 28 #number of pixels for input



class myNeuralNet:
	def __init__(self, nInput, nHidden, nOutput):
		self.nInput = nInput
		self.nHidden = nHidden
		self.nOutput = nOutput




def main():
	nInput = int(sys.argv[1]) #number of neurons in the input layer
	nHidden = int(sys.argv[2])#number of neurons in the hidden layer
	nOutput = int(sys.argv[3]) #number of neurons in the output layer 
	#train1 #training set
	#train2 #labels associated with the training set
	#test #the test set
	#predict #Predicted labels for the test set



	myList = [[0 for i in range(SIZE)] for j in range(SIZE)] 
	x = numpy.loadtxt('TestDigitX.csv.gz', dtype=float, delimiter=',')
	print(type(x))


	testNet = myNeuralNet(nInput, nHidden, nOutput)


def setup():
	pass



if __name__ == "__main__":
	main()