import sys
import gzip
import csv

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

	"""
	test = gzip.open('TestDigitX.csv.gz', 'rb')
	with open(test,'r') as csvFile:
		file = csv.reader(csvFile)

		for line in file:
			print(line)
	
	"""

	myList = [[0 for i in range(SIZE)] for j in range(SIZE)] 
	i = 0
	j = 0

	with gzip.open('TestDigitX.csv.gz', 'rb') as f:
		for line in f:
			actualLine = line.decode('cp855').split(',')
			for number in actualLine:
				myList[i].append(float(number))
				j+= 1
				if j == 27 and i < 27:
					i += 1
					j = 0

			

	print(myList[0][0])


	testNet = myNeuralNet(nInput, nHidden, nOutput)


def setup():
	pass



if __name__ == "__main__":
	main()