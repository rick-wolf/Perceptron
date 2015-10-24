"""
trains a neural network and uses stratified cross validation

usage: python perceptron.py train.arff n l e
n: number of folds for cross validation
l: learning rate
e: number of training epochs

Rick Wolf
"""

import sys
import math
import random
from Dataset import Dataset
from Node import Node
from Perceptron import Perceptron



def readFile(fname):
		"""
		Takes a filename of an ARFF file from the current directory and returns
		a Dataset object
		"""

		attributes = [] # an ordered list of the attribute names
		attributeValues = {} # a dictionary with attrib name as keys
		instances = []
		labels = []

		lines = open(fname, 'r')
		for line in lines:
			if not (line.startswith("%")):
				line = line.strip("\n")
				if (line.startswith("@attribute")):
					# line is now a list of words
					line = [word.lstrip("{ '\t").rstrip("} ',\t") for word in line.split() if word != '{']
					if (line[1].lower() == "class"):
						labels = line[2:]
					else:
						attributes.append(line[1]) # the name will always be the second item
						attributeValues[line[1]] = line[2:] #the third to end of the list is the values
				elif  not (line.startswith("@")):
					line = line.split(',')
					newline = []
					for i in range(len(attributes)):
						if len(attributeValues[attributes[i]]) == 1:
							newline.append(float(line[i]))
						else:
							newline.append(line[i])
					newline.append(line[-1])
					instances.append(newline[:])
					
		return Dataset(labels, attributes, attributeValues, instances)




def main(argv):

	# for debugging
	random.seed(1)

	# Handle User input
	trainFile = ''
	n = 1         # default to training on the whole set
	l = .01
	e = 1

	if len(sys.argv) == 5:
		trainFile = sys.argv[1]
		n = int(sys.argv[2])
		l = float(sys.argv[3])
		e = int(sys.argv[4])
	else:
		sys.exit("Bad input: Please provide a test file, number of folds, \
		 learning rate, and training epochs")


	# read in dataset
	dset = readFile(trainFile)

	# split the dataset into stratified folds
	allPos = [inst for inst in dset.instances if inst[-1] == dset.labels[1]]
	allNeg = [inst for inst in dset.instances if inst[-1] == dset.labels[0]]
	numPosPerSet = int(round(float(len(allPos))/n))
	numNegPerSet = int(round(float(len(allNeg))/n))
	

	# a better way of assigning folds
	foldAssignList = [0 for i in range(len(dset.instances))]
	currentFold = 1
	foldSize = 0
	for i in range(len(dset.instances)):
		if foldSize >= numPosPerSet:
			currentFold +=1
			foldSize = 0
		if currentFold > n:
			currentFold = n
		if dset.instances[i][-1] == dset.labels[1]:
			foldAssignList[i] = currentFold
			foldSize += 1

	currentFold = 1
	foldSize = 0
	for i in range(len(dset.instances)):
		if foldSize >= numNegPerSet:
			currentFold +=1
			foldSize = 0
		if currentFold > n:
			currentFold = n
		if dset.instances[i][-1] == dset.labels[0]:
			foldAssignList[i] = currentFold
			foldSize += 1
	
	folds = []
	for i in range(n):
		folds.append([dset.instances[j] for j in range(len(dset.instances)) if foldAssignList[j]-1 == i])


	
	# for fold in folds:
	# 	posCount = 0
	# 	negCount = 0
	# 	for inst in fold:
	# 		if inst[-1]=="Mine": 
	# 			posCount+=1
	# 		else:
	# 			negCount+=1
	# 	print("FoldPos: " + str(posCount) + ", FoldNeg: " + str(negCount))


	# generate txt file for use in graphing answer to number 5
	f = open("question5.csv", "w+")
	f.write("numEpochs,Accuracy,set\n")
	
	epochList = [1,10,100,1000]
	for epoch in epochList:
		testAccs = []
		trainAccs = []
		for j in range(len(folds)):
			testFold = j
			trainSet = []
			for i in range(len(folds)):
				if i != testFold:
					trainSet.extend(folds[i])

			nnet = Perceptron(trainSet, dset.attributes, dset.labels, l, epoch, .1)

			# classify the test instances
			testCountCorrect = 0
			for inst in folds[testFold]:
				out = nnet.calculateOutput(inst)
				ind = 1 if out > .5 else 0
				if inst[-1] == dset.labels[ind]:
					testCountCorrect += 1
			testAcc = testCountCorrect / float(len(folds[testFold]))
			testAccs.append(testAcc)

			# classify the training instances
			trainCountCorrect = 0
			for inst in trainSet:
				out = nnet.calculateOutput(inst)
				ind = 1 if out > .5 else 0
				if inst[-1] == dset.labels[ind]:
					trainCountCorrect += 1
			trainAcc = trainCountCorrect / float(len(trainSet))
			trainAccs.append(trainAcc)

		# get the average accuracy across folds for  
		trainAcc = sum(trainAccs) / len(trainAccs)
		testAcc = sum(testAccs) / len(testAccs)
		f.write(str(epoch) +','+ str(trainAcc) +',train\n')
		f.write(str(epoch) +','+ str(testAcc)  +',test\n')









if __name__ == "__main__":
	main(sys.argv[:])




