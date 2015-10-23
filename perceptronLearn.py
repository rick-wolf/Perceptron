"""
trains a neural network and uses stratified cross validation

usage: python perceptron.py train.arff n l e
n: number of folds for cross validation
l: learning rate
e: number of training epochs

Rick Wolf
"""

import sys, math, random
from Dataset import Dataset



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
	numPosPerSet = len(allPos)/n
	numNegPerSet = len(allNeg)/n

	# make each separate fold
	folds = []
	for i in range(n):	
		tmp = []
		# add positive values
		for j in range(numPosPerSet):
			ind = random.randint(0,len(allPos)-1)
			tmp.append(allPos.pop(ind))
		
		# add negative values
		for j in range(numNegPerSet):
			ind = random.randint(0,len(allNeg)-1)
			tmp.append(allNeg.pop(ind))

		folds.append(tmp)










if __name__ == "__main__":
	main(sys.argv[:])




