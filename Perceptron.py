
import math,
from Dataset import Dataset
from Node import Node

class Perceptron(object):
	"""
	A class representing a perceptron with input units and a single output unit.

	Author: Rick Wolf
	"""


	def __init__(self, trainingSet, featureList, labels, learningRate, maxEpoch, weightDefault):

		self.trainingSet = trainingSet
		self.featureList = featureList
		self.learningRate = learningRate
		self.maxEpoch = maxEpoch
		self.weightDefault = weightDefault
		self.labels = labels

		# initialize input nodes
		self.inputNodes = []
		for i in range(len(self.featureList)):
			self.inputNodes.append(Node(1))
		self.inputNodes.append(Node(0))

		# initialize output node
		self.outputNode = Node(2)
		for i in self.inputNodes:
			self.outputNode.addParent(self.weightDefault,i)



	def calculateOutput(self, instance):
		# set all of the inputs for the instances
		for i in range(len(self.featureList)):
			self.inputNodes[i].setInput(instance[i])
		self.inputNodes[-1].setInput(1) # the bias node always gets a 1
		return self.outputNode.getOutput()






	def train(self):
		"""
		Trains the perceptron with online learning
		"""

		# epoch loop
		for i in range(len(self.maxEpoch)):

			# instance loop
			for instance in self.trainingSet:
				targetClass = self.labels.index(instance[-1])
				output = self.calculateOutput(instance)

				# node loop
				for x in range(len(self.featureList)):
					#apply perceptron learning rule
					delta = self.learningRate * (targetClass-output)*instance[x]
					self.outputNode.changeWeight(x, delta)
				# update the bias node
				delta = self.learningRate * (targetClass-output)*instance[-1]
				self.outputNode.changeWeight(-1,delta)












