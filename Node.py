import math


class Node(object):



	def __init__(self, nodeType):
		"""
		type: 0 is bias, 1 is input, 2 is output
		parentList will be a list of tuples pairing nodes and weights
		"""

		self.nodeInput = None
		self.nodeType = nodeType
		if nodeType == 2:
			self.parentList = []



	def setInput(self, nodeInput):
		self.nodeInput = nodeInput


	def addParent(self, nodeWeightPair):
		if self.nodeType == 2:
			self.parentList.append(nodeWeightPair)

	def changeWeight(self, ind, delta):
		newWeight = self.parentList[ind][0] + delta
		self.parentList[ind] = (newWeight, self.parentList[ind][1])


	def getOutput(self):
		"""
		returns the output of sigmoid function on linear weights of 
		all inputs to the node
		"""

		if self.nodeType < 2:
			return self.nodeInput
		else:
			# get linear combination of inputs
			z = 0
			for parent in self.parentList:
				z += parent[0] * parent[1].getOutput()
			return 1/(1+math.exp(-z))


		











