from random import random, randint, sample, choice
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import numpy, Queue, threading, sys
from datetime import datetime
from new_inputOutput import printText

class variable: 
	def __init__(self, value):
 		self.value = value

class functionWrapper: 
	def __init__(self, function, childcount, childrennodepatterns, name):
		self.function = function
		self.childcount = childcount
		self.childrennodepatterns = childrennodepatterns
		self.name = name

class node:
	def __init__(self, type, children, functionWrapper, value=None, nodepattern=0, fitness=0):
		self.type = type
		self.children = children
		self.functionWrapper = functionWrapper
		self.variable = value
		self.depth = self.refreshdepth()
		self.size = self.refreshsize()
		self.innersize = self.refreshinnersize()
		self.nodepattern = nodepattern
		self.fitness = fitness

	def eval(self, imagePatches):
		if self.type == "imagePatch":
			return imagePatches[self.variable.value]
		if self.type == "coordinate":
			return self.variable.value
		else:
			result = [c.eval(imagePatches) for c in self.children]
			temp = self.functionWrapper.function(result)
			return temp

	def refreshdepth(self):
		if self.type == "imagePatch" or self.type == "coordinate":
			return 0
		else:
			depth = []
			for c in self.children:
				depth.append(c.refreshdepth())
			self.depth = max(depth) + 1
			return max(depth) + 1

	def refreshsize(self):
		if self.type == "imagePatch" or self.type == "coordinate":
			return 1
		else:
			size = 0
			for c in self.children:
				size += c.refreshsize()
			return size + 1

	def refreshinnersize(self):
		if self.type == "imagePatch" or self.type == "coordinate":
			return 0
		else:
			size = 0
			for c in self.children:
				size += c.refreshinnersize()
			size += 1
			self.innersize = size
			return size

	def display(self, indent=0):
		if self.type == "function":
			print ('  '*indent) + self.functionWrapper.name
		elif self.type == "imagePatch":
			print ('  '*indent) + 'imagePatch'
		elif self.type == "coordinate":
			print ('  '*indent) + str(self.variable.value)
		if self.children:
			for c in self.children:
				c.display(indent + 1)

class queueElement:
	def __init__(self, result, data, id, program):
		self.result = result
		self.data = data
		self.id = id
		self.program = program

class gpEnvironment:
	def __init__(self, funwraplist, quasiterminalwraplist, learndata, learnlabels, testdata, testlabels, 
				population=None, size=50, minimaxtype="max", maxgen=50, tournamentsize=4, 
				initmindepth=2, initmaxdepth=6, evolutionmaxdepth=12, crossrate=0.3, crossinner=0.9, crossleaf=0.1,
				subtreemutation=0.4, mutationmaxdepth=4, pointmutation=0.3, elitism=0.01, workersnumber=50):
		self.funwraplist = funwraplist
		self.quasiterminalwraplist = quasiterminalwraplist
		self.learndata = learndata
		self.learnlabels = learnlabels
		self.testdata = testdata
		self.testlabels = testlabels
		self.initmindepth = initmindepth
		self.initmaxdepth = initmaxdepth
		self.population = population or self._makepopulation(size)
		self.size = size
		self.minimaxtype = minimaxtype 
		self.maxgen = maxgen
		self.tournamentsize = tournamentsize
		self.evolutionmaxdepth = evolutionmaxdepth
		self.crossrate = crossrate
		self.crossinner = crossinner
		self.crossleaf = crossleaf
		self.subtreemutation = subtreemutation
		self.mutationmaxdepth = mutationmaxdepth
		self.pointmutation = pointmutation
		self.elitism = elitism
		self.workersnumber = workersnumber

	def _makepopulation(self, popsize):
		return [self._maketree(0, self.initmaxdepth) for i in range(0, popsize)]

	def _maketree(self, currentDepth, maxdepth, nodepattern=0):
		#nodepattern=	0	-	any
		#				1	-	quasi-terminal
		#				2	-	imagePatch [0,49]
		#				3	-	coordinate [0,4]
		if nodepattern == 0 and currentDepth == maxdepth-1:
			nodepattern = 1
		if nodepattern == 0:
			selectedfun = randint(0, len(self.funwraplist)+len(self.quasiterminalwraplist)-1)
			if selectedfun < len(self.funwraplist):
				funwrapper = self.funwraplist[selectedfun]
			else: 
				funwrapper = self.quasiterminalwraplist[selectedfun-len(self.funwraplist)]
			children = []
			for i in range(0, funwrapper.childcount):
				child = self._maketree(currentDepth+1, maxdepth, funwrapper.childrennodepatterns[i])
				children.append(child)
			return node("function", children, funwrapper)
		elif nodepattern == 1:
			selectedfun = randint(0, len(self.quasiterminalwraplist)-1)
			funwrapper = self.quasiterminalwraplist[selectedfun]
			children = []
			for i in range(0, funwrapper.childcount):
				child = self._maketree(currentDepth+1, maxdepth, funwrapper.childrennodepatterns[i])
				children.append(child)
			if funwrapper.childcount == 5:
				tempx1 = min(children[1], children[3])
				tempx2 = max(children[1], children[3])
				children[1] = tempx1
				children[3] = tempx2
				tempx1 = min(children[2], children[4])
				tempx2 = max(children[2], children[4])
				children[2] = tempx1
				children[4] = tempx2
			return node("function", children, funwrapper)
		elif nodepattern == 2:
			return node("imagePatch", None, None, variable(randint(0,49)))
		elif nodepattern == 3:
			return node("coordinate", None, None, variable(randint(0,4)))

	def crossover(self, parent1, parent2):
		parent1crossoverpoint = randint(0, parent1.innersize-1)
		parent2crossoverpoint = randint(0, parent2.innersize-1)
		newSubtree = self.findnode(parent2, parent2crossoverpoint)
		return self.replaceNode(deepcopy(parent1), parent1crossoverpoint, 0, newSubtree)

	def mutateSubtree(self, parent):
		parentcrossoverpoint = randint(0, parent.innersize-1)
		newSubtree = self._maketree(0, self.mutationmaxdepth)
		return self.replaceNode(deepcopy(parent), parentcrossoverpoint, 0, newSubtree)

	def findnode(self, tree, id, depth=0):
		if tree.innersize == 1 or tree.children[0].innersize == id:
			return deepcopy(tree)
		elif tree.children[0].innersize > id:
			return self.findnode(tree.children[0], id)
		else:
			return self.findnode(tree.children[1], id - (tree.children[0].innersize + 1))

	def replaceNode(self, tree, id, currentdepth, subtree):
		if subtree.depth + currentdepth == self.mutationmaxdepth or tree.innersize == 1 or tree.children[0].innersize == id:
			return subtree
		elif tree.children[0].innersize > id:
			return self.replaceNode(tree.children[0], id, currentdepth + 1, subtree)
		else:
			return self.replaceNode(tree.children[1], id - (tree.children[0].innersize + 1), currentdepth + 1, subtree)

	def mutateNode(self, tree, probability):
		result = deepcopy(tree)
		if random() < probability:
			if result.type == "imagePatch":
				result.value = variable(randint(0,49))
			if result.type == "coordinate":
				result.value = variable(randint(0,4))
			if result.type == "function":
				result.funwrapper = choice([x for x in self.funwraplist + self.quasiterminalwraplist if x.childcount == tree.funwrapper.childcount])
				result.children = [self.mutateNode(c, probability) for c in tree.children]
		return result

	def evolve(self):
		for i in range(0, self.maxgen):
			childlist = []
			for j in range(0, int(self.size*self.elitism)):
				childlist.append(population[j])
			for j in range(0, self.size-int(self.size*self.elitism)):
				generationMethod = random()
				if generationMethod < self.crossrate:
					parent1 = self.tournamentsel()
					while True:
 						parent2 = self.tournamentsel()
						if parent2 is not parent1:
							break 
					childlist.append(self.crossover(parent1, parent2))
				elif generationMethod < self.crossrate + self.subtreemutation:
					parent = self.tournamentsel()
					childlist.append(self.mutateSubtree(parent))
				else:
					parent = self.tournamentsel()
					childlist.append(self.mutateNode(parent, 1/parent.size))
			self.population = childlist
			self.sortpopulation()

	def sortpopulation(self):
		for i in range(0, self.size):
			self.computeFitness(self.population[i])
		self.population.sort(key=lambda x: x.fitness, reverse=True)

	def computeFitness(self, program):

		def threadWorker():
			while True:
				item = q.get()
				temp1 = self._performTLayer(item.data[item.id], item.program)
				temp2 = self._performPLayer(temp1)
				temp3 = self._performCLayer(temp2)
				item.result[item.id] = temp3
				q.task_done()

		q = Queue.Queue()

		for i in range(100):
			t=threading.Thread(target = threadWorker)
			t.daemon = True
			t.start()

		timeZero = datetime.now()
		learnFeatureMaps = range(0, len(self.learndata))
		for i in learnFeatureMaps:
			# temp1 = self._performTLayer(self.learndata[i], program)
			# temp2 = self._performPLayer(temp1)
			# temp3 = self._performCLayer(temp2)
			# learnFeatureMaps.append(temp3)
			q.put(queueElement(learnFeatureMaps, self.learndata, i, program))
		testFeatureMaps = range(0, len(self.testdata))
		for i in testFeatureMaps:
			# temp1 = self._performTLayer(self.testdata[i], program)
			# temp2 = self._performPLayer(temp1)
			# temp3 = self._performCLayer(temp2)
			# testFeatureMaps.append(temp3)
			q.put(queueElement(testFeatureMaps, self.testdata, i, program))

		q.join()
		logisticRegression = LogisticRegression()
		print "logreg"
		printText("logreg")
		logisticRegression.fit(learnFeatureMaps, self.learnlabels)
		print "fit"
		printText("fit")
		program.fitness = logisticRegression.score(testFeatureMaps, self.testlabels)
		print program.fitness
		printText(str(program.fitness))

	def _performTLayer(self, featureMaps, program):
		fmSize = len(featureMaps[0])
		fmFromLayerT = []
		for i in range(0, fmSize - 4):
			fmRow = []
			for j in range(0, fmSize - 4):
				fmRow.append(program.eval(featureMaps[:, i:i + 5, j:j + 5]))
			fmFromLayerT.append(fmRow)
		return numpy.array(fmFromLayerT)

	def _performPLayer(self, featureMap):
		fmSize = len(featureMap)
		fmFromLayerP = []
		for i in range(0, fmSize, 4):
			featureMapRow = []
			for j in range(0, fmSize, 4):
				featureMapRow.append(self._my_mean(numpy.array(featureMap[i:i + 4, j:j + 4])))
			fmFromLayerP.append(featureMapRow)
		return fmFromLayerP

	def _my_mean(self, xs):
		xs2 = [item for sublist in xs for item in sublist]
		return sum(x / len(xs2) for x in xs2)
		
	def _performCLayer(self, featureMap):
		return numpy.array(featureMap).flatten()

	def getTop5(self):
		return self.population[:5]

	def createFeatureMapsFromBest50Prog(self, imgs):
		best50progs = self.population[:50]
		newFeatureMaps = []
		if imgs == 'learn':
			for i in range(0, len(self.learndata)):
				bestFeatureMapsForImg = []
				for j in range(0, 50):
					bestFeatureMapsForImg.append(self._performTLayer(self.learndata[i], best50progs[j]))
				newFeatureMaps.append(bestFeatureMapsForImg)
		if imgs == 'test':
			for i in range(0, len(self.testdata)):
				bestFeatureMapsForImg = []
				for j in range(0, 50):
					bestFeatureMapsForImg.append(self._performTLayer(self.testdata[i], best50progs[j]))
				newFeatureMaps.append(bestFeatureMapsForImg)
		return newFeatureMaps

	def tournamentsel(self, reverseOrder=True):
		samples = sample(self.population, 4)
		samples.sort(key=lambda x: x.fitness, reverse=reverseOrder)
		return samples[0]

	def listpopulation(self):
		for i in range(0, self.size):
			self.population[i].display()





	


















































