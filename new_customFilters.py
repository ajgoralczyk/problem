import random, numpy
from scipy import signal

class filterEnvironment:

	def __init__(self):
		self.filterList = []
		for i in xrange(10):
			self.filterList.append(self._generateFilter("U", 5, -1.0, 1.0))
			self.filterList.append(self._generateFilter("U", 5, -5.0, 5.0))
			self.filterList.append(self._generateFilter("UD", 5, 1, 5))
			self.filterList.append(self._generateFilter("N", 5, 1.0))
			self.filterList.append(self._generateFilter("N", 5, 5.0))

	def _generateFilter(self, distribution, windowSize, value1, value2 = 1.0):
		if distribution == "U":
			return self._generateFilterWithUniformSamplingOfRealvaluedNumbers(windowSize, value1, value2)
		elif distribution == "UD":
			return self._generateFilterWithUniformSamplingOfDiscreteNumbers(windowSize, value1, value2)
		elif distribution == "N":
			return self._generateFilterWithSamplingFromNormalDistribution(windowSize, value1)
		
	def _generateFilterWithUniformSamplingOfRealvaluedNumbers(self, windowSize, value1, value2):
		filterMatrix = [] 
		for i in range(windowSize):
			filterRow = []
			for j in range(windowSize):
				filterRow.append(random.uniform(value1, value2)) # difference: open interval
			filterMatrix.append(filterRow)
		return filterMatrix

	def _generateFilterWithUniformSamplingOfDiscreteNumbers(self, windowSize, value1, value2):
		filterMatrix = [] 
		for i in range(windowSize):
			filterRow = []
			for j in range(windowSize):
				filterRow.append(random.randint(value1, value2))
			filterMatrix.append(filterRow)
		return filterMatrix

	def _generateFilterWithSamplingFromNormalDistribution(self, windowSize, value1):
		filterMatrix = [] 
		for i in range (windowSize):
			filterRow = []
			for j in range (windowSize):
				filterRow.append(random.normalvariate(0.0, value1))
			filterMatrix.append(filterRow)
		return filterMatrix

	def _applyFilterToImage(self, image, imageSize=28, windowSize=5):
		featureMapsForImage = []
		for f in self.filterList:
			convolutionResult = signal.convolve2d(image, f, 'same')
			featureMapsForImage.append(convolutionResult[windowSize/2 : imageSize-windowSize/2, windowSize/2 : imageSize-windowSize/2])
		return numpy.array(featureMapsForImage)

	def createFeatureMaps(self, images):
		featureMaps = []
		for i in images:
			featureMaps.append(self._applyFilterToImage(i))
		return featureMaps

			



