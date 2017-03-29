import new_inputOutput, sys, random, numpy, math, scipy
from new_customFilters import filterEnvironment
from gp import functionWrapper, gpEnvironment
from datetime import datetime

def noInf(f):
	def wrapped(*args, **kwargs):
		res = f(*args, **kwargs)
		if res == float('inf'):
			return sys.float_info.max
		if res == float('-inf'):
			return sys.float_info.min
		return res
	return wrapped

#functions
@noInf
def add(ValuesList):
	return ValuesList[0] + ValuesList[1]

@noInf
def subtract(ValuesList):
	return ValuesList[0] - ValuesList[1]

@noInf
def multiply(ValuesList): #TODO: overflow encountered
	return ValuesList[0] * ValuesList[1]

@noInf
def divide(ValuesList): #TODO: overflow encountered
	if (ValuesList[1] == 0): 
		return ValuesList[0] / 0.01
	return ValuesList[0] / ValuesList[1]

@noInf
def exp(ValuesList):
	if (ValuesList[0] > 709):
		return sys.float_info.max
	else:
		return math.exp(ValuesList[0])

@noInf
def log(ValuesList): 
	if (ValuesList[0] <= 0):
		return math.log(0.01)
	return math.log(ValuesList[0])

@noInf
def sqrt(ValuesList): 
	if (ValuesList[0] <= 0):
		return math.sqrt(0.01)
	return math.sqrt(ValuesList[0])

@noInf
def sin(ValuesList):
	return math.sin(ValuesList[0])

@noInf
def tanh(ValuesList):
	return math.tanh(ValuesList[0])

@noInf
def intensity(ValuesList):
	temp = ValuesList[0][ValuesList[1], ValuesList[2]]
	return temp

@noInf
def minFromPatch(ValuesList):
	temp = numpy.amin(ValuesList[0][min(ValuesList[1],ValuesList[3]):max(ValuesList[1],ValuesList[3])+1,min(ValuesList[2],ValuesList[4]):max(ValuesList[2],ValuesList[4])+1])
	return temp

@noInf
def maxFromPatch(ValuesList):
	temp = numpy.amax(ValuesList[0][min(ValuesList[1],ValuesList[3]):max(ValuesList[1],ValuesList[3])+1,min(ValuesList[2],ValuesList[4]):max(ValuesList[2],ValuesList[4])+1])
	return temp

@noInf
def standartDeviation(ValuesList):
	temp = numpy.std(ValuesList[0][min(ValuesList[1],ValuesList[3]):max(ValuesList[1],ValuesList[3])+1,min(ValuesList[2],ValuesList[4]):max(ValuesList[2],ValuesList[4])+1])
	return temp

@noInf
def mean(ValuesList):
	temp = numpy.mean(ValuesList[0][min(ValuesList[1],ValuesList[3]):max(ValuesList[1],ValuesList[3])+1,min(ValuesList[2],ValuesList[4]):max(ValuesList[2],ValuesList[4])+1])
	return temp

@noInf
def entropy(ValuesList):
	array = ValuesList[0][min(ValuesList[1],ValuesList[3]):max(ValuesList[1],ValuesList[3])+1,min(ValuesList[2],ValuesList[4]):max(ValuesList[2],ValuesList[4])+1].flatten()
	temp = scipy.stats.entropy(array)
	return temp

# SETUP
addwrapper = functionWrapper(add, 2, [0, 0],"Add")
subwrapper = functionWrapper(subtract, 2, [0, 0], "Sub")
mulwrapper = functionWrapper(multiply, 2, [0, 0], "Mul")
divwrapper = functionWrapper(divide, 2, [0, 0], "Div")
expwrapper = functionWrapper(exp, 1, [0], "Exp")
logwrapper = functionWrapper(log, 1, [0], "Log")
sqrtwrapper = functionWrapper(sqrt, 1, [0], "Sqrt")
sinwrapper = functionWrapper(sin, 1, [0], "Sin")
tanhwrapper = functionWrapper(tanh, 1, [0], "Tanh")
#quasi-terminal functions
intwrapper = functionWrapper(intensity, 3, [2, 3, 3], "Intensity")
minwrapper = functionWrapper(minFromPatch, 5, [2, 3, 3, 3, 3], "Min")
maxwrapper = functionWrapper(maxFromPatch, 5, [2, 3, 3, 3, 3], "Max")
stdwrapper = functionWrapper(standartDeviation, 5, [2, 3, 3, 3, 3], "Std")
meanwrapper = functionWrapper(mean, 5, [2, 3, 3, 3, 3], "Mean")
entwrapper = functionWrapper(entropy, 5, [2, 3, 3, 3, 3], "Ent")


def main(id, x, y):

	timeZero = datetime.now()
	# 1. generate filters & upload images
	trainingImages, trainingLabels = new_inputOutput.readMnist('training')
	xsample = random.sample(xrange(len(trainingImages)), x)
	trainingImages = [trainingImages[i] for i in xsample]
	trainingLabels = [trainingLabels[i] for i in xsample]
	testingImages, testingLabels = new_inputOutput.readMnist('testing')
	ysample = random.sample(xrange(len(testingImages)), y)
	testingImages = [testingImages[i] for i in ysample]
	testingLabels = [testingLabels[i] for i in ysample]
	filterEnv = filterEnvironment()
	
	print "imgs, filters "

	trainingImgFeatureMaps = filterEnv.createFeatureMaps(trainingImages)
	testingImgFeatureMaps = filterEnv.createFeatureMaps(testingImages)

	# funwraplist, quasiterminalwraplist, learndata, learnlabels, testdata, testlabels, \
	# population=None, size=500, minimaxtype="min", maxgen=50, tournamentsize=4, \
	# initmindepth=2, initmaxdepth=6, evolutionmaxdepth=12, crossrate=0.3, crossinner=0.9, \
	# crossleaf=0.1, subtreemutation=0.4, mutationmaxdepth=4, pointmutation=0.3, elitism=0.01):

	# 2. init environment
	gpEnv = gpEnvironment([addwrapper, subwrapper, mulwrapper, divwrapper, expwrapper, logwrapper,
				sqrtwrapper, sinwrapper, tanhwrapper], [intwrapper, minwrapper, maxwrapper,
				stdwrapper, meanwrapper, entwrapper], trainingImgFeatureMaps, numpy.array(trainingLabels).flatten(),
				testingImgFeatureMaps, numpy.array(testingLabels).flatten(), None, 50, "min", 5)

	print "gp init"

	gpEnv.sortpopulation()

	print "gp sorted"

	new_inputOutput.printDataToFile(id, '0 evolutions - best 5: ', gpEnv.getTop5())

	# 3. evolve gp for 50 gens
	gpEnv.evolve()

	print "gp evolved"

	new_inputOutput.printDataToFile(id, '50 evolutions - best 5: ', gpEnv.getTop5())

	newTrainingImgFeatureMaps = gpEnv.createFeatureMapsFromBest50Prog('learn')
	newTestingImgFeatureMaps = gpEnv.createFeatureMapsFromBest50Prog('test')
	evolvedPopulation = gpEnv.population # TODO: copy? pointers?

	# 4B.
	doubleStageGpEnv = gpEnvironment([addwrapper, subwrapper, mulwrapper, divwrapper, expwrapper, logwrapper,
				sqrtwrapper, sinwrapper, tanhwrapper], [intwrapper, minwrapper, maxwrapper,
				stdwrapper, meanwrapper, entwrapper], newTrainingImgFeatureMaps, numpy.array(trainingLabels).flatten(),
				newTestingImgFeatureMaps, numpy.array(testingLabels).flatten(), evolvedPopulation, 50, "min", 5)
	new_inputOutput.printDataToFile(id, 'DST results - best 5: ', doubleStageGpEnv.getTop5())
	
	print "dst evolved"

	# 4A. 
	gpEnv.evolve()
	new_inputOutput.printDataToFile(id, 'SST results - best 5: ', gpEnv.getTop5())

	print "sst evolved"
	print "total time", datetime.now() - timeZero


if __name__ == "__main__":
	if len(sys.argv) > 1:
		if sys.argv[1] == '0':
			main(0, 200, 200)
		elif sys.argv[1] == '1':
			main(1, 500, 500)
		elif sys.argv[1] == '2':
			main(2, 1000, 500)
		elif sys.argv[1] == '3':
			main(3, 2000, 1000)
		elif sys.argv[1] == '4':
			main(4, 4000, 2000)
		elif sys.argv[1] == '5':
			main(5, 5000, 2000)
		elif sys.argv[1] == '6':
			main(6, 10000, 5000)
		elif sys.argv[1] == '7':
			main(7, 15000, 5000)
		elif sys.argv[1] == '8':
			main(8, 20000, 10000)
		elif sys.argv[1] == '9':
			main(9, 30000, 10000)
	else:
		main(10, 100, 100)

