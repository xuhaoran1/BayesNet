import unittest
import util
import random
import copy as cp
from collections import OrderedDict
## For the sake of brevity...
T, F = True, False

## From class:
def P(var, value, evidence={}):
	'''The probability distribution for P(var | evidence),
	when all parent variables are known (in evidence)'''
	if len(var.parents)==1:
		# only one parent
		row = evidence[var.parents[0]]
	else:
		# multiple parents
		row = tuple(evidence[parent] for parent in var.parents)
	return var.cpt[row] if value else 1-var.cpt[row]

## Also from class:
class BayesNode:

	def __init__(self, name, parents, values, cpt):
		if isinstance(parents, str):
			parents = parents.split()

		if len(parents)==0:
			# if no parents, empty dict key for cpt
			cpt = {(): cpt}
		elif isinstance(cpt, dict):
			# if there is only one parent, only one tuple argument
			if cpt and isinstance(list(cpt.keys())[0], bool):
				cpt = {(v): p for v, p in cpt.items()}

		self.variable = name
		self.parents = parents
		self.cpt = cpt
		self.values = values

	def __repr__(self):
		return repr((self.variable, ' '.join(self.parents)))

class BayesNet:
	'''Bayesian network containing only boolean-variable nodes.'''

	def __init__(self, nodes):
		'''Initialize the Bayes net by adding each of the nodes,
		which should be a list BayesNode class objects ordered
		from parents to children (`top` to `bottom`, from causes
		to effects)'''

		# self.nodes = {}
		self.nodes = OrderedDict()
		for node in nodes:
			self.nodes[node.variable] = node
		# for node in self.nodes.values():
		# 	print(node)

				
	def add(self, node):
		'''Add a new BayesNode to the BayesNet. The parents should all
		already be in the net, and the variable itself should not be'''
		assert node.variable not in self.variables
		assert all((parent in self.variables) for parent in node.parents)
		
		# your code goes here...
		self.nodes[node.variable] = node
	
			
	def find_node(self, var):
		'''Find and return the BayesNode in the net with name `var`'''

		for node in self.nodes.values():
			if node.variable == var:
				return node
		return False

		
	def find_values(self, var):
		'''Return the set of possible values for variable `var`'''

		node = self.find_node(var)
		if node != False:
			values = (T, F)
			return values
		else:
			return False
		
	def __repr__(self):
		return 'BayesNet({})'.format(self.nodes)


def normalize(prob_distr):
	total = sum(prob_distr)
	if total != 0:
		return map(lambda a: a / total, prob_distr)
	else:
		return prob_distr

def get_prob(Q, e, bn):
	# Get possible values
	X = Q.variable

	nodeValues = bn.nodes[X].values

	# Initialize probabilities
	probs = {}
	for x in nodeValues:
		probs[x] = 0

	for x in nodeValues:
		e = cp.deepcopy(e)
		e[X] = x



	# Calculate through enumeration
	for x in nodeValues:
		e = cp.deepcopy(e)
		e_ = cp.deepcopy(e)
		e[X] = x
		probs[x] = enumerateVars(bn.nodes, e, bn)
		temp = enumerateVars(bn.nodes,e_,bn)
		probs[x] = probs[x]/temp
	#Normalize
	sumProbs = sum(probs.values())
	normProbs = {}
	for x in probs:
		normProbs[x] = probs[x]/sumProbs

	ret_probs = []
	ret_probs.append(normProbs[True])
	ret_probs.append(normProbs[False])
	return ret_probs

def enumerateVars(variables, e, bn):
	# Check if variables are invalid
	if len(variables)==0 or type(variables)==str:
		return 1


	# Get variable
	var = list(variables.keys())[0]

	otherVars = cp.deepcopy(variables)
	del otherVars[var]
	
	# If variable in evidence, then return it, otherwise enumerate until you can
	if var in e:
		probVar = e[var]

		if type(P(bn.nodes[var], probVar, e)) != dict:
			return P(bn.nodes[var], probVar, e)*enumerateVars(otherVars, e, bn)
		else:
			return P(bn.nodes[var], probVar, e)[probVar]*enumerateVars(otherVars, e, bn)
	else:
		p = 0
		eNew = cp.deepcopy(e)
		
		for x in bn.nodes[var].values:
			eNew[var] = x
			if type(P(bn.nodes[var], x, eNew)) != dict:
				p += enumerateVars(otherVars, eNew, bn)*P(bn.nodes[var], x, eNew)
			else:
				p += enumerateVars(otherVars, eNew, bn)*P(bn.nodes[var], x, eNew)[x]

		return p

def make_Prediction(Q,e, bn):
	'''Return most likely value for variable Q given evidence e in BayesNet bn
	 '''

	vals = get_prob(Q, e, bn)
	if vals[0] > vals[1]:
		return True
	else:
		return False


def prior_sample_n(bn, n):
	'''Return a list of samples from the BayesNet bn, where each sample is a dictionary
	Use Prior sampling (no evidence) to generate your samples, you will need
	to sample in the correct order '''
	return_dicts = []

	for i in range(0,n):
		sample_dict = {}
		for variable in bn.nodes.keys():
			generated_prob_dist = get_prob(bn.nodes[variable], sample_dict, bn)
			random_val = random.random()
			if random_val <= generated_prob_dist[0]:
				sample_dict[variable] = True
			else:
				sample_dict[variable] = False
		return_dicts.append(sample_dict)

	return return_dicts