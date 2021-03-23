T, F = True, False
## Define probability node takes on value given evidence
def P(node, value, evidence={}):
    '''The probability distribution for P(var | evidence), 
    when all parent variables are known (in evidence)'''
    if len(node.parents) == 1:
        # only one parent
        row = evidence[node.parents[0]]
    else:
        # multiple parents
        row = tuple(evidence[parent] for parent in node.parents)

    return node.cpt[row] if value else 1 - node.cpt[row]

## Define BayesNode object
class BayesNode:

    def __init__(self, name, parents, values, cpt):
        if isinstance(parents, str):
            parents = parents.split()

        if len(parents) == 0:
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
        self.children = []

    def __repr__(self):
        return repr((self.variable, ' '.join(self.parents)))

class BayesNet:
    '''Bayesian network containing only boolean-variable nodes.'''

    def __init__(self, nodes):
        '''Initialize the Bayes net by:
        '''
        self.nodes = nodes
        self.variables = [node.variable for node in nodes]
        self.next_node = []
        self.next_variable = []

    def find_node(self, var):
        '''Find and return the BayesNode in the net with name `var`
        Return an exception if the requested variable does not exist.
        '''
        for i in self.nodes:
            if (i.variable == var):
                return i

        return Exception

    def add(self, node):
        '''This function adds a new BayesNode to the BayesNet.
        '''
        assert node.variable not in self.variables
        assert all((parent in self.variables) for parent in node.parents)
        self.nodes.append(node)
        self.variables.append(node.variable)
        for i in node.parents:
            i.children.append(node)

    def find_values(self, var):
        '''Return the set of possible values associated with
        the variable `var`
        '''
        result = []
        for i in self.nodes:
            if i.variable == var:
                for val in i.values:
                    result.append(val)
        return result

    # Do not need to modify the below function
    def __repr__(self):
        return 'BayesNet({})'.format(self.nodes)

import numpy as np
class PDF_discrete:
    '''Define a discrete probability distribution function.'''

    def __init__(self, varname='?', freqs=None):
        '''Create a dictionary of values - frequency pairs,
        then normalize the distribution to sum to 1.'''
        self.prob = {}
        self.varname = varname
        self.values = []
        if freqs:
            for (v, p) in freqs.items():
                self[v] = p
        self.normalize()

    def __getitem__(self, value):
        '''Given a value, return P[value]'''
        try:
            return self.prob[value]
        except KeyError:
            return 0

    def __setitem__(self, value, p):
        '''Set P[value] = p, input argument if '''
        if value not in self.values:
            self.values.append(value)
        self.prob[value] = p

    def normalize(self):
        '''Normalize the probability distribution and return it.
        If the sum of PDF values is 0, then return a 0
        '''

        total = sum(self.prob.values())
        if not np.isclose(total, 0.0):
            for value in self.prob:
                self.prob[value] /= total
        return self


def extend(s, var, val):
    """Copy the substitution s and extend it by setting var to val; return copy."""
    s2 = s.copy()
    s2[var] = val
    return s2


def get_prob(X, e, bn):
    '''Return the conditional probability distribution of variable X
    given evidence e
    Return normalized instance of PDF_discrete
    '''
    temp = bn.find_node(X)
    temp_val = bn.find_values(X)
    result ={}
    if(e==None or e==''):
        if temp.parents==[]:
            for i in temp_val:
                result[i] = P(temp,i)
            return result
        else:
            if isinstance(temp.parents, str):
                parent = get_prob(temp.parents,None,bn)
                for i in temp_val:
                    result[i] = 0
                    for j in temp_val:
                        parent_dic = {}
                        parent_dic[temp.parent] = j
                        result[i]+=parent[j]*P(temp,i,parent_dic)
            else:
                for val in temp_val:
                    result[val] = 0
                    for i in temp.cpt.keys():
                        parent_dic = {}
                        for parent,parent_val in zip(temp.parents,i):
                            parent_dic[parent] = parent_val
                        parent_v = P(temp, val, parent_dic)
                        for p,v in parent_dic.items():
                            parent_v = parent_v*get_prob(p,None,bn)[v]
                        result[val]+=parent_v

    else:
        #have evidence

        pass
    return PDF_discrete(freqs=result).prob



# Define the nodes from the problem statement
Sm = BayesNode('Sm', '', [T,F], 0.2)
ME = BayesNode('ME', '', [T,F], 0.5)
HBP = BayesNode('HBP', ['Sm', 'ME'], [T,F], {(T, T): 0.6, (T, F): 0.72, (F, T): 0.33, (F, F): 0.51})
Ath = BayesNode('Ath', '', [T,F], 0.53)
FH = BayesNode('FH', '', [T,F], 0.15)
HD = BayesNode('HD', ['Ath', 'HBP', 'FH'], [T,F],
               {(T, T, T): 0.92, (T, T, F): 0.91, (T, F, T): 0.81, (T, F, F): 0.77,
                (F, T, T): 0.75, (F, T, F): 0.69, (F, F, T): 0.38, (F, F, F): 0.23})
Ang = BayesNode('Ang', 'HD', [T,F], {T: 0.85, F: 0.4})
Rapid = BayesNode('Rapid', 'HD', [T,F], {T: 0.99, F: 0.3})

# Create a Bayes net with those nodes and connections
bnHeart = BayesNet([Sm, ME, HBP, Ath, FH, HD, Ang, Rapid])


a1 = get_prob('FH',None,bnHeart)[T]


# a2 = None
# # temp_a2 = PDF_discrete('HD',bnHeart.find_node('Ang').cpt)
#
# a2 = get_prob('Ang',['HD'],bnHeart)


# a3 = None
# # YOUR CODE HERE
# raise NotImplementedError()
#
a4 = get_prob('HD',None,bnHeart)[T]

assert (np.isclose(a4, 0.661, atol=0.002)), "P(HD) = 0.662 and your code returned %f" % a4

# a5 = None
# # YOUR CODE HERE
# raise NotImplementedError()
#
# a6 = None
# # YOUR CODE HERE
# raise NotImplementedError()
#
# a7 = None
# # YOUR CODE HERE
# raise NotImplementedError()