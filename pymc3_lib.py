''' Helper functions Bayes nets
'''
import pymc3 as pm
import theano
import numpy as np
import pandas

def C(p):
    '''
    Parse input to 2 valued Categorical distribution
    Eg.: Probability of event happening is 20%
    >>> C(0.2)
    [0.8, 0.2]
    '''
    return [1-p, p]


def get_prob_by_conditions(prior_trace, target, conditions=None, neg_signs=['!', '¬']):
    ''' P(target|conditions) = P(target|conditions[0], .. ,conditions[n])
    Parameters:
        prior_trace (result of pm.sample_prior_predictive())
        target (str): variable
        conditions (str list of conditions): Use "!var" for negation of variable condition
    Returns:
        conditioned probabilitys
    '''
    if conditions:
        tmp_var_name = conditions[0] if conditions[0][0] not in neg_signs else conditions[0][1:]
        conditions_mask = np.ones(len(prior_trace[tmp_var_name]), dtype=bool)
        for n,condition in enumerate(conditions):
            sign = 1
            if condition[0] in neg_signs:
                conditions[n] = condition[1:]
                sign = 0
            conditions_mask = conditions_mask & (prior_trace[conditions[n]] == sign)
        prob = prior_trace[target][ conditions_mask ].mean()
        if np.isnan(prob): prob = 0.
        return prob
    else:
        prob = prior_trace[target][:].mean()
        if np.isnan(prob): prob = 0.
        return prob

class CPT:
    def __init__(self, cpt_dict):
        self.cpt_dict = cpt_dict
        self.cpt = self.CPT_to_theanoarray(self.cpt_dict)
        self.vars = self.get_vars_from_dict()

    def get_vars_from_dict(self):
        vars = []
        tmp = self.cpt_dict.copy()
        while not isinstance(tmp, list):
            vars.append(list(tmp.keys())[0])
            tmp = tmp[list(tmp.keys())[0]]
        return vars

    def CPT_to_theanoarray(self, CPT):
        a = np.zeros([len(CPT.keys()), len(CPT[list(CPT.keys())[0]].keys()), 2])
        for n,key in enumerate(reversed(CPT)):
            for m,key2 in enumerate(reversed(CPT[key])):
                a[n,m] = np.array(CPT[key][key2])

        return theano.shared(np.asarray(a))

    def __getitem__(self, vs):
        v1, v2 = vs
        v1 = self.vars[0] if v1 else '!'+self.vars[0]
        v2 = self.vars[1] if v2 else '!'+self.vars[1]
        return f'P({v1},{v2}) = {self.cpt_dict[v1][v2][1]}'

    def pymc_model_fn(self, v1, v2):
        return self.cpt[v1][v2]

def get_prob_by_conditions_cat(prior_trace, target, conditions=None):
    target, target_cat = target.split('=')
    target_cat = int(target_cat)

    # get n categories
    if target+'_n' in prior_trace.keys():
        target_n_cat = prior_trace[target+'_n']
    else:
        target_n_cat = max(prior_trace[target])+1

    sums = []
    for s in range(target_n_cat):
        sums.append( len(prior_trace[target][prior_trace[target]==s]) )
    total = sum(sums)

    if conditions:
        tmp_var_name = conditions[0].split('=')[0]
        conditions_mask = np.ones(len(prior_trace[tmp_var_name]), dtype=bool)
        for n,condition in enumerate(conditions):
            condition, id = condition.split('=')
            id = int(id)
            conditions_mask = conditions_mask & (prior_trace[condition] == id)

        satisfied_sum = sum(prior_trace[target][ conditions_mask ]==target_cat)
        prob = satisfied_sum / len(prior_trace[target][ conditions_mask ])
        if np.isnan(prob): prob = 0.
        return prob
    else:
        return sums[target_cat]/total

def switch3(id, mt):
    ''' Manual 3-variate categorical '''
    a = pm.math.switch(pm.math.eq(id, 0),
            mt[0],
            pm.math.switch(pm.math.eq(id, 1),
              mt[1],
              mt[2])
            )
    return a

def switch4(id, mt):
    ''' Manual 4-variate categorical '''
    a = pm.math.switch(pm.math.eq(id, 0),
            mt[0],
            pm.math.switch(pm.math.eq(id, 1),
              mt[1],
              pm.math.switch(pm.math.eq(id, 2),
                mt[2],
                mt[3])
              )
            )
    return a

def switchn(id, mt):
    return mt[id]

class CPTMapping:
    def __init__(self, cpt, input=None, output=None):
        #self.cpt_dict = cpt_dict
        self.cpt = theano.shared(np.asarray(cpt))
        self.input = input
        self.output = output

    def __call__(self):
        return pandas.DataFrame(self.cpt.eval(), index=self.output, columns=self.input)

    def __getitem__(self, index):
        return self.cpt[index]

class CPTCat:
    def __init__(self, cpt_dict):
        self.cpt_dict = cpt_dict
        self.vars = self.get_vars_from_dict()
        self.n_vars, self.out_n_vars = self.get_n_vars_from_dict()
        self.cpt = self.CPT_to_theanoarray(self.cpt_dict)

    def get_vars_from_dict(self):
        vars = []
        tmp = self.cpt_dict.copy()
        while not isinstance(tmp, list):
            vars.append(list(tmp.keys())[0].split('=')[0])
            tmp = tmp[list(tmp.keys())[0]]
        return vars

    def get_n_vars_from_dict(self):
        tmp = self.cpt_dict.copy()
        n_vars = []
        for n,var in enumerate(self.vars):
            n_vars.append( len(tmp.keys()) )
            tmp = tmp[self.vars[n]+"=0"]
        out_n_vars = len(tmp)
        return n_vars, out_n_vars

    def CPT_to_theanoarray(self, CPT):
        a = np.zeros([*self.n_vars, self.out_n_vars])

        for n,key in enumerate(CPT):
            for m,key2 in enumerate(CPT[key]):
                a[n,m] = np.array(CPT[key][key2])

        return theano.shared(np.asarray(a))

    def pymc_model_fn(self, v1, v2):
        return self.cpt[v1][v2]

    def fn(self, v1, v2):
        return self.cpt[v1][v2]



def graph_replace_text(graphvizfig, varname, text):
    for n,part in enumerate(graphvizfig.body):
        if varname+' ' in part:
            graphvizfig.body[n] = part.replace('Categorical', text)
            return graphvizfig
    raise Exception("var not found!")
