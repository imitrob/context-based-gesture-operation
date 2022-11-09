
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
    def __init__(self, cpt_dict, vars=None, n_vars=None, out_n_vars=None):
        '''
        Parameters:
            cpt_dict (dict): Input always as: {'var1=0': {'var2=0': v1, 'var2=1': v2}, 'var2=1': {'var2=0': v3, 'var2=1': v4}}
            vars (Str []): Variable names, Only if incomplete cpt_dict
            n_vars (Int []): Variable categorical sizes, Only if incomplete cpt_dict
            out_n_vars (Int): Number of output variables C()=2, Only if incomplete cpt_dict
        '''
        self.cpt_dict = cpt_dict
        self.vars = vars
        if vars is None: self.vars = self.get_vars_from_dict()
        self.n_vars, self.out_n_vars = n_vars, out_n_vars
        if not self.n_vars or not self.out_n_vars: self.n_vars, self.out_n_vars = self.get_n_vars_from_dict()
        self.init()

    def __call__(self):
        ''' Plots CPT Table '''
        rows = np.prod(self.n_vars)

        if self.out_n_vars == 2:
            columns = len(self.n_vars) + self.out_n_vars - 1
            t = np.zeros([rows, columns])
            for row in range(rows):
                decoded = decode(row,dims=self.n_vars)
                t[row,-1] = self.cpt_eval.item(tuple([*decoded, 1]))
                t[row,:-1] = np.array(decoded, dtype=int)
            return pandas.DataFrame(t, columns=[*self.vars, 'p'])
        else:
            columns = len(self.n_vars) + self.out_n_vars
            t = np.zeros([rows, columns])
            for row in range(rows):
                decoded = decode(row,dims=self.n_vars)
                t[row,-self.out_n_vars:] = self.cpt_eval[(tuple([*decoded]))]
                t[row,:-self.out_n_vars] = np.array(decoded, dtype=int)

            p_out = list(range(self.out_n_vars))
            p_out = ['p_out='+str(p) for p in p_out]
            return pandas.DataFrame(t, columns=[*self.vars, *p_out])

    def info(self):
        print(self.__call__())

    def set(self,index,value):
        if isinstance(value, (int, float)):
            value = C(value)

        if isinstance(index, int):
            self.cpt_dict[index] = value
        elif len(index) == 1:
            self.cpt_dict[index[0]] = value
        elif len(index) == 2:
            self.cpt_dict[index[0]][index[1]] = value
        elif len(index) == 3:
            self.cpt_dict[index[0]][index[1]][index[2]] = value
        self.init()

    def init(self):
        self.cpt = self.CPT_to_theanoarray(self.cpt_dict)
        self.cpt_eval = self.cpt.eval()

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
        if len(self.n_vars) == 1:
            for n,key in enumerate(CPT):
                a[n] = np.array(CPT[key])
        elif len(self.n_vars) == 2:
            for n,key in enumerate(CPT):
                for m,key2 in enumerate(CPT[key]):
                    a[n,m] = np.array(CPT[key][key2])
        else: raise Exception("CPT not dim=2 or dim=1")

        return theano.shared(np.asarray(a))

    def pymc_model_fn(self, v1, v2=None):
        if len(self.n_vars) == 1:
            return self.cpt[v1]
        elif len(self.n_vars) == 2:
            return self.cpt[v1][v2]
        else: raise Exceptions("TODO")

    def fn(self, v1, v2=None):
        return self.pymc_model_fn(v1=v1, v2=v2)



def graph_replace_text(graphvizfig, varname, text):
    for n,part in enumerate(graphvizfig.body):
        if varname+' ' in part:
            graphvizfig.body[n] = part.replace('Categorical', text)
            return graphvizfig
    raise Exception("var not found!")


def encode(state_values=[1,1,2], dims=[2,2,3]):
    n_states = np.prod(np.array(dims))
    dims.reverse()
    state_values.reverse()
    state = 0
    cumulative = 1
    for d,s in zip(dims,state_values):
        state += cumulative * s
        cumulative *= d
    return state

def decode(state=11, dims=[2,2,3]):
    n_states = np.prod(np.array(dims))
    states = []
    for d in dims:
        s = state % d
        states.append(s)
        state = state//d
    return states

if __name__ == '__main__':
    state = encode([1,1,2])
    state
    decode(state)

    state = encode([9,9,9], dims=[10,10,10])
    state
    decode(state, dims=[10,10,10])
