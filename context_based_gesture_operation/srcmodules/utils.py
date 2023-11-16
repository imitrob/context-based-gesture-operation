import numpy as np

def show_each_action_occurance(dataset, A):
    lenA = len(A)
    sums = np.zeros(lenA)
    for sample in dataset:
        a = A.index(sample[1][0])
        sums[a]+=1
    return f"Action occurances: {sums}"