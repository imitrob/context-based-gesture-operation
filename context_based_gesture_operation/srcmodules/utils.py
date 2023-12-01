import numpy as np
from pathlib import Path
import sys, os


def show_each_action_occurance(dataset, A):
    lenA = len(A)
    sums = np.zeros(lenA)
    for sample in dataset:
        a = A.index(sample[1][0])
        sums[a]+=1
    return f"Action occurances: {sums}"


def get_parent_folder(path):
    #path_here = Path('/home/<user>/<ws>/src/context_based_gesture_operation/context_based_gesture_operation/agent_nodes/g2i.py')
    #path_here = Path(__file__)
    path_here = Path(path)
    for parent in path_here.parents:
        # parent = path_here.parents[0]
        if parent.stem == 'src':
            return parent
    raise Exception("Not found src folder")

def add_teleop_gesture_toolbox_path():
    
    sys.path.append(str(get_parent_folder(__file__).joinpath("teleop_gesture_toolbox", "teleop_gesture_toolbox")))
