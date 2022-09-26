''' Conditioned by o1, o2, measure
'''
import numpy as np
from srcmodules import Scenes, Objects, Robots

class ConditionedAction():
    def __init__(self):
        pass

class Put(ConditionedAction):
    def do(self, s, o1=None,o2=None,measure=None):
        common_sense_proba = 1.
        if o1.type == 'drawer':
            return False
        if o2.type == 'drawer': # put o1 into drawer o2
            if not o2.open(): return False
            if not o1.gripper_move(o2.position): return False
            if not o2.put_in(o1): return False
            if not o2.close(): return False
        else: # stack o1 on top of o2
            if not o1.on_top:
                if not o1.unstack(): return False
            if not o1.gripper_move(o2.position): return False
            if not o2.stack(o1): return False
        return common_sense_proba

class Pour(ConditionedAction):
    def do(self, s, o1=None,o2=None,measure=None):
        common_sense_proba = 1.
        if o1.type != 'cup':
            return False
        common_sense_proba *= o2.pourable
        if not o1.full: common_sense_proba *= 0.2
        if o2.full: common_sense_proba *= 0.2

        if not o1.empty(): return False
        if not o2.fill(): return False

        return common_sense_proba

class Push(ConditionedAction):
    def do(self, s, o1=None,o2=None,measure=None):
        common_sense_proba = 1.
        direction = np.array([-1,0,0])
        if not o1.pushable:
            return False

        new_position = np.int64(o1.position + direction * measure)

        if not s.collision_free_position(new_position): common_sense_proba *= 0.8
        if not s.in_scene(new_position): common_sense_proba *= 0.8

        if not o1.push_move(new_position): return False

        return common_sense_proba



if __name__ == '__main__':
    # Put cup into drawer
    s = Scenes.Scene(init='drawer_and_cup')
    print(s.drawer)
    Put().do(s, o1=s.cup,o2=s.drawer)
    print(s.drawer)

    # Pour cup one into cup two
    s = Scenes.Scene(init='drawer_and_cups')
    s.cup1.fill()
    print(s.cup1)
    print(s.cup2)
    Pour().do(s, o1=s.cup1,o2=s.cup2)
    print(s.cup1)
    print(s.cup2)

    # Pour cup one into drawer
    s = Scenes.Scene(init='drawer_and_cups')
    s.cup1.fill()
    print(s.cup1)
    Pour().do(s, o1=s.cup1,o2=s.drawer)
    print(s.cup1)
    print(s.drawer)

    # Spill the cup
    s.cup2.rotate([1,0,0])
    print(s.cup2)


    s = Scenes.Scene(init='drawer_and_cups')
    print(s)
    s.cup1.position
    Push().do(s, o1=s.cup1, measure=1)
    s.cup1.position




#
