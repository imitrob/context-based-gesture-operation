# Pose = np.array([x,y,z,roll,pitch,yaw])
# Point = np.array([x,y,z]) or Vector3 np.array([xlen, ylen, zlen])

import numpy as np
from numpy import array as a

class Object():
    ''' static attributes '''
    all_types = ['cup', 'drawer']

    def __init__(self, name, # String
                       position, # Point
                       random = True # don't have effect on this object
                       ):
        self.name = name
        self.position = np.array(position)
        self.direction = np.array([-1, 0, 0]) # default
        self.type = 'object'
        self.inside_drawer = False
        self.under = None # object
        self.above = None # object
        self.size = 0.05
        self.max_allowed_size = 0.0
        self.graspable = True
        self.pushable = True
        self.pourable = 0.1
        self.full = False

    def gripper_rotate(self, direction):
        if not self.graspable: return False
        self.rotation = direction
        return True

    def gripper_move(self, position):
        if sum(abs(self.position - position)) > 1: return False
        if not self.graspable: return False
        self.position = position
        return True

    def push_move(self, position):
        if sum(abs(self.position - position)) > 1: return False
        if position[2] != 0: return False
        if not self.pushable: return False
        self.position = position
        return True

    @property
    def on_top(self):
        if not self.under: return True
        return False

    @property
    def free(self):
        return self.on_top

    @property
    def info(self):
        print(self.__str__())

    def __str__(self):
        return f'{self.name}, {self.type}, {self.position}'

    def __eq__(self, obj2):
        ''' Returns True if collision
        >>> self = o1
        >>> obj2 = o2
        '''
        obj1_positions = [self.position]
        obj2_positions = [obj2.position]
        if self.type == 'drawer' and self.opened:
            obj1_positions.append(self.position + self.direction)
        if obj2.type == 'drawer' and obj2.opened:
            obj2_positions.append(obj2.position + obj2.direction)

        for obj1_position in obj1_positions:
            for obj2_position in obj2_positions:
                if np.array_equal(obj1_position, obj2_position):
                    # check option if object inside drawer
                    if not self.is_obj_inside_drawer(obj2):
                        return True

        return False

    def stack(self, object_under=None):
        if object_under:
            if object_under.above is not None:
                return False
            if object_under.type == 'cup':
                return False
        if self.inside_drawer:
            return False
        # current object name of object under
        self.under = object_under
        if object_under: object_under.above = self
        return True

    def unstack(self):
        if self.above is not None:
            return False
        if self.inside_drawer:
            return False
        if not self.under:
            return False
        self.under.above = None
        self.under = None
        return True

    @property
    def above_str(self):
        return obj.above.name

    @property
    def under_str(self):
        return obj.under.name

    @property
    def above_list(self):
        obj = self
        above_list = []
        while obj.above:
            above_list.append(obj.above.name)
            obj = obj.above
        return above_list

    @property
    def under_list(self):
        obj = self
        under_list = []
        while obj.under:
            under_list.append(obj.under.name)
            obj = obj.under
        return under_list

    def print_structure(self):
        obj = self
        above_list = []
        while obj.above:
            above_list.append(obj.above.name)
            obj = obj.above
        obj = self
        under_list = []
        while obj.under:
            under_list.append(obj.under.name)
            obj = obj.under

        print("Structure:")
        above_list.reverse()
        for item in above_list:
            print(item)
        print(f"[{self.name}]")
        for item in under_list:
            print(item)

    def is_obj_inside_drawer(self, object2):
        object1 = self
        if object1.type == 'drawer' and object2.size < object1.max_allowed_size and object2 in object1.contains:
            return True
        if object2.type == 'drawer' and object1.size < object2.max_allowed_size and object1 in object2.contains:
            return True
        return False


class Drawer(Object):
    def __init__(self, name="?", position=[0,0,0], opened=False, random=True):
        super().__init__(name, position)
        self.opened = opened
        if random:
            self.opened = bool(np.random.randint(2))
        self.contains = []
        self.type = 'drawer'
        self.max_allowed_size = 0.15
        self.size = 0.2
        self.graspable = False
        self.pushable = False
        self.pourable = 0.2
        ## experimental
        self.open_close_count = 0


    def stack(self, object_under=None):
        return False # drawer cannot be stacked

    def __str__(self):
        return f'{self.name}, {self.type}, {self.position}, {self.opened_str}, {[c.name for c in self.contains]}'

    @property
    def contains_list(self):
        return [c.name for c in self.contains]

    @property
    def opened_str(self):
        return 'opened' if self.opened else 'closed'

    def open(self):
        self.open_close_count += 1
        opened_before = self.opened
        self.opened = True
        if not opened_before:
            return True
        else:
            return False

    def close(self):
        self.open_close_count += 1
        opened_before = self.opened
        self.opened = False
        if opened_before:
            return True
        else:
            return False

    def put_in(self, object=None):
        if not object:
            return False
        if object in self.contains:
            return False
        if not self.opened:
            return False
        self.contains.append(object)
        object.inside_drawer = True
        return True

    def pick_up(self, object=None):
        if not object:
            return False
        if object not in self.contains:
            return False
        if not self.opened:
            return False
        self.contains.remove(object)
        return True

    def fill(self):
        self.full = False
        return True

    def empty(self):
        self.full = False
        return True

class Cup(Object):
    def __init__(self, name="?", position=[0,0,0], full=False, random=True):
        super().__init__(name, position)
        self.full = full
        if random: self.full = bool(np.random.randint(2))
        self.type = 'cup'
        self.direction = np.array([0, 0, 1]) # default
        self.size = 0.01 # [m] height
        if random: self.size = np.random.randint(1,10)/100
        self.inside_drawer = False
        self.graspable = True
        self.pushable = True
        self.pourable = 0.99

    def rotate(self, direction):
        self.direction = np.array(direction)
        if not (direction == np.array([0,0,1])).all():
            self.full = False
        return

    def __str__(self):
        return f'{self.name}, {self.type}, {self.position}, {self.full_str}'

    @property
    def full_str(self):
        return 'full' if self.full else 'empty'

    def fill(self):
        full_before = self.full
        self.full = True
        if not full_before:
            return True
        else:
            return False

    def empty(self):
        full_before = self.full
        self.full = False
        if full_before:
            return True
        else:
            return False



if __name__ == '__main__':
    # Collisions test check
    o1 = Object('o1', [0,0,0])
    o2 = Object('o2', [0,0,0])
    print(o1)
    o1 == o2
    o1 = Object('o1', [1,0,0])
    o2 = Object('o2', [0,0,0])
    o1 == o2
    o1 = Drawer('o1', [0,0,0])
    o2 = Object('o2', [0,0,0])
    o1 == o2
    o1 = Drawer('o1', [0,0,0])
    o2 = Object('o2', [1,0,0])
    o1 == o2
    o1 = Drawer('o1', [1,0,0])
    o1.open()
    o2 = Object('o2', [0,0,0])
    o1 == o2
    o1 = Drawer('o1', [0,1,0])
    o2 = Object('o2', [0,0,0])
    o1 == o2
    o1 = Drawer('o1', [0,0,0])
    o2 = Drawer('o2', [0,0,0])
    o1 == o2
    o1 = Drawer('o1', [0,0,0])

    print(o1)
    o2 = Cup('o2', [0,0,0])
    o1.open()
    o1.put_in('o2')
    o1.contains
    o1 == o2

    o_1 = Object('o_1', [0,0,0])
    o_2 = Object('o_2', [0,0,0])
    o_3 = Object('o_3', [0,0,0])
    o_4 = Object('o_4', [0,0,0])
    o_2.stack(o_1)
    o_3.stack(o_2)
    o_4.stack(o_3)
    o_1.print_structure()
    o_2.print_structure()
    o_3.print_structure()
    o_4.print_structure()

    o_1.unstack()
    o_2.unstack()
    o_3.unstack()

    o_4.unstack()
    o_4.print_structure()
    o_3.print_structure()
    o_3.unstack()
    o_2.unstack()
    o_1.unstack()
    o_1.print_structure()


    print("Testing")
    d1 = Drawer(name='drawer1')
    d1.position
    d1.name
    d1.open()
    d1.opened
    d1.close()
    d1.opened
    d1.position


    class a():
        def __init__(self):
            self.stories = [1,2,3]
    class b():
        def __init__(self, odk):
            self.odkaxnaa = odk
    aa = a()
    bb = b(aa)
    bb.odkaxnaa.stories = [1,2,3,4]
    aa.stories
    bb.odkaxnaa

    arr_obj = [aa,bb]
    bb in arr_obj
    arr_obj.remove(bb)

#
