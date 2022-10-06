# Pose = np.array([x,y,z,roll,pitch,yaw])
# Point = np.array([x,y,z]) or Vector3 np.array([xlen, ylen, zlen])

import numpy as np
from numpy import array as a

class Object():
    ''' static attributes '''
    all_types = ['cup', 'drawer', 'object']

    def __init__(self, name, # String
                       position, # Point
                       random = True
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
        self.stackable = True
        self.graspable = True
        self.pushable = True
        self.pourable = 0.1
        self.full = False
        self.color = 'b'
        if random: self.color = np.random.choice(['r','g','b'])

    @property
    def type_id(self):
        return self.all_types.index(self.type)

    def get_unique_state(self):
        ''' Unique state of the box is if it is on top '''
        return int(self.on_top)

    def experimental__get_obs(self):
        o = []
        o.append(1 if self.under is not None else 0)
        o.append(1 if self.above is not None else 0)
        o.append(int(self.inside_drawer))
        o.extend(list(self.position))
        if self.type == 'drawer':
            o.append(len(self.contains))
        if self.type == 'cup':
            o.append(int(self.full))

        return np.array(o).flatten()

    def experimental__get_obs2(self):
        o = np.zeros([1])
        #o[0] = (1 if self.under is not None else 0)
        #o[1] = (1 if self.above is not None else 0)
        #o[2] = (int(self.inside_drawer))
        if self.type == 'drawer':
            o[0] = (len(self.contains))
        if self.type == 'cup':
            o[0] = (int(self.full))
        if self.type == 'object':
            o[0] = (int(self.on_top))
        return o

    def random_capacity(self):
        return np.random.choice(['', 'stacked'])

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
        if (not self.above and not self.inside_drawer): return True
        return False

    @property
    def free(self):
        return self.on_top

    @property
    def info(self):
        print(self.__str__())

    def __str__(self):
        return f'{self.name},\t{self.type},\t{self.position},\t{self.print_structure(out_oneline_str=True)}'

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
    '''
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
    '''
    def stack(self, object_attached=None):
        if object_attached is None:
            return False
        if object_attached.above is not None:
            return False
        if not self.free:
            return False
        if not self.stackable:
            return False
        if self.inside_drawer:
            return False
        if self is object_attached:
            return False
        # current object name of object under
        self.above = object_attached
        if object_attached: object_attached.under = self
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
        if self.above is None: return ""
        return self.above.name

    @property
    def under_str(self):
        if self.under is None: return ""
        return self.under.name

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

    def print_structure(self, out_oneline_str=False):
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

        if out_oneline_str: # return in one line string, don't print anything
            strg = "|| "
            under_list.reverse()
            for item in under_list:
                strg += item
                strg += ' '
            strg += f"[{self.name}]"
            strg += ' '
            for item in above_list:
                strg += item
                strg += ' '
            strg += ">>"
            return strg
        else:
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

    def position_real(self, scene_lens=[4,4,4], max_scene_len=0.8):
        scene_lens = np.array(scene_lens)

        one_tile_lens = max_scene_len/scene_lens
        y_translation = (scene_lens[1]-1)*one_tile_lens[1]/2

        position_scaled = self.position * one_tile_lens
        position_translated = position_scaled + [0.2, -y_translation, self.size/2]

        z_add = 0
        slf = self
        for _ in range(len(self.under_list)):
            slf = self.under
            z_add += slf.size
        position_translated[2] += z_add

        return position_translated

class Drawer(Object):
    def __init__(self, name="?", position=[0,0,0], opened=False, random=True):
        super().__init__(name, position)
        self.opened = opened
        if random:
            self.opened = bool(np.random.randint(2))
        self.contains = []
        self.type = 'drawer'
        self.max_allowed_size = 0.15
        self.size = 0.3
        self.stackable = True
        self.graspable = False
        self.pushable = False
        self.pourable = 0.2
        ## experimental
        self.open_close_count = 0

    def get_unique_state(self):
        ''' Unique state of the drawer is if it opened or closed '''
        return int(self.opened)

    def random_capacity(self):
        return np.random.choice(['', 'contains'])

    def stack(self, object_under=None):
        return False # drawer cannot be stacked

    def __str__(self):
        return f'{self.name},\t{self.type},\t{self.position}, {self.opened_str}, cont: {[c.name for c in self.contains]},\t{self.print_structure(out_oneline_str=True)}'

    @property
    def contains_list(self):
        return [c.name for c in self.contains]

    @property
    def opened_str(self):
        ### TODO: not general !!! semi opened == opened now
        return 'semi-opened' if self.opened else 'closed'

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
        if random: self.size = np.random.randint(3,8)/100
        self.inside_drawer = False
        self.stackable = False
        self.graspable = True
        self.pushable = True
        self.pourable = 0.99

    def get_unique_state(self):
        ''' Unique state of the cup is if it full or empty '''
        return int(self.full)

    def random_capacity(self):
        return np.random.choice(['', 'stacked'])

    def rotate(self, direction):
        self.direction = np.array(direction)
        if not (direction == np.array([0,0,1])).all():
            self.full = False
        return

    def __str__(self):
        return f'{self.name},\t{self.type},\t{self.position}, {self.full_str},\t{self.print_structure(out_oneline_str=True)}'

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
    o_2.print_structure()
    o_2.stack(o_1)
    o_2.info
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
