# Pose = np.array([x,y,z,roll,pitch,yaw])
# Point = np.array([x,y,z]) or Vector3 np.array([xlen, ylen, zlen])

import numpy as np
from numpy import array as a

class Object():
    ''' static attributes '''
    all_types = ['object', 'cup', 'drawer']

    def __init__(self, name, # String
                       position = None, # Point
                       random = True,
                       orientation = np.array([0., 0., 0., 1.]),
                       size = 0.05,
                       # Additional
                       ycb = False,
                       color = 'b',
                       scale = 1.,
                       shape_type = None, # Optional
                       mass = None, # Optional
                       friction = None, # Optional
                       inertia = None, # Optional
                       inertia_transformation = None, # Optional
                       position_real = None,
                       ):
        self.name = name
        self.size = size
        if isinstance(self.size, (tuple,np.ndarray,list)):
            self.size = size[0]
        assert ((position is not None) or (position_real is not None)), "Position is required"
        if position is not None:
            self.position = np.array(position)
        else:
            self.position_real = position_real

        self.direction = np.array([-1, 0, 0]) # default
        self.type = 'object'
        self.inside_drawer = False
        self.under = None # object
        self.above = None # object

        self.max_allowed_size = 0.0
        self.stackable = True
        self.graspable = True
        self.pushable = True
        self.pourable = 0.1
        self.full = False
        self.color = color
        if random: self.color = np.random.choice(['r','g','b'])

        self.quaternion = orientation


        self.ycb = ycb

        ''' Additional '''
        self.scale = scale
        self.shape_type = shape_type
        self.mass = mass
        self.friction = friction
        self.inertia = inertia
        self.inertia_transformation = inertia_transformation

        ''' Generate remaining variables '''
        if position is not None:
            self.position_real = self.make_position_real(random=random)
        else:
            self.position = self.pos_real_to_grid(position_real)

    @property
    def orientation(self):
        return self.quaternion

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
    def stack(self, object_attached=None, debug=False):
        if object_attached is None:
            if debug: print("no object written")
            return False
        if object_attached.above is not None:
            if debug: print("target object is not on top")
            return False
        if not self.free:
            if debug: print("object is not free")
            return False
        if not self.stackable:
            if debug: print("object is not stackable")
            return False
        if self.inside_drawer:
            if debug: print("object is inside drawer")
            return False
        if self is object_attached:
            if debug: print("object is attached")
            return False
        # current object name of object under
        self.above = object_attached
        if object_attached: object_attached.under = self
        return True

    def unstack(self, debug=False):
        if self.above is not None:
            if debug: print("something is above the object")
            return False
        if self.inside_drawer:
            if debug: print("object is in the drawer")
            return False
        if not self.under:
            if debug: print("object is not stacked")
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

    def make_position_real(self, scene_lens=[4,4,4], max_scene_len=0.8, random=False, random_magnitude=0.3):
        scene_lens = np.array(scene_lens)

        one_tile_lens = max_scene_len/scene_lens
        y_translation = (scene_lens[1]-1)*one_tile_lens[1]/2

        position_scaled = self.position * one_tile_lens
        position_translated = position_scaled + [0.2, -y_translation, self.size/2]

        if random: # Add xy noise
            position_translated[0:2] += np.random.random(2) * random_magnitude * one_tile_lens[0:2]

        z_add = 0
        slf = self
        for _ in range(len(self.under_list)):
            slf = self.under
            z_add += slf.size
        position_translated[2] += z_add

        return position_translated

    def make_position_real_using_position(self, position, max_scene_len=0.8, grid_lens=[4,4,4]):
        ''' Duplicite function in object.py
        '''
        grid_lens = np.array(grid_lens)

        one_tile_lens = max_scene_len/grid_lens
        y_translation = (grid_lens[1]-1)*one_tile_lens[1]/2

        position_scaled = position * one_tile_lens
        position_translated = position_scaled - [-0.2, y_translation, 0.]

        return position_translated

    def generate_grid(self, grid_lens=[4,4,4]):
        assert np.allclose(*grid_lens), "Not Implemented for different scene lens"
        xs, ys, zs = [], [], []
        for i in range(grid_lens[0]):
            x,y,z = self.make_position_real_using_position(position=[i,i,i])
            xs.append(x)
            ys.append(y)
            zs.append(z)
        return np.array(xs), np.array(ys), np.array(zs)

    def pos_real_to_grid(self, p, out=''):
        xs,ys,zs = self.generate_grid()
        x,y,z = p

        x_ = np.argmin(abs(xs-x))
        y_ = np.argmin(abs(ys-y))
        z_ = np.argmin(abs(zs-z))

        close = np.allclose(p, self.make_position_real_using_position(position=(x_,y_,z_)), atol=2e-2)

        if out == 'with close':
            return np.array([x_,y_,z_]), close
        return np.array([x_,y_,z_])


class Drawer(Object):
    def __init__(self, name="?", position=None, opened=False, random=True, *args, **kwargs):
        super().__init__(name, position, *args, **kwargs)
        if opened:
            self.opened_amount = 0.0
        else:
            self.opened_amount = 1.0
        


        self.opened = opened
        if random:
            self.opened = bool(np.random.randint(2))
        self.contains = []
        self.type = 'drawer'
        self.max_allowed_size = 0.15
        self.stackable = True
        self.graspable = False
        self.pushable = False
        self.pourable = 0.2
        ## experimental
        self.open_close_count = 0

    @property
    def opened(self):
        return round(self.opened_amount)

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
    def __init__(self, name="?", position=None, full=False, random=True, *args, **kwargs):
        super().__init__(name, position, *args, **kwargs)
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
    test_objects()

def test_objects():
    # Collisions test check
    o1 = Object('o1', [0,0,0])
    o2 = Object('o2', [0,0,0])
    assert str(o1) == 'o1,\tobject,\t[0 0 0],\t|| [o1] >>'

    assert np.allclose(o1.position_real, np.array([ 0.2  , -0.3  ,  0.025]))

    assert np.allclose(o1.position_real, o1.make_position_real())
    assert(o1 == o2)
    o1 = Object('o1', [1,0,0])
    o2 = Object('o2', [0,0,0])
    assert not (o1 == o2)
    o1 = Drawer('o1', [0,0,0])
    o2 = Object('o2', [0,0,0])
    assert o1 == o2
    o1 = Drawer('o1', [0,0,0])
    o2 = Object('o2', [1,0,0])
    assert not (o1 == o2)
    o1 = Drawer('o1', [1,0,0])
    assert o1.open()
    o2 = Object('o2', [0,0,0])
    assert o1 == o2
    o1 = Drawer('o1', [0,1,0])
    o2 = Object('o2', [0,0,0])
    assert not o1 == o2
    o1 = Drawer('o1', [0,0,0])
    o2 = Drawer('o2', [0,0,0])
    assert o1 == o2

    o1 = Drawer('o1', [0,0,0])
    o2 = Cup('o2', [0,0,0])
    o1.open()
    o1.put_in(o2)
    assert o1.contains[0].name == 'o2'

    o_1 = Object('o_1', [0,0,0])
    o_2 = Object('o_2', [0,0,0])
    o_3 = Object('o_3', [0,0,0])
    o_4 = Object('o_4', [0,0,0])
    assert str(o_2.print_structure(out_oneline_str=True)) == "|| [o_2] >>"

    o_1.above
    o_1.under
    o_1.info
    o_2.above
    o_2.under
    o_2.info

    o_3.above
    o_3.under
    o_3.info

    o_3.on_top
    o_2.on_top
    o_1.on_top

    o_2.stack(o_1)
    o_2.info
    o_3.stack(o_1, debug=True)
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
