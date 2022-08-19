'''
>>> import sys; sys.path.append('..')
'''
import numpy as np
import submodules.Objects as Objects
import submodules.Robots as Robots

class Scene():

    def __init__(self, grid_lens = [4,4,4], objects=[], init='no_task', random=True, import_data=None):
        self.r = Robots.Robot()
        self.objects = objects
        self.grid_lens = grid_lens
        if init in ['no_task', '']:
            self.objects = objects
            if self.has_duplicate_objects(): raise Exception("Init. objects with same name!")

        elif init == 'from_dict':
            random = False
            self.from_dict(import_data)

        elif init == 'drawer_and_cup':
            drawer = Objects.Drawer(name='drawer', position=[2,0,0], random=random)
            cup = Objects.Cup(name='cup', position=[2,1,0], random=random)
            self.objects = [drawer, cup]
        elif init == 'drawer_and_cups':
            drawer = Objects.Drawer(name='drawer', position=[2,0,0], random=random)
            cup1 = Objects.Cup(name='cup1', position=[2,1,0], random=random)
            cup2 = Objects.Cup(name='cup2', position=[2,2,0], random=random)
            self.objects = [drawer, cup1, cup2]
        elif init == 'objects':
            object1 = Objects.Object(name='object1', position=[2,0,0], random=random)
            object2 = Objects.Object(name='object2', position=[2,1,0], random=random)
            object3 = Objects.Object(name='object3', position=[2,2,0], random=random)
            self.objects = [object1, object2, object3]
        else:
            self.objects = []
            obj_list = init.split(',')
            for obj in obj_list:
                self.objects.append(getattr(Objects, obj.capitalize())(name=self.get_unique_object_name(obj), position=self.get_random_position_in_scene(), random=random))

        if random:
            self.r.eef_position = self.get_random_position_in_scene('z=3')
            self.r.gripper_opened = bool(np.random.randint(2))
            self.r.eef_rotation = np.random.randint(2)

    def to_state(self):
        np.zeros([4,4,4,2,2])
        np.zeros([*self.grid_lens, len(self.O), len(self.O) + 1])

        self.r.eef_position

        return

    @property
    def info(self):
        print(self.__str__())

    def __str__(self):
        s = f"Scene info. shape: {self.grid_lens}\n"
        for o in self.objects:
            s += str(o)
            s += '\n'
        s += str(self.r)
        return s

    def get_object(self, id):
        return self.O[id]

    def get_random_object(self):
        return np.random.choice(self.O)

    def get_unique_object_name(self, unique_name, name_list=None):
        '''
        >>> get_unique_object_name(None, 'drawer', ['drawer', 'drawer1', 'cup', 'drawer0'])
        >>> get_unique_object_name(None, 'cup', ['drawer', 'drawer1', 'cup', 'drawer0', 'cup1'])
        '''
        if not name_list: name_list = self.O
        n=0
        name = unique_name
        while name in name_list:
            n+=1
            name = unique_name+str(n)
        return name

    def get_random_position_in_scene(self, constraint='on_ground,x!=0,free'):
        xlen, ylen, zlen = self.grid_lens
        if not constraint:
            p = np.hstack([np.random.choice(range(xlen)), np.random.choice(range(ylen)), np.random.choice(range(zlen))])
        elif constraint == 'on_ground':
            p = np.hstack([np.random.choice(range(xlen)), np.random.choice(range(ylen)), 0])
        elif constraint == 'z=3':
            p = np.hstack([np.random.choice(range(xlen)), np.random.choice(range(ylen)), 3])
        elif constraint == 'on_ground,x!=0':
            p = np.hstack([np.random.choice(range(xlen-1))+1, np.random.choice(range(4)), 0])
        elif constraint == 'on_ground,x!=0,free':
            p = np.hstack([np.random.choice(range(xlen-1))+1, np.random.choice(range(4)), 0])
            i = 0
            while not self.collision_free_position(p):
                p = np.hstack([np.random.choice(range(xlen-1))+1, np.random.choice(range(4)), 0])
                i+=1
                if i > 1000: raise Exception("Didn't found free space, scene too small!")
        return p

    @property
    def O(self):
        return [object.name for object in self.objects]

    @property
    def object_positions(self):
        return [obj.position for obj in self.objects]

    def collision_free(self):
        for object1 in self.objects:
            for object2 in self.objects:
                if object1.name == object2.name: continue
                if object1 == object2:
                    return False
        return True

    def collision_free_position(self, position):
        tmp_obj = Objects.Object('tmp_obj', position=position)
        for object1 in self.objects:
            if object1 == tmp_obj:
                return False
        return True

    def has_duplicate_objects(self):
        O = []
        for object in self.objects:
            if object.name in O:
                return True
            O.append(object.name)
        return False

    def in_scene(self, position):
        if (np.array([0,0,0]) <= position).all() and (position < self.grid_lens).all():
            return True
        return False

    def __getattr__(self, attr):
        return self.objects[self.O.index(attr)]

    def generate_scene_state(self, A, G, U, selected_id, TaTo):
        scene_state = self.to_dict()
        scene_state['A'] = A
        scene_state['G'] = G
        scene_state['obj_types'] = Objects.Object.all_types
        scene_state['User'] = U
        scene_state['User_C'] = selected_id
        scene_state['TA'] = TaTo[0]
        scene_state['TO'] = TaTo[1]
        return scene_state

    def to_dict(self):
        scene_state = {}
        scene_state['objects'] = {}
        for o in self.objects:
            scene_state['objects'][o.name] = {}
            scene_state['objects'][o.name]['position'] = o.position
            scene_state['objects'][o.name]['type'] = o.type
            scene_state['objects'][o.name]['graspable'] = o.graspable
            scene_state['objects'][o.name]['pushable'] = o.pushable
            scene_state['objects'][o.name]['free'] = o.free
            scene_state['objects'][o.name]['size'] = o.size
            scene_state['objects'][o.name]['above_str'] = o.above_list
            scene_state['objects'][o.name]['under_str'] = o.under_list

            if o.type == 'drawer':
                scene_state['objects'][o.name]['opened'] = o.opened
                scene_state['objects'][o.name]['contains_list'] = o.contains_list

            if o.type == 'cup':
                scene_state['objects'][o.name]['full'] = o.full

        scene_state['robot'] = {
        'eef_position': self.r.eef_position,
        'gripper_opened': self.r.gripper_opened,
        'eef_rotation': self.r.eef_rotation,
        'attached': self.r.attached
            }
        return scene_state

    def copy(self):
        return Scene(init='from_dict', import_data=self.to_dict())

    def from_dict(self, scene_state):
        objects = scene_state['objects']
        robot = scene_state['robot']
        self.objects = []
        for n,name in enumerate(objects.keys()):
            self.objects.append(getattr(Objects, objects[name]['type'].capitalize())(name=name, position=objects[name]['position']))
            self.objects[n].type = objects[name]['type']
            self.objects[n].graspable = objects[name]['graspable']
            self.objects[n].pushable = objects[name]['pushable']
            self.objects[n].size = objects[name]['size']

            if objects[name]['type'] == 'drawer':
                self.objects[n].opened = objects[name]['opened']
            if objects[name]['type'] == 'cup':
                self.objects[n].full = objects[name]['full']

        for n,name in enumerate(objects.keys()):
            if 'under_str' in objects[name].keys():
                under_str = objects[name]['under_str']
                for o in self.objects:
                    if o.name == under_str:
                        self.objects[n].under = o
                        break
            if 'above_str' in objects[name].keys():
                above_str = objects[name]['above_str']
                for o in self.objects:
                    if o.name == above_str:
                        self.objects[n].above = o
                        break
            if objects[name]['type'] == 'drawer':
                if 'contains_list' in objects[name]:
                    contains_list = objects[name]['contains_list']
                    for contain_item in contains_list:
                        for o in self.objects:
                            if o.name == contain_item:
                                self.objects[n].contains.append(o)
                                break

        self.r = Robots.Robot()
        self.r.eef_position = robot['eef_position']
        self.r.gripper_opened = robot['gripper_opened']
        self.r.eef_rotation = robot['eef_rotation']
        self.r.attached = None
        if robot['attached'] is not None:
            for o in self.objects:
                if o.name == robot['attached']:
                    self.r.attached = o
                    break

    def __eq__(self, obj2):
        ''' Reward function
        '''

        if len(self.object_positions) != len(obj2.object_positions):
            raise Exception("scenes havent got same objects")
        reward = 0.
        max_reward = 0.
        for n,(o1,o2) in enumerate(zip(self.object_positions, obj2.object_positions)):
            reward -= sum(abs(o1 - o2))
        reward -= sum(abs(self.r.eef_position - obj2.r.eef_position))


        for n,(o1,o2) in enumerate(zip(self.objects, obj2.objects)):
            max_reward += 2
            if o1.under == o2.under:
                reward += 1
            if o1.above == o2.above:
                reward += 1
            if o1.type == 'drawer':
                if o2.type != 'drawer': raise Exception("scenes havent got same objects")
                reward += len(list(set(o1.contains_list).intersection(o2.contains_list)))
                max_reward += len(o1.contains_list)
                max_reward += 1
                if o1.opened == o2.opened:
                    reward += 1

        if reward == max_reward: return True
        return reward


if __name__ == '__main__':
    drawer1 = Objects.Drawer(name='drawer1', position=[0,0,0])
    drawer2 = Objects.Drawer(name='drawer2', position=[2,0,0])
    cup1 = Objects.Cup(name='cup1', position=[2,0,0])
    drawer2.open()
    drawer2.contains
    drawer2.put_in(cup1)

    scene = Scene(objects=[drawer1, drawer2, cup1])
    scene.O
    scene.collision_free()

    scene.objects[1].contains = ['cup1']
    scene.collision_free()

    scene = Scene(init='drawer_and_cup')
    scene.object_positions
    scene.collision_free()

    objects = {
        'drawer': {'position': np.array([3, 2, 0]),
            'type': 'drawer',
            'graspable': False,
            'pushable': False,
            'free': True,
            'size': 0.2,
            'opened': False,
            'above_str': 'cup1'},
        'cup1': {'position': np.array([3, 1, 0]),
            'type': 'cup',
            'graspable': True,
            'pushable': True,
            'free': True,
            'size': 0.01,
            'under_str': 'drawer',
            'above_str': 'cup2'},
        'cup2': {'position': np.array([1, 2, 0]),
            'type': 'cup',
            'graspable': True,
            'pushable': True,
            'free': True,
            'size': 0.01,
            'under_str': 'cup1'}
        }

    scene = Scene(objects=objects, init='from_dict')
    scene.cup1.print_structure()
    scene.cup2.print_structure()
    scene.drawer.print_structure()

    objects = {
        'drawer': {'position': np.array([3, 2, 0]),
            'type': 'drawer',
            'graspable': False,
            'pushable': False,
            'free': True,
            'size': 0.2,
            'opened': False,
            'contains_list': ['cup1', 'cup2'],
            'above_str': 'cup1'},
        'cup1': {'position': np.array([3, 1, 0]),
            'type': 'cup',
            'graspable': True,
            'pushable': True,
            'free': True,
            'size': 0.01,
            'under_str': 'drawer',
            'above_str': 'cup2'},
        'cup2': {'position': np.array([1, 2, 0]),
            'type': 'cup',
            'graspable': True,
            'pushable': True,
            'free': True,
            'size': 0.01,
            'under_str': 'cup1'}
        }

    scene = Scene(objects=objects, init='from_dict')
    scene.cup1.print_structure()
    scene.drawer.contains_list
    scene.drawer.contains
    scene.collision_free()


























#
