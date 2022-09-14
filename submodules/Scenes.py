'''
>>> import sys; sys.path.append('..')
'''
import numpy as np
import submodules.Objects as Objects
import submodules.Robots as Robots
from submodules.Users import Users
from submodules.Actions import Actions
from submodules.features import Features
from copy import deepcopy

class Scene():

    def __init__(self, grid_lens = [4,4,4], objects=[], init='no_task', user=None, random=True, import_data=None):
        self.r = Robots.Robot()
        if user is None:
            user = np.random.randint(2)
        self.u = Users(user)
        self.objects = objects
        self.grid_lens = grid_lens
        if init in ['no_task', '']:
            self.objects = objects
            if self.has_duplicate_objects(): raise Exception("Init. objects with same name!")

        elif init == 'from_dict':
            random = False
            self.from_dict(import_data)
        else:
            self.objects = []
            obj_list = init.split(',')
            for obj in obj_list:
                self.objects.append(getattr(Objects, obj.capitalize())(name=self.get_unique_object_name(obj), position=self.get_random_position_in_scene(), random=random))

        if random:
            self.r.eef_position = self.get_random_position_in_scene('z=3')
            self.r.gripper_opened = bool(np.random.randint(2))
            self.r.eef_rotation = np.random.randint(2)

            object_names = self.O
            np.random.shuffle(object_names)
            for o in object_names:
                oobj = getattr(self, o)
                conf = oobj.random_capacity()
                if conf == '':
                    continue
                elif conf == 'contains':
                    o2 = self.get_random_object(constrains=['free','graspable'], exclude=[o])
                    if o2 is None: continue

                    if not Actions.do(self, ('pick_up', o2), fake_handle_location=True): print(f"Error! put_to_drawer: pick_up {o2}")
                    if not getattr(self,o).opened:
                        if not Actions.do(self, ('open', o), fake_handle_location=True): print(f"Error! put_to_drawer: open {o}")
                    if not Actions.do(self, ('put', o), fake_handle_location=True): print(f"Error! put_to_drawer {o2}: put {o}")
                    #print(f"DONE put to drawer {o2} to {o}")
                elif conf == 'stacked':
                    if not oobj.free: continue
                    o2 = self.get_random_object(constrains=['free','graspable','stackable'], exclude=[o])
                    if o2 is None: continue

                    if not Actions.do(self, ('pick_up', o), fake_handle_location=True): print(f"Error! stack: pick_up {o}\n{self.info}\n")
                    if not Actions.do(self, ('put', o2), fake_handle_location=True): print(f"Error! stack {o}: put {o2}")
                    #print(f"DONE stacked {o} to {o2}")
            attached = np.random.randint(2)
            if attached:
                o = self.get_random_object(constrains=['free','graspable'])
                if o is not None:
                    if not Actions.do(self, ('pick_up', o), fake_handle_location=True): print(f"Error! attached {o}")
                    if not Actions.do(self, ('move_up', o), fake_handle_location=True): print(f"Error! attached {o}")


    def scene_to_observation(self, type=1, focus_point=None, max_n_objects=7):
        v1 = np.zeros([max_n_objects+1])
        v1[self.get_gripper_object_id()] = 1

        v2 = np.zeros([max_n_objects])
        focf = Features.eeff__feature(self.object_positions, focus_point)
        v2[:len(focf)] = focf

        v3 = np.zeros([max_n_objects])
        eeff = Features.eeff__feature(self.object_positions, self.r.eef_position)
        v3[:len(eeff)] = eeff

        v2_diff = np.zeros([max_n_objects])
        tmp_ = self.object_positions - focus_point
        tmp__ = np.sum(np.power(tmp_,2), axis=1)
        v2_diff[:len(tmp__)] = tmp__

        v3_diff = np.zeros([max_n_objects])
        tmp_ = self.object_positions - self.r.eef_position
        tmp__ = np.sum(np.power(tmp_,2), axis=1)
        v3_diff[:len(tmp__)] = tmp__

        vo = np.zeros([1*max_n_objects])
        for n,obj in enumerate(self.objects):
            vo[n*1:n*1+1] = (list(obj.experimental__get_obs2()))

        v4 = np.zeros([len(Objects.Object.all_types)])
        if self.r.attached is not None:
            v4[Objects.Object.all_types.index(self.r.attached.type)] = 1

        if type == 0: # first trial
            return []
        elif type == 1: # all info - just to try it out
            return self.experimental__get_obs()
        elif type == 2:
            return Features.eeff__feature(self.object_positions, focus_point)
        elif type == 3:
            return [*v1, *v2]
        elif type == 4:
            return [*v1, *v2, *vo]
        elif type == 5:
            return [*v1, *v2, *v3, *vo]
        elif type == 6:
            return [*v1, *v2_diff, *v3_diff, *vo]
        elif type == 7:
            return [*v4]
        elif type == 8:
            return [*v2, *vo]

        else: raise Exception("Scene to observation - not the right type!")

    def scene_encode_to_state(self, TaTo=None):
        # State is one integer number trying to encode the whole scene state
        dims = [*self.grid_lens, len(self.O)+1, len(self.O)]
        n_states = np.prod(np.array(dims))
        #                  eef_position   x gripper obj                 x target obj
        state_values = [*self.r.eef_position, self.get_gripper_object_id()]
        dims.reverse()
        state_values.reverse()
        state = 0
        cumulative = 1
        for d,s in zip(dims,state_values):
            state += cumulative * s
            cumulative *= d
        return state

    @property
    def info(self):
        print(self.__str__())

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = f"Scene info. shape: {self.grid_lens}\n"
        for o in self.objects:
            s += str(o)
            s += '\n'
        s += str(self.r)
        return s

    def experimental__get_obs(self):
        o = []
        o.extend(list(self.r.eef_position))
        o.append(self.r.eef_rotation)
        o.append(self.get_gripper_object_id())
        for obj in self.objects:
            o.extend(list(obj.experimental__get_obs()))
        return np.array(o).flatten()

    def experimental__get_obs2(self):
        o = []
        Features.eeff__feature(self.object_positions, self.r.eef_position)
        Features.feaf__feature(object_sizes, gripper_range)
        return np.array(o).flatten()

    def get_gripper_object_id(self):
        if self.r.attached is None:
            return len(self.O)
        return self.O.index(self.r.attached.name)

    def get_object_id(self, name):
        return self.O.index(name)

    def get_object(self, id):
        return self.O[id]

    def get_random_object(self, constrains=[], exclude=[]):
        objects = []
        for o in self.O:
            oobj = getattr(self, o)
            if o in exclude: continue
            valid = True
            for constraint in constrains:
                if not getattr(oobj, constraint):
                    valid = False
            if valid: objects.append(o)
        if objects == []: return None
        return np.random.choice(objects)

    def get_random_stackable_object(self, free_condition=True):
        stackable_objects = []
        for o in self.O:
            oobj = getattr(self, o)
            if oobj.type != 'drawer' and oobj.type != 'cup':
                if free_condition:
                    if oobj.free:
                        stackable_objects.append(o)
                    else:
                        continue
                else:
                    stackable_objects.append(o)
        if stackable_objects == []:
            return None
        else:
            return np.random.choice(stackable_objects)

    def get_random_drawer(self, free_condition=False):
        drawers = []
        for o in self.O:
            if getattr(self, o).type == 'drawer':
                if free_condition: # check condition if drawer is free
                    if getattr(self, o).contains == []:
                        drawers.append(o)
                    else:
                        continue
                else: # do NOT check condition if drawer is free
                    drawers.append(o)
        if drawers == []:
            return None
        else:
            return np.random.choice(drawers)

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

    def generate_scene_state(self, A, G, U, selected_id):
        scene_state = self.to_dict()
        scene_state['A'] = A
        scene_state['G'] = G
        scene_state['obj_types'] = Objects.Object.all_types
        scene_state['User'] = U
        scene_state['User_C'] = selected_id
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
        'attached': self.r.attached_str,
        'gripper_range': self.r.gripper_range,
            }
        scene_state['user'] = self.u.name
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
        if robot['attached'] != '':
            for o in self.objects:
                if o.name == robot['attached']:
                    self.r.attached = o
                    break
        self.u = Users(scene_state['user'])

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
                reward += len(list(set(o2.contains_list).intersection(o1.contains_list)))
                max_reward += len(o2.contains_list)
                max_reward += 1
                if o1.opened == o2.opened:
                    reward += 1

        if reward == max_reward: return True
        return reward

    def plan_path_to_position(self, position, Actions):
        start = deepcopy(self.r.eef_position)
        goal = position

        n=0
        move_sequence = []
        while sum(start-goal)>0 and n<20:
            dir = goal - start

            # The set of moves might be conditioned by positions
            s2 = self.copy()
            s2.r.eef_position = start
            possible_actions = Actions.get_possible_move_actions(s2)
            possible_moves = []
            for possible_action in possible_actions:
                possible_moves.append(Actions.A_move_directions[Actions.A_move.index(possible_action)])

            dir = dir/np.linalg.norm(dir)

            # heuristics
            best_possible_move_id = np.argmin([np.linalg.norm(move - dir) for move in possible_moves])
            best_move_id = Actions.A_move_directions.index(possible_moves[best_possible_move_id])

            move_sequence.append(best_move_id)
            # apply move
            start += Actions.A_move_directions[best_move_id]
            n+=1
        if n >=20: raise Exception("Path couldn't be found!")

        return [Actions.A_move[i] for i in move_sequence]

    def position_real(self, position, scene_lens=[4,4,4], max_scene_len=0.8):
        ''' Duplicite function in object.py
        '''
        scene_lens = np.array(scene_lens)

        one_tile_lens = max_scene_len/scene_lens
        y_translation = (scene_lens[1]-1)*one_tile_lens[1]/2

        position_scaled = position * one_tile_lens
        position_translated = position_scaled - [-0.2, y_translation, 0.]

        return position_translated

class SceneCoppeliaInterface():
    def __init__(self, interface_handle=None, print_info=False):
        '''
        Parameters:
            interface_handle (object): Control simulator or real robot
        '''
        if interface_handle is None: raise Exception("No interface_handle!")
        self.interface_handle = interface_handle
        self.s = None

        self.print_info = print_info

    def new_scene(self, s):
        '''
        Parameters:
            s (Scene): Scene instance
        '''
        if self.s is not None:
            self.remove_objects_from_scene()
        self.s = s


        for o in s.objects:
            ''' simplified object creation '''
            if o.type == 'object':
                self.interface_handle.add_or_edit_object(name=o.name, frame_id='panda_link0', size=o.size, color=o.color, pose=o.position_real(), shape="cube")
            elif o.type == 'cup':
                self.interface_handle.add_or_edit_object(name=o.name, frame_id='panda_link0', size=o.size, color=o.color, pose=o.position_real(), shape="cylinder")
            elif o.type == 'drawer':
                if o.name == 'drawer2': raise Exception("TODO")
                self.interface_handle.add_or_edit_object(name=o.name, frame_id='panda_link0', size=o.size, color=o.color, pose=o.position_real())
            else: raise Exception(f"Object type {o.type} not in the list!")

            #interface_handle.add_or_edit_object(file=f"{settings.paths.home}/{settings.paths.ws_folder}/src/mirracle_gestures/include/models/{file}", size=size, color=color, mass=mass, friction=friction, inertia=inertia, inertiaTransformation=inertiaTransformation, dynamic=dynamic, pub_info=pub_info, texture_file=texture_file, name=obj_name, pose=self.scenes[id].object_poses[i], frame_id=settings.base_link)
            #interface_handle.add_or_edit_object(name=obj_name, frame_id=settings.base_link, size=size, color=color, pose=self.scenes[id].object_poses[i], shape='cube', mass=mass, friction=friction, inertia=inertia, inertiaTransformation=inertiaTransformation, dynamic=dynamic, pub_info=pub_info, texture_file=texture_file)

        position_real = self.s.position_real(position=self.s.r.eef_position)
        self.interface_handle.go_to_pose(position_real)

        if self.print_info: print("Scene initialization done!")

    def remove_objects_from_scene(self):
        if self.s is None: return False
        for o in self.s.objects:
            if o.type == 'drawer' and o.name == 'drawer':
                self.interface_handle.add_or_edit_object(name=o.name, frame_id='panda_link0', size=o.size, color=o.color, pose=[1.0,1.0,-0.5], object_state=o.opened_str)
            elif o.type == 'drawer' and o.name == 'drawer1':
                self.interface_handle.add_or_edit_object(name=o.name, frame_id='panda_link0', size=o.size, color=o.color, pose=[1.0,-0.0,0.5])
            else:
                self.interface_handle.remove_object(name=o.name)

        position_real = self.s.position_real(position=[2,0,3])
        self.interface_handle.go_to_pose(position_real)



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

    1
    import numpy as np
    np.histogram(np.array(np.round(np.random.normal(3.5, 1, 50)), dtype=int))
