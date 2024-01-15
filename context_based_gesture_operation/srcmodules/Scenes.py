'''
>>> import sys; sys.path.append('..')
'''
import sys
import numpy as np
from random import choice

try:
    import srcmodules.Objects as Objects
    import srcmodules.Robots as Robots
    from srcmodules.Users import Users
    from srcmodules.Actions import Actions
    from srcmodules.SceneFieldFeatures import SceneFieldFeatures
except ModuleNotFoundError:
    import context_based_gesture_operation.srcmodules.Objects as Objects
    import context_based_gesture_operation.srcmodules.Robots as Robots
    from context_based_gesture_operation.srcmodules.Users import Users
    from context_based_gesture_operation.srcmodules.Actions import Actions
    from context_based_gesture_operation.srcmodules.SceneFieldFeatures import SceneFieldFeatures    

from copy import deepcopy
from geometry_msgs.msg import Point, Quaternion, Pose

class Scene():

    def __init__(self, grid_lens = [4,4,4], objects=[], init='no_task', user=None, random=True, import_data=None, name="scene0"):
        self.r = Robots.Robot()
        if user is None:
            if random:
                user = np.random.randint(2)
            else:
                user = 0
        self.u = Users(user)

        self.objects = []
        self.objects = objects
        self.grid_lens = grid_lens
        self.name = name
        if init in ['no_task', '']:
            if self.has_duplicate_objects(): raise Exception("Init. objects with same name!")

        elif init == 'from_dict':
            random = False
            self.from_dict(import_data)
        elif init == 'from_ros':
            random = False
            self.from_ros(import_data)
        else:
            self.objects = []
            obj_list = init.split(',')
            for obj in obj_list:
                p = None
                if random: p = self.get_random_position_in_scene(type=obj)
                else: p = self.get_position_in_scene()
                self.objects.append(getattr(Objects, obj.capitalize())(name=self.get_unique_object_name(obj), position=p, random=random))

        if random:
            eef_position = self.get_random_position_in_scene('z=3,x<4')
            self.r.eef_position = eef_position.copy()
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
            else:
                self.r.eef_position = eef_position

    def scene_to_observation(self, type=1, focus_point=None, max_n_objects=7, real=True):
        if real:
            object_positions = self.object_positions_real
            eef_position = self.eef_position_real
        else:
            object_positions = self.object_positions
            eef_position = self.r.eef_position

        v1 = np.zeros([max_n_objects+1])
        v1[self.get_gripper_object_id()] = 1

        v2 = np.zeros([max_n_objects])
        focf = SceneFieldFeatures.eeff__feature(object_positions, focus_point)
        v2[:len(focf)] = focf

        v3 = np.zeros([max_n_objects])
        eeff = SceneFieldFeatures.eeff__feature(object_positions, eef_position)
        v3[:len(eeff)] = eeff

        v2_diff = np.zeros([max_n_objects])
        tmp_ = object_positions - focus_point
        tmp__ = np.sum(np.power(tmp_,2), axis=1)
        v2_diff[:len(tmp__)] = tmp__

        v3_diff = np.zeros([max_n_objects])
        tmp_ = object_positions - eef_position
        tmp__ = np.sum(np.power(tmp_,2), axis=1)
        v3_diff[:len(tmp__)] = tmp__

        vo = np.zeros([1*max_n_objects])
        for n,obj in enumerate(self.objects):
            vo[n*1:n*1+1] = (list(obj.experimental__get_obs2()))

        v4 = np.zeros([len(Objects.Object.all_types)])
        if self.r.attached is not None:
            v4[Objects.Object.all_types.index(self.r.attached.type)] = 1

        # object types
        v5 = np.zeros([7,3])
        for n,obj, in enumerate(self.objects):
            v5[n][obj.type_id] = 1
        v5 = list(v5.flatten())

        if type == 0: # first trial
            return []
        elif type == 1: # all info - just to try it out
            return self.experimental__get_obs()
        elif type == 2:
            return SceneFieldFeatures.eeff__feature(object_positions, focus_point)
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
            # focus field, object states
            return [*v2, *vo]
        elif type == 9:
            #focus field, object types,
            return [*v2, *v5]
        elif type == 10:
            # focus field, object states, object types
            return [*v2, *vo, *v5]

        else: raise Exception("Scene to observation - not the right type!")

    def scene_encode_to_state(self, TaTo=None):
        ''' Experimental '''
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
    def n(self):
        return len(self.O)
    
    @property
    def empty_scene(self):
        if len(self.O) > 0:
            return False
        else: 
            return True
    
    @property
    def info(self):
        print(self.__str__())

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = f"Scene info. shape: {self.grid_lens}\n"
        for n, o in enumerate(self.objects):
            s += f'{n}. '
            s += str(o)
            s += '\n'
        s += str(self.r)
        s += '\n'
        s += str(self.u)
        return s

    def experimental__get_obs(self):
        raise NotImplementedError("Revision needed!")
        o = []
        o.extend(list(self.r.eef_position))
        o.append(self.r.eef_rotation)
        o.append(self.get_gripper_object_id())
        for obj in self.objects:
            o.extend(list(obj.experimental__get_obs()))
        return np.array(o).flatten()

    def experimental__get_obs2(self, object_positions, eef_position):
        raise NotImplementedError("Revision needed!")
        o = []
        o.extend(SceneFieldFeatures.eeff__feature(object_positions, eef_position))
        o.extend(SceneFieldFeatures.feaf__feature(self.object_sizes, self.r.gripper_range))
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

    def get_position_in_scene(self, constraint='', type=''):
        for x in range(self.grid_lens[0]):
            for y in range(self.grid_lens[1]):
                p = [x, y, 0]
                if self.collision_free_position(p):
                    return p
        raise Exception("Full scene?")

    def get_random_position_in_scene(self, constraint='on_ground,x-cond,free', type='object'):

        xlen, ylen, zlen = self.grid_lens
        
        if type=='object' or type=='cup': x_position = np.random.choice(range(xlen-2))+1
        elif type=='drawer': x_position = 3
        else: raise Exception(f"Not the right object type: {type}")


        if not constraint:
            p = np.hstack([np.random.choice(range(xlen)), np.random.choice(range(ylen)), np.random.choice(range(zlen))])
        elif constraint == 'on_ground':
            p = np.hstack([np.random.choice(range(xlen)), np.random.choice(range(ylen)), 0])
        elif constraint == 'z=3,x<4':
            p = np.hstack([np.random.choice(range(xlen-1)), np.random.choice(range(ylen)), 3])
        elif constraint == 'on_ground,x-cond':
            p = np.hstack([x_position, np.random.choice(range(2))+2, 0])
        elif constraint == 'on_ground,x-cond,free':
            p = np.hstack([x_position, np.random.choice(range(2))+2, 0])
            i = 0
            while not self.collision_free_position(p):
                p = np.hstack([np.random.choice(range(xlen-1))+1, np.random.choice(range(self.grid_lens[1])), 0])
                i+=1
                if i > 1000: raise Exception("Didn't found free space, scene too small!")
        return p
    
    def get_random_position_in_scene2(self, constraint='on_ground,free', type='object'):
        '''
        Panda table GRID, reachable:
         Y=0 Y=1 Y=2 Y=3
        |---------------|
        |       P       | X=0
        |--- --- --- ---| 
        |-x- -x- -x- -x-| X=1
        |-x- -x- -x- -x-| X=2
        |--- -x- -x- ---| X=3
        |--- --- --- ---| X=4
        |               |
        |---------------|
        '''
        grid = np.array([
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ])
        xlen, ylen, zlen = self.grid_lens
        assert xlen == 4, "TODO"
        assert ylen == 4, "TODO"
        assert zlen == 4, "TODO"

        if 'on_ground' in constraint:
            z = 0
        else:
            z = np.random.choice(range(zlen))

        if 'free' in constraint:
            x,y = None,None
            for x_,y_ in np.argwhere(grid):
                p = [x_,y_,z]
                if self.collision_free_position(p):
                    x,y = x_,y_
                    break
            if x is None:
                print("Didn't found free space, choosing [x=1,y=2,z=1]_GRID values")
                x, y, z = 1, 2, 1 # choose space above ground
        else:
            x,y = choice(np.argwhere(grid))

        return np.array([x,y,z])

    @property
    def O(self):
        return [object.name for object in self.objects]

    @property
    def object_positions(self):
        return [obj.position for obj in self.objects]

    @property
    def object_positions_real(self):
        return [obj.position_real for obj in self.objects]

    @property
    def object_sizes(self):
        return [obj.size for obj in self.objects]

    @property
    def object_types(self):
        return [obj.type for obj in self.objects]

    @property
    def object_poses(self):
        return [[*obj.position, *obj.quaternion] for obj in self.objects]

    @property
    def object_poses_ros(self):
        return [Pose(position=Point(x=obj.position[0], y=obj.position[1], z=obj.position[2]), orientation=Quaternion(x=obj.quaternion[0],y=obj.quaternion[1],z=obj.quaternion[2],w=obj.quaternion[3])) for obj in self.objects]

    @property
    def object_names(self):
        return [obj.name for obj in self.objects]

    def get_object_by_type(self, type):
        for obj in self.objects:
            if obj.type == type:
                return obj
        return None

    def get_object_by_name(self, name):
        for obj in self.objects:
            if obj.name == name:
                return obj
        return None

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

    def get_closest_object(self, goal_pose):
        if isinstance(goal_pose, (list,tuple,np.ndarray)):
            pass
        else:
            goal_pose = [goal_pose.position.x, goal_pose.position.y, goal_pose.position.z]
        mindist, mindistid = np.inf, None
        for n,obj in enumerate(self.objects):
            dist = np.linalg.norm(np.array(obj.position_real) - np.array(goal_pose))
            if dist < mindist:
                mindist = dist
                mindistid = n
        return mindistid

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
            scene_state['objects'][o.name]['above_str'] = o.above_str
            scene_state['objects'][o.name]['under_str'] = o.under_str

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

    def to_ros(self, rosobj=None):
        if rosobj is None: raise Exception("to_ros() function needs Scene ROS object to be filled!")
        for n in range(7):
            rosobj.objects[n].position_real = np.array([0,0,0], dtype=float)
        for n,o in enumerate(self.objects):
            rosobj.objects[n].name = o.name

            rosobj.objects[n].position_real = np.array(o.position_real, dtype=float)
            rosobj.objects[n].type = o.type
            rosobj.objects[n].graspable = o.graspable
            rosobj.objects[n].pushable = o.pushable
            rosobj.objects[n].free = o.free
            rosobj.objects[n].size = o.size
            rosobj.objects[n].above_str = o.above_str
            rosobj.objects[n].under_str = o.under_str

            if o.type == 'drawer':
                rosobj.objects[n].opened = float(o.opened_)
                if o.contains_list != []:
                    rosobj.objects[n].contains_list = o.contains_list[0]
                else:
                    rosobj.objects[n].contains_list = ''
            if o.type == 'cup':
                rosobj.objects[n].full = float(o.full)

        rosobj.robot_eef_position_real = np.array(self.r.eef_position_real, dtype=float)
        rosobj.robot_gripper_opened = self.r.gripper_opened
        rosobj.robot_eef_rotation = 0.#np.array(self.r.eef_rotation, dtype=float)
        rosobj.robot_attached_str = self.r.attached_str
        rosobj.robot_gripper_range = self.r.gripper_range

        rosobj.user = self.u.name
        return rosobj

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
                if 'full' in objects[name].keys():
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

    def from_ros(self, scene_state):
        objects = scene_state.objects
        nobj=0
        while True:
            if scene_state.objects[nobj].name == '':
                break
            nobj+=1

        self.objects = []
        object_names = []
        for n in range(nobj):
            name = objects[n].name
            object_names.append(name)

            self.objects.append(getattr(Objects, objects[n].type.capitalize())(name=name, position_real=objects[n].position_real))
            self.objects[n].type = objects[n].type
            self.objects[n].graspable = objects[n].graspable
            self.objects[n].pushable = objects[n].pushable
            self.objects[n].size = objects[n].size

            if objects[n].type == 'drawer':
                self.objects[n].opened = objects[n].opened
            if objects[n].type == 'cup':
                self.objects[n].full = objects[n].full

        for n in range(nobj):

            under_str = objects[n].under_str
            for o in self.objects:
                if o.name == under_str:
                    self.objects[n].under = o
                    break
            above_str = objects[n].above_str
            for o in self.objects:
                if o.name == above_str:
                    self.objects[n].above = o
                    break
            if objects[n].type == 'drawer':
                contains_list = objects[n].contains_list
                for contain_item in contains_list:
                    for o in self.objects:
                        if o.name == contain_item:
                            self.objects[n].contains.append(o)
                            break

        self.r = Robots.Robot()
        self.r.eef_position_real = scene_state.robot_eef_position_real
        self.r.gripper_opened = scene_state.robot_gripper_opened
        self.r.eef_rotation = scene_state.robot_eef_rotation
        self.r.attached = None
        if scene_state.robot_attached_str != '':
            for o in self.objects:
                if o.name == scene_state.robot_attached_str:
                    self.r.attached = o
                    break
        self.u = Users(scene_state.user)

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

    def are_objects_separated(self):
        ''' Object positions are always update from real/sim
            - Checks if objects are in boundaries which corresponds to brackets
        '''
        for object in self.objects:
            if not np.allclose(object.position_real, object.position_grid, atol=2e-1):
                return False
        return True

    def position_real(self, position, max_scene_len=0.8):
        ''' Duplicite function in object.py
        (grid)position -> real_position
        '''
        grid_lens = np.array(self.grid_lens)

        one_tile_lens = max_scene_len/grid_lens
        y_translation = (grid_lens[1]-1)*one_tile_lens[1]/2

        position_scaled = position * one_tile_lens
        position_translated = position_scaled - [-0.2, y_translation, 0.]

        return position_translated

    def generate_grid(self):
        assert np.allclose(*self.grid_lens), "Not Implemented for different scene lens"
        xs, ys, zs = [], [], []
        for i in range(self.grid_lens[0]):
            x,y,z = self.position_real(position=[i,i,i])
            xs.append(x)
            ys.append(y)
            zs.append(z)
        return np.array(xs), np.array(ys), np.array(zs)

    def pos_real_to_grid(self, p, out=""):
        '''
        real_position -> (grid)position
        '''
        xs,ys,zs = self.generate_grid()
        x,y,z = p

        x_ = np.argmin(abs(xs-x))
        y_ = np.argmin(abs(ys-y))
        z_ = np.argmin(abs(zs-z))

        close = np.allclose(p, self.position_real(position=(x_,y_,z_)), atol=2e-2)

        if out == "with close": # bool close to grid
            return np.array([x_,y_,z_]), close
        return np.array([x_,y_,z_])

    @property
    def eef_position_real(self):
        return self.position_real(position = self.r.eef_position)

    def check_semantic_feasibility(self, target_action, target_object, ignore_location=True):
        possible_actions = Actions.get_possible_actions(self, ignore_location=ignore_location)

        possible_actions_for_target_object = [(a) for a in possible_actions if a[1]==target_object]

        #print(f"=== {target_action}, {a[0]}, {possible_actions_for_target_object}")

        if [(a) for a in possible_actions_for_target_object if a[0]==target_action] == []:
            return False
        else:
            return True

class SceneCoppeliaInterface():
    '''
    DEPRECATED
    '''
    def __init__(self, interface_handle=None, print_info=False):
        '''
        Parameters:
            interface_handle (object): Control simulator or real robot
        '''
        if interface_handle is None: raise Exception("No interface_handle!")
        self.interface_handle = interface_handle
        self.s = None

        self.print_info = print_info

    def new_observation(self, s, object):
        oobj = getattr(s, object)
        if oobj.type == 'drawer':
            focus_point = oobj.position_real + np.random.random(3)/10 + np.array([-0.10,0,0.0])
        else:
            focus_point = oobj.position_real + np.random.random(3)/10 + np.array([0,0,0.02])

        self.interface_handle.add_or_edit_object(name='Focus_target', pose=focus_point)

    def new_scene(self, s):
        '''
        Parameters:
            s (Scene): Scene instance
        '''
        self.interface_handle.open_gripper()
        if self.s is not None:
            self.remove_objects_from_scene()
        self.s = s

        for o in s.objects:
            ''' simplified object creation '''
            if o.type == 'object':
                self.interface_handle.add_or_edit_object(name=o.name, frame_id='panda_link0', size=o.size, color=o.color, pose=o.position_real, shape="cube")
            elif o.type == 'cup':
                self.interface_handle.add_or_edit_object(file="cup", name=o.name, frame_id='panda_link0', size=o.size*10, color=o.color, pose=o.position_real)
            elif o.type == 'drawer':
                if o.name == 'drawer2': raise Exception("TODO")
                self.interface_handle.add_or_edit_object(name=o.name, frame_id='panda_link0', size=o.size, color=o.color, pose=o.position_real, object_state=o.opened_str)
            else: raise Exception(f"Object type {o.type} not in the list!")

        position_real = self.s.position_real(position=self.s.r.eef_position)
        self.interface_handle.go_to_pose(position_real)

        if s.r.attached is not None:
            input("Scene finished -> press Enter")
            self.interface_handle.add_or_edit_object(name=s.r.attached.name, pose=position_real)
            self.interface_handle.pick_object(object=s.r.attached.name)

        if self.print_info: print("Scene initialization done!")

    def remove_objects_from_scene(self):
        if self.s is None: return False
        for o in self.s.objects:
            if o.type == 'drawer' and o.name == 'drawer':
                self.interface_handle.add_or_edit_object(name=o.name, frame_id='panda_link0', size=o.size, color=o.color, pose=[1.0,1.0,-0.5])
            elif o.type == 'drawer' and o.name == 'drawer1':
                self.interface_handle.add_or_edit_object(name=o.name, frame_id='panda_link0', size=o.size, color=o.color, pose=[1.0,-0.0,0.5])
            else:
                self.interface_handle.remove_object(name=o.name)

        position_real = self.s.position_real(position=[2,0,3])
        self.interface_handle.go_to_pose(position_real)
        input("Object removed on the scene? Press enter")

def test_specific_scenarios():
    ''' Cheking semantics '''
    from srcmodules.Actions import Actions
    ''' Pour multiple cups into the bowl '''
    o1 = Objects.Drawer(name='drawer1', position=[0,-1,0], random=False)
    o2 = Objects.Cup(name='cup1', position=[0,0,0], random=False)
    o3 = Objects.Cup(name='cup2', position=[0,1,0], random=False)
    s = Scene(objects=[o1, o2, o3], random=False)

    s.O
    s.object_positions_real
    s.object_sizes

    Actions.do(s, ('pick_up', 'cup1'), ignore_location=True)
    Actions.do(s, ('pour', 'drawer1'), ignore_location=True)
    Actions.do(s, ('place', 'cup1'), ignore_location=True)

    Actions.do(s, ('pick_up', 'cup2'), ignore_location=True)
    Actions.do(s, ('pour', 'drawer1'), ignore_location=True)
    Actions.do(s, ('place', 'cup2'), ignore_location=True)
    s

    ''' Stack three objects onto each other '''
    s = Scene(init='object,object,object', random=False)
    s
    Actions.do(s, ('pick_up', 'object'), ignore_location=True)
    Actions.do(s, ('put_on', 'object1'), ignore_location=True)
    Actions.do(s, ('pick_up', 'object2'), ignore_location=True)
    Actions.do(s, ('put_on', 'object'), ignore_location=True)
    s
    ''' Place Rotated Spam on Soup Can'''
    s = Scene(init='object,object', random=False)
    Actions.do(s, ('pick_up', 'object'), ignore_location=True)
    Actions.do(s, ('rotate', 'object1'), ignore_location=True)

    Actions.do(s, ('put_on', 'object1'), ignore_location=True)

    ''' Place Cup and tomatos on the right, spam to the left '''
    s = Scene(init='cup,cup,object', random=False)
    s.r.eef_position = np.array([2,2,2])
    Actions.do(s, ('pick_up', 'cup'), ignore_location=True)
    Actions.do(s, ('move_up', ''), ignore_location=True)
    Actions.do(s, ('move_right', ''), ignore_location=True)
    Actions.do(s, ('place', 'cup'), ignore_location=True)

    Actions.do(s, ('pick_up', 'object'), ignore_location=True)
    Actions.do(s, ('move_up', ''), ignore_location=True)
    Actions.do(s, ('move_left', ''), ignore_location=True)
    Actions.do(s, ('place', 'object'), ignore_location=True)

    Actions.do(s, ('pick_up', 'cup1'), ignore_location=True)
    Actions.do(s, ('move_up', ''), ignore_location=True)
    Actions.do(s, ('move_right', ''), ignore_location=True)
    Actions.do(s, ('place', 'cup'), ignore_location=True)

    ''' Clean up '''
    s
    s = Scene(init='drawer,object,object', random=False)
    Actions.do(s, ('open', 'drawer'), ignore_location=True)
    Actions.do(s, ('pick_up', 'object'), ignore_location=True)
    Actions.do(s, ('put', 'drawer'), ignore_location=True)

    Actions.do(s, ('pick_up', 'object1'), ignore_location=True)
    Actions.do(s, ('put', 'drawer'), ignore_location=True)
    s




def test_scenes():
    drawer1 = Objects.Drawer(name='drawer1', position=[0,0,0], random=False)
    drawer2 = Objects.Drawer(name='drawer2', position=[2,0,0], random=False)
    drawer1.info
    cup1 = Objects.Cup(name='cup1', position=[2,0,0], random=False)
    assert drawer2.open()
    drawer1.info
    drawer2.info
    assert drawer2.contains == []
    drawer2.put_in(cup1)
    assert drawer2.contains
    drawer2.info

    scene = Scene(objects=[drawer1, drawer2, cup1])
    scene.info

    scene.pos_real_to_grid([0.6,0.3,0.43])
    assert scene.are_objects_separated()
    grid = scene.generate_grid()


    assert scene.O == ['drawer1', 'drawer2', 'cup1']
    assert scene.collision_free()

    scene.objects[1].contains = [scene.cup1]
    assert scene.collision_free()

    scene = Scene(init='drawer,cup', random=False)
    scene.object_positions
    assert scene.collision_free()

    data = {
        'objects': {
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
            },
        'robot': {
            'eef_position': np.array([0,0,3]),
            'gripper_opened': True,
            'eef_rotation': 0,
            'attached': "",
            },
        'user': 0,
        }

    scene = Scene(init='from_dict', import_data=data)
    assert scene.cup1.print_structure(out_oneline_str='str') == '|| drawer [cup1] cup2 >>'
    assert scene.cup2.print_structure(out_oneline_str='str') == '|| drawer cup1 [cup2] >>'
    assert scene.drawer.print_structure(out_oneline_str='str') == '|| [drawer] cup1 cup2 >>'

    data = {
        'objects': {
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
            },
        'robot': {
            'eef_position': np.array([0,0,3]),
            'gripper_opened': True,
            'eef_rotation': 0,
            'attached': "",
            },
        'user': 0,
    }

    scene = Scene(init='from_dict', import_data=data)
    assert scene.cup1.print_structure(out_oneline_str='str') == '|| drawer [cup1] cup2 >>'
    assert scene.drawer.contains_list == ['cup1', 'cup2']
    scene.drawer.contains
    assert scene.collision_free()


    s = Scene(init='drawer,drawer')
    ''' Must be between 0 and 1 '''
    assert s.drawer.opened_ not in [0.0, 1.0]
    assert s.drawer1.opened_ not in [0.0, 1.0]
    ''' Must be 0 or 1 '''
    assert s.drawer.opened in [0.0, 1.0]
    assert s.drawer1.opened in [0.0, 1.0]

    s
    s.drawer.position_real
    s.drawer1.position_real
    s.objects[0].color

if __name__ == '__main__':
    test_scenes()
    test_specific_scenarios()