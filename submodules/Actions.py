''' Conditioned by o
'''
import numpy as np

class Actions():
    ''' Static Params '''
    A = ['move_up', 'open', 'put', 'pour', 'close', 'pick_up']
    ## Moves -> might be changed or will be subset of actions A
    A_move = ['move_back','move_right','move_up','move_front','move_left','move_down']
    A_move_directions = [[1,0,0], [0,1,0], [0,0,1],  [-1,0,0], [0,-1,0], [0,0,-1]]

    @staticmethod
    def is_action_from_move_category(action):
        if action in Actions.A_move:
            return True
        else:
            return False

    @staticmethod
    def step(s, s2, action, ignore_location=True, out=False):
        next_state = s.copy()
        action_executed = Actions.do(next_state, action, ignore_location=ignore_location, out=out)

        reward = next_state == s2
        done = False
        if reward == True: done = True

        return next_state, reward, done, action_executed

    @staticmethod
    def get_possible_actions_bool_array(s, ignore_location=False):
        possible_actions = np.zeros([len(Actions.A), len(s.O)], dtype=bool)
        for n,a in enumerate(Actions.A):
            for m,o in enumerate(s.O):
                s2 = s.copy()
                if Actions.do(s2, (a, o), ignore_location=ignore_location):
                    possible_actions[n,m] = True
        return possible_actions

    @staticmethod
    def get_possible_actions(s, ignore_location=False, p=0.9):
        possible_actions = []
        for a in Actions.A:
            for o in s.O:
                s2 = s.copy()
                if Actions.do(s2, (a, o), ignore_location=ignore_location, p=p):
                    possible_actions.append((a,o))
        return possible_actions

    @staticmethod
    def get_possible_move_actions(s, ignore_location=False):
        possible_actions = []
        for a in Actions.A_move:
            s2 = s.copy()
            if Actions.do(s2, (a, None), ignore_location=ignore_location):
                possible_actions.append(a)
        return possible_actions

    @staticmethod
    def get_random_action():
        return np.random.choice(Actions.A)

    @staticmethod
    def get_random_possible_action(s, ignore_location=False, p=0.9):
        TaTos = Actions.get_possible_actions(s, ignore_location=ignore_location, p=p)
        if TaTos == []: return False
        return TaTos[np.random.randint(len(TaTos))]

    @staticmethod
    def get_action(id):
        return Actions.A[id]

    @staticmethod
    def get_valid_actions(s):
        valid_actions = []
        for action in Actions.A:
            for object in s.O:
                s_ = s.copy()
                r = Actions.do(s_, (action, object))
                if r:
                    valid_actions.append((action,object))

    @staticmethod
    def do(s, action, ignore_location=False, out=False, p=0.9, handle_location=False):
        '''
        Parameters:
            s (Scene): Scene
            a (Tuple):
                0 : Action
                1 : Object name
            ignore_location (bool): Doesn't check the eef position to target obj
            out (bool): Prints on the screen
        '''
        if isinstance(action, dict):
            action = (action['target_action'], action['target_object'])

        o = getattr(s, action[1]) if action[1] else ""
        if not Actions.is_action_from_move_category(action[0]) and handle_location:
            move_action_seq = s.plan_path_to_position(o.position, Actions)
            Actions.execute_path_to_position(s, move_action_seq)
            #print(f"Executed move actions: {move_action_seq}")

        ret = getattr(Actions, action[0])(s, o, p, ignore_location=ignore_location)
        if not ret:
            if out: print("Action cannot be performed!")
            return False
        if out: print(f'{action} done!')
        return True

    @staticmethod
    def sequence_of_actions(self):
        pass

    @staticmethod
    def A_exec():
        return ['move_upCup1', 'move_upCup2', 'openDrawer', 'moveFrontCup1', 'moveFrontCup2', 'pourCup1', 'pourCup2', 'closeDrawer', 'MoveBackCup1', 'MoveBackCup2']

    ''' Action definitions
    '''
    @staticmethod
    def put_on_target(s, o=None, p=None, ignore_location=False):
        common_sense_proba = 1.
        if not s.r.attached: return False
        i = 0
        while True:
            new_position = np.int64(np.hstack([np.random.choice(4, 2), 0]))
            if s.collision_free_position(new_position):
                if s.in_scene(new_position):
                    break
            i += 1
            if i > 10000:
                print(f"The object {o}")
                s.info
                raise Exception("Couldn't found new place where to push!")

        if common_sense_proba < p: return False
        s.r.attached.position = new_position
        s.r.attached = None

        return True

    @staticmethod
    def put(s, o, p, ignore_location=False):
        common_sense_proba = 1.

        if not ignore_location and sum(abs(s.r.eef_position - o.position)) > 1: return False
        if not s.r.attached: return False
        if o.type == 'drawer':
            if not o.opened: return False
            #if not s.r.attached.gripper_move(o.position): return False
            if common_sense_proba < p: return False
            if not o.put_in(s.r.attached): return False
        else:
            #if not s.r.attached.gripper_move(o.position): return False
            if common_sense_proba < p: return False
            if not o.stack(s.r.attached): return False
        s.r.attached = None

        return True

    @staticmethod
    def pour(s, o, p, ignore_location=False):
        common_sense_proba = 1.
        common_sense_proba *= o.pourable
        if not s.r.attached: return False
        if not s.r.attached.full: common_sense_proba *= 0.2
        if o.full: common_sense_proba *= 0.2

        if common_sense_proba < p: return False
        if not s.r.attached.empty(): return False
        if not o.fill(): return False

        if not ignore_location and sum(abs(s.r.eef_position - o.position)) > 1: return False

        return True

    @staticmethod
    def pick_up(s, o, p, ignore_location=False):
        common_sense_proba = 1.
        if not o.graspable: return False
        if s.r.attached is not None: return False
        if not o.on_top: return False
        if o.inside_drawer: return False

        if common_sense_proba < p: return False
        if not ignore_location and sum(abs(s.r.eef_position - o.position)) > 1: return False
        if (o.direction == np.array([0,0,1])).all():
            s.r.eef_direction = 0
        else:
            s.r.eef_direction = 1

        o.unstack()
        s.r.attached = o
        return True

    @staticmethod
    def open(s, o, p, ignore_location=False):
        common_sense_proba = 1.
        if o.type != 'drawer': return False
        if o.opened: common_sense_proba *= 0.2
        '''
        >>> import numpy as np
        >>> x = np.arange(1,50)
        >>> y = 1-x*0.01 if x<50 else 0.8
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(y)
        '''
        common_sense_proba *= 1-o.open_close_count*0.01 if o.open_close_count<50 else 0.8
        if common_sense_proba < p: return False
        if not ignore_location and sum(abs(s.r.eef_position - o.position)) > 1: return False

        o.open()
        return True

    @staticmethod
    def close(s, o, p, ignore_location=False):
        common_sense_proba = 1.
        if o.type != 'drawer': return False
        if not o.opened: common_sense_proba *= 0.2

        common_sense_proba *= 1-o.open_close_count*0.01 if o.open_close_count<50 else 0.8
        if common_sense_proba < p: return False
        if not ignore_location and sum(abs(s.r.eef_position - o.position)) > 1: return False

        o.close()
        return True
    @staticmethod
    def push(s, o, p, ignore_location=False):
        common_sense_proba = 1.
        if not o.pushable: return False

        if not ignore_location and sum(abs(s.r.eef_position - o.position)) > 1: return False
        i = 0
        while True:
            new_position = np.int64(o.position + np.hstack([np.random.choice(3, 2) - 1, 0]))
            if s.collision_free_position(new_position):
                if s.in_scene(new_position):
                    break
            i += 1
            if i > 10000: raise Exception("Couldn't found new place where to push!")

        if common_sense_proba < p: return False
        if not o.push_move(new_position): return False

        return True

    @staticmethod
    def move_back(s, o=None, p=None, ignore_location=False):
        return Actions._move_by_direction(s, np.array([1, 0, 0]))
    @staticmethod
    def move_right(s, o=None, p=None, ignore_location=False):
        return Actions._move_by_direction(s, np.array([0, 1, 0]))
    @staticmethod
    def move_up(s, o=None, p=None, ignore_location=False):
        return Actions._move_by_direction(s, np.array([0, 0, 1]))
    @staticmethod
    def move_front(s, o=None, p=None, ignore_location=False):
        return Actions._move_by_direction(s, np.array([-1, 0, 0]))
    @staticmethod
    def move_left(s, o=None, p=None, ignore_location=False):
        return Actions._move_by_direction(s, np.array([0,-1, 0]))
    @staticmethod
    def move_down(s, o=None, p=None, ignore_location=False):
        return Actions._move_by_direction(s, np.array([0, 0,-1]))
    '''
    @staticmethod
    def open_gripper(self):
        gripper_opened_before = self.gripper_opened
        self.gripper_opened = True
        if gripper_opened_before:
            return False
        else:
            return True
    @staticmethod
    def close_gripper(self):
        gripper_opened_before = self.gripper_opened
        self.gripper_opened = False
        if gripper_opened_before:
            return True
        else:
            return False
    '''
    @staticmethod
    def rotate(s, o=None, p=None):
        if self.eef_rotation == 0:
            self.eef_rotation = 1
            if s.r.attached:
                s.r.attached.direction = np.array([0,0,1])
        else:
            self.eef_rotation = 0
    @staticmethod
    def _move_by_direction(s, direction):
        if s.in_scene(s.r.eef_position + direction):
            s.r.eef_position = s.r.eef_position + direction
            if s.r.attached:
                s.r.attached.position = s.r.eef_position
            return True
        else:
            return False

    @staticmethod
    def execute_path_to_position(s, plan):
        for action in plan:
            if not Actions.do(s, (action, None)):
                print("Plan cannot be executed")
                return False
        return True
