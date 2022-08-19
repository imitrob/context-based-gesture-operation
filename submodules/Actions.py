''' Conditioned by o
'''
import numpy as np

class Actions():
    ''' Static Params '''
    A = ['move_up', 'open', 'put', 'pour', 'close', 'pick_up']

    @staticmethod
    def step(s, s2, action, ignore_location=True):
        next_state = s.copy()
        Actions.do(next_state, action, ignore_location=ignore_location, out=True)

        reward = next_state == s2
        done = False
        if reward == True: done = True

        return next_state, reward, done

    @staticmethod
    def get_random_action():
        return np.random.choice(Actions.A)

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
    def do(s, action, ignore_location=False, out=False):
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
        if not getattr(Actions, action[0])(s, o, ignore_location=ignore_location):
            if out: print("Action cannot be performed!")
            return False
        print(f'{action} done!')
        return True

    @staticmethod
    def sequence_of_actions(self):
        pass

    @staticmethod
    def A_exec():
        return ['move_upCup1', 'move_upCup2', 'openDrawer', 'moveFrontCup1', 'moveFrontCup2', 'pourCup1', 'pourCup2', 'closeDrawer', 'MoveBackCup1', 'MoveBackCup2']

    @staticmethod
    def put_on_target(s, o=None, ignore_location=False):
        common_sense_proba = 1.
        if not s.r.attached: return False
        i = 0
        while True:
            new_position = np.int64(o.position + np.hstack([np.random.choice(3, 2) - 1, 0]))
            if s.collision_free_position(new_position):
                if s.in_scene(new_position):
                    break
            i += 1
            if i > 10000: raise Exception("Couldn't found new place where to push!")

        s.r.attached.position = new_position
        s.r.attached = None
        return True

    @staticmethod
    def put(s, o, ignore_location=False):
        common_sense_proba = 1.

        if not ignore_location and sum(abs(s.r.eef_position - o.position)) > 1: return False
        if not s.r.attached: return False
        if o.type == 'drawer':
            if not o.opened: return False
            #if not s.r.attached.gripper_move(o.position): return False
            if not o.put_in(s.r.attached): return False
        else:
            #if not s.r.attached.gripper_move(o.position): return False
            if not o.stack(s.r.attached): return False
        s.r.attached = None
        return common_sense_proba

    @staticmethod
    def pour(s, o, ignore_location=False):
        common_sense_proba = 1.
        common_sense_proba *= o.pourable
        if not s.r.attached: return False
        if not s.r.attached.full: common_sense_proba *= 0.2
        if o.full: common_sense_proba *= 0.2

        if not s.r.attached.empty(): return False
        if not o.fill(): return False

        if not ignore_location and sum(abs(s.r.eef_position - o.position)) > 1: return False

        return common_sense_proba

    @staticmethod
    def pick_up(s, o, ignore_location=False):
        common_sense_proba = 1.
        if not o.graspable: return False
        if s.r.attached is not None: return False
        if not o.on_top: return False
        if o.inside_drawer: return False

        if not ignore_location and sum(abs(s.r.eef_position - o.position)) > 1: return False
        if (o.direction == np.array([0,0,1])).all():
            s.r.eef_direction = 0
        else:
            s.r.eef_direction = 1

        s.r.attached = o
        return True

    @staticmethod
    def open(s, o, ignore_location=False):
        common_sense_proba = 1.
        if o.type != 'drawer': return False
        if o.opened: common_sense_proba *= 0.2

        if not ignore_location and sum(abs(s.r.eef_position - o.position)) > 1: return False

        o.open()
        return True
    @staticmethod
    def close(s, o, ignore_location=False):
        common_sense_proba = 1.
        if o.type != 'drawer': return False
        if not o.opened: common_sense_proba *= 0.2

        if not ignore_location and sum(abs(s.r.eef_position - o.position)) > 1: return False

        o.close()
        return True
    @staticmethod
    def push(s, o, ignore_location=False):
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

        if not o.push_move(new_position): return False

        return common_sense_proba
    @staticmethod
    def move_left(s, o=None, ignore_location=False):
        return Actions._move_by_direction(s, np.array([0,-1, 0]))
    @staticmethod
    def move_right(s, o=None, ignore_location=False):
        return Actions._move_by_direction(s, np.array([0, 1, 0]))
    @staticmethod
    def move_up(s, o=None, ignore_location=False):
        return Actions._move_by_direction(s, np.array([0, 0, 1]))
    @staticmethod
    def move_down(s, o=None, ignore_location=False):
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
    def rotate(s, o=None):
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
