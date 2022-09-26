from geometry_msgs.msg import Pose, Point, Quaternion
from copy import deepcopy
import numpy as np
from srcmodules.Actions import Actions

def act(s, cop, TaTo):
    if len(TaTo) == 3:
        # TODO: --->
        if not Actions.do(s, TaTo, ignore_location=True): raise Exception("Action not valid!")
        ret = RobotActions.do(s, cop, TaTo)

        return ret
    elif len(TaTo) == 2:
        if not Actions.do(s, TaTo, ignore_location=True): raise Exception("Action not valid!")
        ret = RobotActions.do(s, cop, TaTo)

        return ret
    else: raise Exception("TaTo not the right length")

class RobotActions():

    @staticmethod
    def do(s, cop, TaTo):
        ## TODO add v1
        target_action, target_object = TaTo
        return getattr(RobotActions, target_action)(s, cop, target_object)

    @staticmethod
    def move_in_dir(s, cop, object_name, direction):
        block_position = s.r.eef_position# + np.array(direction)
        pos = s.position_real(block_position)

        cop.go_to_pose(Pose(Point(*pos), Quaternion(0,1,0,0)))
        if bool(input("1/1, enter to continue, anystr else to abort!")): return False
        return True

    @staticmethod
    def move_up(s, cop, object_name):
        return RobotActions.move_in_dir(s, cop, object_name, direction=[0,0,1])
    @staticmethod
    def move_down(s, cop, object_name):
        return RobotActions.move_in_dir(s, cop, object_name, direction=[0,0,-1])
    @staticmethod
    def move_left(s, cop, object_name):
        return RobotActions.move_in_dir(s, cop, object_name, direction=[0,-1,0])
    @staticmethod
    def move_right(s, cop, object_name):
        return RobotActions.move_in_dir(s, cop, object_name, direction=[0,1,0])
    @staticmethod
    def move_front(s, cop, object_name):
        return RobotActions.move_in_dir(s, cop, object_name, direction=[1,0,0])
    @staticmethod
    def move_back(s, cop, object_name):
        return RobotActions.move_in_dir(s, cop, object_name, direction=[-1,0,0])

    @staticmethod
    def pick_up(s, cop, object_name):
        pos = getattr(s, object_name).position_real()
        pos_upper = deepcopy(pos)
        pos_upper[2] += 0.1

        cop.open_gripper()
        if bool(input("1/4, enter to continue, anystr else to abort!")): return False
        cop.go_to_pose(Pose(Point(*pos_upper), Quaternion(0,1,0,0)))
        if bool(input("2/4, enter to continue, anystr else to abort!")): return False
        cop.go_to_pose(Pose(Point(*pos), Quaternion(0,1,0,0)))
        if bool(input("3/4, enter to continue, anystr else to abort!")): return False
        cop.pick_object(object_name)
        if bool(input("4/4, enter to continue, anystr else to abort!")): return False

        return True

    @staticmethod
    def put_on(s, cop, object_name):
        pos = getattr(s, object_name).position_real()

        def get_pose_of_action_put(pos):
            o = getattr(s, object_name)
            if o.type == 'drawer':
                raise Exception("TODO")
            elif o.type == 'object':
                ret = o.position_real()
                ret[2] += o.size
                return ret
            else: raise Exception("put to wrong type of object")

        pos_release = get_pose_of_action_put(pos)
        pos_release_upper = deepcopy(pos_release)
        pos_release_upper[2] += 0.3

        cop.go_to_pose(Pose(Point(*pos_release_upper), Quaternion(0,1,0,0)))
        if bool(input("1/4, enter to continue, anystr else to abort!")): return False
        cop.go_to_pose(Pose(Point(*pos_release), Quaternion(0,1,0,0)))
        if bool(input("2/4, enter to continue, anystr else to abort!")): return False
        cop.release_object()
        if bool(input("3/4, enter to continue, anystr else to abort!")): return False
        cop.go_to_pose(Pose(Point(*pos_release_upper), Quaternion(0,1,0,0)))
        if bool(input("4/4, enter to continue, anystr else to abort!")): return False

        return True

    @staticmethod
    def put(s, cop, object_name):
        position_handle = RobotActions.drawer_base_to_handle_position(getattr(s,object_name).position_real(), drawer_state=0.0)
        pos_release = position_handle - np.array([0.05, 0.0, -0.05])

        o = getattr(s, object_name)
        if o.type != 'drawer':
            raise Exception("TODO")
        pos_release_upper = deepcopy(pos_release)
        pos_release_upper[2] += 0.1

        cop.go_to_pose(Pose(Point(*pos_release_upper), Quaternion(0,1,0,0)))
        if bool(input("1/4, enter to continue, anystr else to abort!")): return False
        cop.go_to_pose(Pose(Point(*pos_release), Quaternion(0,1,0,0)))
        if bool(input("2/4, enter to continue, anystr else to abort!")): return False
        cop.release_object()
        if bool(input("3/4, enter to continue, anystr else to abort!")): return False
        cop.go_to_pose(Pose(Point(*pos_release_upper), Quaternion(0,1,0,0)))
        if bool(input("4/4, enter to continue, anystr else to abort!")): return False

        return True

    @staticmethod
    def place(s, cop, object_name, place_position):
        position_upper = place_position + np.array([0.0, 0.0, 0.05])

        cop.go_to_pose(Pose(Point(*position_upper), Quaternion(0,1,0,0)))
        if bool(input("1/4, enter to continue, anystr else to abort!")): return False
        cop.go_to_pose(Pose(Point(*place_position), Quaternion(0,1,0,0)))
        if bool(input("2/4, enter to continue, anystr else to abort!")): return False
        cop.release_object()
        if bool(input("3/4, enter to continue, anystr else to abort!")): return False
        cop.go_to_pose(Pose(Point(*position_upper), Quaternion(0,1,0,0)))
        if bool(input("4/4, enter to continue, anystr else to abort!")): return False

        return True

    @staticmethod
    def open(s, cop, object_name, v1=0.5):
        ''' Parameters:
            v1 (Float): How much to open 1.0 - open to the full (20cm), 0.0 - not open at all (0cm)
        '''
        position_handle = RobotActions.drawer_base_to_handle_position(getattr(s,object_name).position_real(), drawer_state=0.)

        position_opened = position_handle - np.array([0.2*v1, 0, 0])
        cop.add_or_edit_object(name='drawer', pose=s.drawer.position_real(), object_state='slightly-opened')

        position_withdrawn = position_opened - np.array([0.0,0,-0.1])
        cop.go_to_pose(pose=Pose(Point(*position_withdrawn), Quaternion(0.0,1.,0.,0)))

        if bool(input("1/5, enter to continue, anystr else to abort!")): return False
        position_withdrawn = position_opened + np.array([0.09,0,0.1])
        cop.go_to_pose(pose=Pose(Point(*position_withdrawn), Quaternion(0.0,1.,0.,0)))

        if bool(input("2/5, enter to continue, anystr else to abort!")): return False
        position_withdrawn = position_opened + np.array([0.09,0,0.0])
        cop.go_to_pose(pose=Pose(Point(*position_withdrawn), Quaternion(0.0,1.,0.,0)))
        if bool(input("3/5, enter to continue, anystr else to abort!")): return False

        position_withdrawn = position_opened + np.array([0.04,0,0.0])
        cop.go_to_pose(pose=Pose(Point(*position_withdrawn), Quaternion(0.0,1.,0.,0)))
        if bool(input("4/5, enter to continue, anystr else to abort!")): return False

        position_withdrawn = position_opened + np.array([-0.02,0,0.0])
        cop.go_to_pose(pose=Pose(Point(*position_withdrawn), Quaternion(0.0,1.,0.,0)))
        if bool(input("5/5, enter to continue, anystr else to abort!")): return False

    @staticmethod
    def close(s, cop, object_name, v1=0.5):
        ''' Parameters:
                v1 (Float): How much to close 1.0 - close to the end, 0.0 - not close at all
        '''
        # Initial drawer state
        drawer_state = 0.1
        position_handle = RobotActions.drawer_base_to_handle_position(getattr(s,object_name).position_real(), drawer_state=drawer_state)

        cop.open_gripper()
        if bool(input("1/5, enter to continue, anystr else to abort!")): return False
        position_front = position_handle - np.array([0.05,0,0])
        cop.go_to_pose(Pose(Point(*position_front), Quaternion(0.0,np.pi/2,0.,1000)))
        if bool(input("2/5, enter to continue, anystr else to abort!")): return False
        ''' Go to the location in front of a drawer '''
        cop.go_to_pose(pose=Pose(Point(*position_handle), Quaternion(0.0,np.pi/2,0.,1000)))
        if bool(input("3/5, enter to continue, anystr else to abort!")): return False
        position_handle_closed = position_handle + np.array([0.1,0.0,0.0])
        cop.go_to_pose(pose=Pose(Point(*position_handle_closed), Quaternion(0.0,np.pi/2,0.,1000)))

        if bool(input("4/5, enter to continue, anystr else to abort!")): return False
        position_withdrawn = position_handle_closed - np.array([0.05,0,0])
        cop.go_to_pose(pose=Pose(Point(*position_withdrawn), Quaternion(0.0,np.pi/2,0.,1000)))
        if bool(input("5/5, enter to continue, anystr else to abort!")): return False

        return True

    @staticmethod
    def pour(s, cop, object_name, v1=1.0):
        pos = getattr(s, object_name).position_real()
        pos_upper = deepcopy(pos)
        pos_upper[2] += 0.1

        ''' Go to position above the pouring object '''
        cop.go_to_pose(Pose(Point(*pos_upper), Quaternion(0,1,0,0)))
        if bool(input("1/3, enter to continue, anystr else to abort!")): return False
        ''' Rotate/Pour '''
        cop.go_to_pose(pose=Pose(Point(*pos_upper), Quaternion(0.0,np.pi/2,0.,1000)))
        if bool(input("2/3, enter to continue, anystr else to abort!")): return False
        ''' Rotate/Pour back '''
        cop.go_to_pose(Pose(Point(*pos_upper), Quaternion(0,1,0,0)))
        if bool(input("3/3, enter to continue, anystr else to abort!")): return False

        return True

    @staticmethod
    def drawer_base_to_handle_position(position, drawer_state=0.):
        ''' Tied to the drawer model
        '''
        position[2] -= (0.15 - 0.1068)
        position[0] -= 0.129
        position[0] -= drawer_state
        return position

    @staticmethod
    def reset__not_general(sci, cop):
        sci.remove_objects_from_scene()
        cop.open_gripper()
        cop.go_to_pose(Pose(Point(0.4,0.,0.4), Quaternion(0,1,0,0)))
        cop.add_or_edit_object(name='Focus_target', pose=[0,0,0.0])

        return True
