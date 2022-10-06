import functools
import py_trees
import py_trees_ros
import py_trees.console as console
import rospy
import sys, os
import numpy as np
import std_msgs.msg as std_msgs

from context_based_gesture_operation.msg import Scene as SceneRos

# if launching by script from this dir, add:
# sys.path.append("..")
from srcmodules.Scenes import Scene, SceneCoppeliaInterface
from srcmodules.Actions import Actions
from srcmodules.RobotActions import RobotActions,act

from py_trees_ros import subscribers
from context_based_gesture_operation.msg import Intent
from context_based_gesture_operation.msg import Gestures as GesturesRos
from context_based_gesture_operation.srv import G2I, G2IResponse

def shutdown(behaviour_tree):
    behaviour_tree.interrupt()

class UpdateScene(subscribers.ToBlackboard):
    def __init__(self, name, threshold=30.0):
        super(UpdateScene, self).__init__(name=name, topic_name="/tree/scene_in", topic_type=SceneRos,
        blackboard_variables={"scene": None}, clearing_policy=py_trees.common.ClearingPolicy.NEVER)

        self.blackboard = py_trees.blackboard.Blackboard()
        self.blackboard.scene = SceneRos()
        self.blackboard.some_var_1 = False

        self.threshold = threshold

    def update(self):
        self.logger.debug("%s.update()" % self.__class__.__name__)
        status = super(UpdateScene, self).update()
        if status != py_trees.common.Status.RUNNING:
            if self.blackboard.scene.robot_attached_str != "":
                self.blackboard.some_var_1 = True
                rospy.logwarn_throttle(60, "%s: gripper attached!" % self.name)
            else:
                self.blackboard.some_var_1 = False

            self.feedback_message = "idk"

        print(f"update scene: {status}")
        return status

class UpdateGestures(subscribers.ToBlackboard):
    def __init__(self, name, threshold=30.0):
        super(UpdateGestures, self).__init__(name=name, topic_name="/tree/gestures_in", topic_type=GesturesRos,
        blackboard_variables={"gestures": None}, clearing_policy=py_trees.common.ClearingPolicy.NEVER)

        self.blackboard = py_trees.blackboard.Blackboard()
        self.blackboard.gestures = GesturesRos()

        self.threshold = threshold

    def update(self):
        self.logger.debug("%s.update()" % self.__class__.__name__)
        status = super(UpdateGestures, self).update()

        self.feedback_message = "idk"
        print(f"update gestures: {status}")
        return status

class GenerateIntent(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(GenerateIntent, self).__init__(name=name)
        self.blackboard = py_trees.blackboard.Blackboard()

    def setup(self, timeout):
        self.feedback_message = "setup"
        return True

    def update(self):
        self.logger.debug("%s.update()" % self.__class__.__name__)
        try:
            g2i = rospy.ServiceProxy('g2i', G2I)
            response = g2i(gestures=self.blackboard.gestures, scene=self.blackboard.scene)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
            return py_trees.common.Status.FAILURE

        intent = response.intent
        self.blackboard.target_action = intent.target_action
        self.blackboard.target_object = intent.target_object
        self.blackboard.auxilary_parameters = intent.auxiliary_parameters

        print(f"generate intent: {py_trees.common.Status.SUCCESS}")
        return py_trees.common.Status.SUCCESS

class ExecuteTA(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(ExecuteTA, self).__init__(name=name)

    def setup(self, timeout):
        self.publisher = rospy.Publisher("/execute_intent", Intent, queue_size=10, latch=True)
        self.feedback_message = "setup"
        return True

    def update(self):
        self.logger.debug("%s.update()" % self.__class__.__name__)
        blackboard = py_trees.blackboard.Blackboard()
        if blackboard.target_action != "":
            self.publisher.publish(Intent(blackboard.target_action, blackboard.target_object, ""))
            self.feedback_message = f"executing {blackboard.target_action},{blackboard.target_object}"
        else:
            self.feedback_message = f"no action"
        print(f"Execute TA {py_trees.common.Status.SUCCESS}! {blackboard.target_action},{blackboard.target_object}")
        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status):
        #self.publisher.publish(std_msgs.String(""))
        self.feedback_message = "cleared"

class DeleteIntent(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(DeleteIntent, self).__init__(name=name)
        self.blackboard = py_trees.blackboard.Blackboard()

    def setup(self, timeout):
        self.feedback_message = "setup"
        return True

    def update(self):
        self.logger.debug("%s.update()" % self.__class__.__name__)
        self.blackboard.target_action = ""
        self.blackboard.target_object = ""
        self.blackboard.auxilary_parameters = ""
        return py_trees.common.Status.SUCCESS

# might be replaced with py_trees.blackboard.CheckBlackboardVariable(name="nnn",
# variable_name='', expected_value=False)
class TONotEqDrawer(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(TONotEqDrawer, self).__init__(name=name)
        self.blackboard = py_trees.blackboard.Blackboard()

    def setup(self, timeout):
        self.feedback_message = "setup"
        return True

    def update(self):
        self.logger.debug("%s.update()" % self.__class__.__name__)
        to_name = self.blackboard.target_object
        for o in self.blackboard.scene.objects:
            if o.name == to_name:
                # to
                if o.type == 'drawer':
                    return py_trees.common.Status.FAILURE
                else:
                    return py_trees.common.Status.SUCCESS
        print("TO is not in object names!!!")
        return py_trees.common.Status.FAILURE

class HoldingPrecondition(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(HoldingPrecondition, self).__init__(name=name)
        self.blackboard = py_trees.blackboard.Blackboard()
        self.blackboard.holding_precondition = False

    def setup(self, timeout):
        self.feedback_message = "setup"
        return True

    def update(self):
        self.logger.debug("%s.update()" % self.__class__.__name__)
        holding_precondition = self.blackboard.holding_precondition
        if holding_precondition == False:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE

class PickTO(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(PickTO, self).__init__(name=name)
        self.publisher = rospy.Publisher("/execute_intent", Intent, queue_size=10, latch=True)
        self.blackboard = py_trees.blackboard.Blackboard()

    def setup(self, timeout):
        self.feedback_message = "setup"
        return True

    def update(self):
        self.blackboard.holding_precondition = False
        self.logger.debug("%s.update()" % self.__class__.__name__)

        self.publisher.publish(Intent('pick_up', self.blackboard.held_object, ""))
        self.feedback_message = f"picking up object {self.blackboard.held_object}"

        return py_trees.common.Status.SUCCESS

class DrawerStateNotInPrecondition(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(DrawerStateNotInPrecondition, self).__init__(name=name)
        self.blackboard = py_trees.blackboard.Blackboard()

    def setup(self, timeout):
        self.feedback_message = "setup"
        return True

    def update(self):
        self.logger.debug("%s.update()" % self.__class__.__name__)

        to_name = self.blackboard.target_object
        n = None
        for n,o in enumerate(self.blackboard.scene.objects):
            if o.name == to_name:
                if o.type == 'drawer':
                    break
                else:
                    return py_trees.common.Status.FAILURE

        # check precondition - TODO: make more general
        if not self.blackboard.scene.objects[n].opened: # drawer is not open
            if self.blackboard.target_action != 'open': # target action is not 'open'
                return py_trees.common.Status.FAILURE # we need to fix them

        return py_trees.common.Status.SUCCESS

class HandEmpty(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(HandEmpty, self).__init__(name=name)
        self.blackboard = py_trees.blackboard.Blackboard()

    def setup(self, timeout):
        self.feedback_message = "setup"
        return True

    def update(self):
        self.logger.debug("%s.update()" % self.__class__.__name__)

        attached = self.blackboard.scene.robot_attached_str
        if attached == "":
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE

class PlaceObject(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(PlaceObject, self).__init__(name=name)
        self.publisher = rospy.Publisher("/execute_intent", Intent, queue_size=10, latch=True)
        self.blackboard = py_trees.blackboard.Blackboard()

    def setup(self, timeout):
        self.feedback_message = "setup"
        return True

    def update(self):
        self.logger.debug("%s.update()" % self.__class__.__name__)

        self.blackboard.held_object = self.blackboard.scene.robot_attached_str
        self.publisher.publish(Intent('place', "", ""))
        self.feedback_message = f"Placeing held object"
        self.blackboard.holding_precondition = True


        return py_trees.common.Status.SUCCESS

class ToggleDrawer(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(ToggleDrawer, self).__init__(name=name)
        self.publisher = rospy.Publisher("/execute_intent", Intent, queue_size=10, latch=True)
        self.blackboard = py_trees.blackboard.Blackboard()

    def setup(self, timeout):
        self.feedback_message = "setup"
        return True

    def update(self):
        self.logger.debug("%s.update()" % self.__class__.__name__)

        to_name = self.blackboard.target_object
        n = None
        o_name = None
        for n,o in enumerate(self.blackboard.scene.objects):
            if o.name == to_name:
                if o.type == 'drawer':
                    o_name = o.name
                    break
                else:
                    return py_trees.common.Status.FAILURE

        if self.blackboard.scene.objects[n].opened: # drawer is opened
            self.publisher.publish(Intent('close', o_name, ""))
            self.feedback_message = f"Opening {o_name}"
        else:
            self.publisher.publish(Intent('open', o_name, ""))
            self.feedback_message = f"Opening {o_name}"

        return py_trees.common.Status.SUCCESS

class HoldingPrecondition(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(HoldingPrecondition, self).__init__(name=name)
        self.blackboard = py_trees.blackboard.Blackboard()
        self.blackboard.holding_precondition = False
    def setup(self, timeout):
        self.feedback_message = "setup"
        return True

    def update(self):
        self.logger.debug("%s.update()" % self.__class__.__name__)
        if self.blackboard.holding_precondition:
            return py_trees.common.Status.FAILURE
            self.blackboard.holding_precondition = False
        else:
            return py_trees.common.Status.SUCCESS

def create_tree():
    ''' BH tree final
    '''
    # LVL 0 - root
    root = py_trees.composites.Sequence("root")
    # LVL 1
    update_scene = UpdateScene(name="Scene2BB")
    update_gestures = UpdateGestures(name="Gestures2BB")
    generate_intent = GenerateIntent("GenIntent")
    seq_lvl1 = py_trees.composites.Sequence("seq_lvl1")
    # LVL 2
    to_is_drawer_q = py_trees.composites.Selector("to is drawer?")
    holding_precondition_q = py_trees.composites.Selector("holding precond.?")
    execute_ta = ExecuteTA("execute_ta")
    delete_intent = DeleteIntent("delete_intent")
    # LVL 3
    to_not_eq_drawer = TONotEqDrawer("to_not_eq_drawer")
    drawer_precond_q = py_trees.composites.Selector("drawer precond.?")

    holding_precondition = HoldingPrecondition('holding_precondition')
    pick_to = PickTO('pick_to')

    # LVL 4
    drawer_state_not_in_preconditions = DrawerStateNotInPrecondition('drawer_state_not_in_preconditions')
    fix_drawer_precond = py_trees.composites.Sequence("fix_drawer_precond")
    toggle_drawer = ToggleDrawer("toggle_drawer")

    # --- LVL 5 ----------------------------------------------

    place_q = py_trees.composites.Selector("place first?")

    # --- LVL 6 ----------------------------------------------
    hand_empty = HandEmpty("hand_empty")
    place_object = PlaceObject("place_object")

    # --- tree -----------------------------------------------

    root.add_children([update_scene, update_gestures, generate_intent, seq_lvl1])
    seq_lvl1.add_children([to_is_drawer_q, holding_precondition_q, execute_ta, delete_intent])
    to_is_drawer_q.add_children([to_not_eq_drawer, drawer_precond_q])

    holding_precondition_q.add_children([holding_precondition, pick_to])

    drawer_precond_q.add_children([drawer_state_not_in_preconditions, fix_drawer_precond])

    fix_drawer_precond.add_children([place_q,toggle_drawer])

    place_q.add_children([hand_empty, place_object])



    return root
