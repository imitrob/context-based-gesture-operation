import functools
import py_trees
import py_trees_ros
import py_trees.console as console
import rclpy
from rclpy.node import Node
import sys, os
import numpy as np
import std_msgs.msg as std_msgs

from teleop_msgs.msg import Scene as SceneRos

# if launching by script from this dir, add:
# sys.path.append("..")
from srcmodules.Scenes import Scene, SceneCoppeliaInterface
from srcmodules.Actions import Actions
from srcmodules.RobotActions import RobotActions,act
from agent_nodes import g2i

from py_trees_ros import subscribers
from teleop_msgs.msg import Intent
from teleop_msgs.msg import Gestures as GesturesRos
from teleop_msgs.srv import G2I

from agent_nodes.g2i import G2IRosNode

import py_trees.console as console

def shutdown(behaviour_tree):
    behaviour_tree.interrupt()

'''
Updates scene to blackboard
'''
class UpdateScene(subscribers.ToBlackboard):
    def __init__(self, topic_name="/tree/scene_in", qos_profile=py_trees_ros.utilities.qos_profile_unlatched(), name=py_trees.common.Name.AUTO_GENERATED, threshold=30.0):
        super(UpdateScene, self).__init__(name=name, topic_name=topic_name, topic_type=SceneRos, qos_profile=qos_profile, blackboard_variables={"scene": None}, clearing_policy=py_trees.common.ClearingPolicy.NEVER)

        self.blackboard.register_key(key="gestures", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="scene", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="some_var_1", access=py_trees.common.Access.WRITE)

        self.blackboard.register_key(key="robotic_sequence", access=py_trees.common.Access.WRITE)

        self.blackboard.scene = SceneRos()
        self.blackboard.some_var_1 = False
        self.blackboard.robotic_sequence = []

        self.threshold = threshold

    def update(self):
        return py_trees.common.Status.SUCCESS
        self.logger.debug("%s.update()" % self.__class__.__name__)
        status = super(UpdateScene, self).update()
        if status != py_trees.common.Status.RUNNING:
            if self.blackboard.scene.robot_attached_str != "":
                self.blackboard.some_var_1 = True
                console.logwarn(console.red + f"{self.name}: gripper attached!" + console.reset)
            else:
                self.blackboard.some_var_1 = False

        self.feedback_message = f"updated: {status}"
        return status

class UpdateGestures(subscribers.ToBlackboard):
    def __init__(self, topic_name="/tree/gestures_in", qos_profile=py_trees_ros.utilities.qos_profile_unlatched(), name=py_trees.common.Name.AUTO_GENERATED, threshold=30.0):
        super(UpdateGestures, self).__init__(name=name, topic_name=topic_name, topic_type=GesturesRos, qos_profile=qos_profile, blackboard_variables={"gestures": None}, clearing_policy=py_trees.common.ClearingPolicy.NEVER)

        self.blackboard.gestures = GesturesRos()

        self.threshold = threshold

    def update(self):
        return py_trees.common.Status.SUCCESS
        self.logger.debug("%s.update()" % self.__class__.__name__)
        status = super(UpdateGestures, self).update()

        self.feedback_message = f"updated: {status}"
        return status

class GenerateIntent(py_trees.behaviour.Behaviour):
    def __init__(self, name, rosnode):
        super(GenerateIntent, self).__init__(name=name)
        self.rosnode = rosnode

        self.g2i_tester = G2IRosNode(init_node=False, inference_type='NN', load_model='M3v10_D6.pkl', ignore_unfeasible=True)

    def setup(self, **kwargs):
        self.g2i_service_client = self.rosnode.create_client(G2I, 'g2i')
        while not self.g2i_service_client.wait_for_service(timeout_sec=1.0):
            print('service not available, waiting again...')
        self.req = G2I.Request(gestures=self.blackboard.gestures, scene=self.blackboard.scene)

        self.blackboard.register_key(key="target_action", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="target_object", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="auxiliary_parameters", access=py_trees.common.Access.WRITE)


    def update(self):
        self.logger.debug("%s.update()" % self.__class__.__name__)

        self.req = G2I.Request(gestures=self.blackboard.gestures, scene=self.blackboard.scene)
        #self.future = self.g2i_service_client.call_async(self.req)
        #print("Waiting for G2I")
        #rclpy.spin_until_future_complete(self.rosnode, self.future)
        #print("G2I Done!")


        response = self.g2i_tester.G2I_service_callback(self.req, G2I.Response())

        #target_action, target_object = self.g2i_tester.predict_with_list_of_gestures(self.blackboard.scene, self.blackboard.gestures, self.blackboard.scene.focus_point, scene_def_id=8)
        intent = response.intent
        self.blackboard.target_action = intent.target_action
        self.blackboard.target_object = intent.target_object
        self.blackboard.auxiliary_parameters = intent.auxiliary_parameters


        #self.blackboard.target_action = target_action
        #self.blackboard.target_object = target_object
        #self.blackboard.auxiliary_parameters = ""

        self.feedback_message = f"intent {intent.target_action}, {intent.target_object}, {intent.auxiliary_parameters} generated"
        return py_trees.common.Status.SUCCESS

class ExecuteTA(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(ExecuteTA, self).__init__(name=name)
        #global g2i
        #self.g2i = g2i.rosnode

    def update(self):
        self.logger.debug("%s.update()" % self.__class__.__name__)
        if self.blackboard.target_action != "":
            self.feedback_message = f"[1 Execute TA] {py_trees.common.Status.SUCCESS}! {self.blackboard.target_action},{self.blackboard.target_object}"
            self.blackboard.robotic_sequence.append(Intent(target_action=self.blackboard.target_action, target_object=self.blackboard.target_object, auxiliary_parameters=""))
        else:
            self.feedback_message = f"no action"
        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status):
        pass#self.feedback_message = "cleared"

class DeleteIntent(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(DeleteIntent, self).__init__(name=name)

    def update(self):
        self.logger.debug("%s.update()" % self.__class__.__name__)
        self.blackboard.target_action = ""
        self.blackboard.target_object = ""
        self.blackboard.auxiliary_parameters = ""
        return py_trees.common.Status.SUCCESS

# might be replaced with py_trees.blackboard.CheckBlackboardVariable(name="nnn",
# variable_name='', expected_value=False)
class TONotEqDrawer(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(TONotEqDrawer, self).__init__(name=name)

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

        self.feedback_message = "TO is not in object names!!!"
        return py_trees.common.Status.FAILURE

class HoldingPrecondition(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(HoldingPrecondition, self).__init__(name=name)
        self.blackboard.holding_precondition = False

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
        #global g2i
        #self.g2i = g2i.rosnode

    def update(self):
        self.blackboard.holding_precondition = False
        self.logger.debug("%s.update()" % self.__class__.__name__)

        self.feedback_message = f"[2 PickTO] ta=pick_up, to={self.blackboard.held_object}"
        self.blackboard.robotic_sequence.append(Intent(target_action='pick_up', target_object=self.blackboard.held_object, auxiliary_parameters=""))
        return py_trees.common.Status.SUCCESS

class DrawerStateNotInPrecondition(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(DrawerStateNotInPrecondition, self).__init__(name=name)

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
                self.feedback_message = f"drawer opened?: {self.blackboard.scene.objects[n].opened}, target_action is open?: {self.blackboard.target_action == 'open'}, fixing conditions"
                return py_trees.common.Status.FAILURE # we need to fix them

        self.feedback_message = f"drawer opened?: {self.blackboard.scene.objects[n].opened}, target_action is open?: {self.blackboard.target_action == 'open'}, not fixing conditions"

        return py_trees.common.Status.SUCCESS

class HandEmpty(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(HandEmpty, self).__init__(name=name)

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
        #global g2i
        #self.g2i = g2i.rosnode

    def setup(self, **kwargs):
        self.blackboard.register_key(key="held_object", access=py_trees.common.Access.WRITE)

    def update(self):
        self.logger.debug("%s.update()" % self.__class__.__name__)

        self.blackboard.held_object = self.blackboard.scene.robot_attached_str
        self.feedback_message = "[4 PlaceObject] ta=place, target_object=none"
        self.blackboard.robotic_sequence.append(Intent(target_action='place', target_object="", auxiliary_parameters=""))
        self.blackboard.holding_precondition = True

        return py_trees.common.Status.SUCCESS

class ToggleDrawer(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(ToggleDrawer, self).__init__(name=name)
        #global g2i
        #self.g2i = g2i.rosnode

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
            self.blackboard.robotic_sequence.append(Intent(target_action='close', target_object=o_name, auxiliary_parameters=""))
            self.feedback_message = f"[3 Toggle Drawer] ta=close, to={o_name}"
        else:
            self.blackboard.robotic_sequence.append(Intent(target_action='open', target_object=o_name, auxiliary_parameters=""))
            self.feedback_message = f"[3 Toggle Drawer] ta=opening, to={o_name}"

        return py_trees.common.Status.SUCCESS

class HoldingPrecondition(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(HoldingPrecondition, self).__init__(name=name)

    def setup(self, **kwargs):
        self.blackboard.register_key(key="holding_precondition", access=py_trees.common.Access.WRITE)
        self.blackboard.holding_precondition = False

    def update(self):
        self.logger.debug("%s.update()" % self.__class__.__name__)
        if self.blackboard.holding_precondition:
            return py_trees.common.Status.FAILURE
            self.blackboard.holding_precondition = False
        else:
            return py_trees.common.Status.SUCCESS

def create_tree(rosnode):
    ''' BH tree final
    '''
    # LVL 0 - root
    root = py_trees.composites.Sequence("root", memory=True)
    # LVL 1
    update_scene = UpdateScene(name="Scene2BB")
    update_gestures = UpdateGestures(name="Gestures2BB")
    generate_intent = GenerateIntent(name="GenIntent", rosnode=rosnode)

    seq_lvl1 = py_trees.composites.Sequence("seq_lvl1", memory=True)
    # LVL 2
    to_is_drawer_q = py_trees.composites.Selector("to is drawer?", memory=True)
    holding_precondition_q = py_trees.composites.Selector("holding precond.?", memory=True)
    execute_ta = ExecuteTA("execute_ta")
    delete_intent = DeleteIntent("delete_intent")
    # LVL 3
    to_not_eq_drawer = TONotEqDrawer("to_not_eq_drawer")
    drawer_precond_q = py_trees.composites.Selector("drawer precond.?", memory=True)

    holding_precondition = HoldingPrecondition('holding_precondition')
    pick_to = PickTO('pick_to')

    # LVL 4
    drawer_state_not_in_preconditions = DrawerStateNotInPrecondition('drawer_state_not_in_preconditions')
    fix_drawer_precond = py_trees.composites.Sequence("fix_drawer_precond", memory=True)
    toggle_drawer = ToggleDrawer("toggle_drawer")

    # --- LVL 5 ----------------------------------------------

    place_q = py_trees.composites.Selector("place first?", memory=True)

    # --- LVL 6 ----------------------------------------------
    hand_empty = HandEmpty("hand_empty")
    place_object = PlaceObject("place_object")

    # --- add blackboard sharing -----------------------------

    generate_intent.blackboard = update_scene.blackboard
    execute_ta.blackboard = update_scene.blackboard
    delete_intent.blackboard = update_scene.blackboard
    to_not_eq_drawer.blackboard = update_scene.blackboard
    holding_precondition.blackboard = update_scene.blackboard
    pick_to.blackboard = update_scene.blackboard
    drawer_state_not_in_preconditions.blackboard = update_scene.blackboard
    toggle_drawer.blackboard = update_scene.blackboard
    hand_empty.blackboard = update_scene.blackboard
    place_object.blackboard = update_scene.blackboard

    # --- tree -----------------------------------------------

    root.add_children([update_scene, update_gestures, generate_intent, seq_lvl1])
    seq_lvl1.add_children([to_is_drawer_q, holding_precondition_q, execute_ta, delete_intent])
    to_is_drawer_q.add_children([to_not_eq_drawer, drawer_precond_q])

    holding_precondition_q.add_children([holding_precondition, pick_to])

    drawer_precond_q.add_children([drawer_state_not_in_preconditions, fix_drawer_precond])

    fix_drawer_precond.add_children([place_q,toggle_drawer])

    place_q.add_children([hand_empty, place_object])

    return root


class BTreeHandler:
    def __init__(self, rosnode, root_func=create_tree):
        self.root = root_func(rosnode)
        self.behaviour_tree = py_trees_ros.trees.BehaviourTree(root=self.root, unicode_tree_debug=True)

        try:
            self.behaviour_tree.setup(timeout=60.0)
        except py_trees_ros.exceptions.TimedOutError as e:
            console.logerror(console.red + "failed to setup the tree, aborting [{}]".format(str(e)) + console.reset)
            self.behaviour_tree.shutdown()
            rclpy.shutdown()
            sys.exit(1)
        except KeyboardInterrupt:
            # not a warning, nor error, usually a user-initiated shutdown
            console.logerror("tree setup interrupted")
            self.behaviour_tree.shutdown()
            rclpy.shutdown()
            sys.exit(1)

    def predict(self, gestures, scene):
        self.behaviour_tree.root.children[0].blackboard.gestures = gestures
        self.behaviour_tree.root.children[0].blackboard.scene = scene

        self.behaviour_tree.root.children[0].blackboard.robotic_sequence = []

        self.behaviour_tree.tick()

        return self.behaviour_tree.root.children[0].blackboard.robotic_sequence



'''
Januray 2023 - More general approach
'''

class CustomToBlackboard(subscribers.ToBlackboard):
    def __init__(self, name, topic_name, topic_type, bb_var_name):
        super(UpdateScene, self).__init__(name=name, topic_name=topic_name, topic_type=topic_type, qos_profile=py_trees_ros.utilities.qos_profile_unlatched(), blackboard_variables={bb_var_name: topic_type()})
        self.blackboard.register_key(key=bb_var_name, access=py_trees.common.Access.WRITE)
        self.threshold = 30.0
    def update(self):
        self.logger.debug("%s.update()" % self.__class__.__name__)
        status = super(CustomToBlackboard, self).update()
        self.feedback_message = f"bb updated: {status}"
        return status


class CustomBehaviourCallService(py_trees.behaviour.Behaviour):
    def __init__(self, name, rosnode, service_topic, service_type, bb_save_var_name, bb_req_var_name):
        super(GenerateIntent, self).__init__(name=name)

        self.rosnode = rosnode
        self.service_topic = service_topic
        self.service_type = service_type
        self.bb_save_var_name = bb_save_var_name

    def setup(self, **kwargs):
        self.service_client = self.rosnode.create_client(self.service_type, self.service_topic)
        while not self.service_client.wait_for_service(timeout_sec=1.0):
            print('service not available, waiting again...')
        self.req = self.service_type.Request()
        self.blackboard.register_key(key=bb_save_var_name, access=py_trees.common.Access.WRITE)

    def update(self):
        self.logger.debug("%s.update()" % self.__class__.__name__)

        self.req = bb_req_var_name
        self.future = self.g2i_service_client.call_async(self.req)
        rclpy.spin_until_future_complete(self.rosnode, self.future)

        setattr(self.blackboard, self.bb_save_var_name, self.future.result())

        self.feedback_message = f"Service {self.service_topic} finished"
        return py_trees.common.Status.SUCCESS


def create_tree_2(rosnode):
    ''' BH tree final
    '''
    # LVL 0 - root
    root = py_trees.composites.Sequence("root")
    # LVL 1
    update_scene = CustomToBlackboard(name="Scene2BB", topic_name="/tree/scene_in", topic_type=SceneRos, bb_var_name="scene")
    update_gestures = UpdateGestures(name="Gestures2BB", topic_name="/tree/gestures_in", topic_type=GesturesRos, bb_var_name="gestures")
    generate_intent = CustomBehaviourCallService(name="GenIntent", rosnode=rosnode, service_topic='g2i', service_type=G2I, bb_save_var_name=intent, bb_req_var_name="g2i")

    generate_intent = GenerateIntent(name="GenIntent")

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

    # --- add blackboard sharing -----------------------------

    generate_intent.blackboard = update_scene.blackboard
    execute_ta.blackboard = update_scene.blackboard
    delete_intent.blackboard = update_scene.blackboard
    to_not_eq_drawer.blackboard = update_scene.blackboard
    holding_precondition.blackboard = update_scene.blackboard
    pick_to.blackboard = update_scene.blackboard
    drawer_state_not_in_preconditions.blackboard = update_scene.blackboard
    toggle_drawer.blackboard = update_scene.blackboard
    hand_empty.blackboard = update_scene.blackboard
    place_object.blackboard = update_scene.blackboard

    # --- tree -----------------------------------------------

    root.add_children([update_scene, update_gestures, generate_intent, seq_lvl1])
    seq_lvl1.add_children([to_is_drawer_q, holding_precondition_q, execute_ta, delete_intent])
    to_is_drawer_q.add_children([to_not_eq_drawer, drawer_precond_q])

    holding_precondition_q.add_children([holding_precondition, pick_to])

    drawer_precond_q.add_children([drawer_state_not_in_preconditions, fix_drawer_precond])

    fix_drawer_precond.add_children([place_q,toggle_drawer])

    place_q.add_children([hand_empty, place_object])

    return root
