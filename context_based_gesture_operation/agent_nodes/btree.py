#!/usr/bin/env python
import sys, os
path_from_ws = f'src/context_based_gesture_operation/context_based_gesture_operation/agent_nodes'
ws_dir = "/".join(os.path.abspath(__file__).split('/')[:-5])
path=f'/{ws_dir}/{path_from_ws}/'
os.chdir(path)
sys.path.append(path)
import sys; sys.path.append("..")

from teleop_msgs.msg import Scene as SceneRos
from teleop_msgs.msg import Gestures as GesturesRos
from teleop_msgs.srv import BTreeSingleCall

from srcmodules.Scenes import Scene
from srcmodules.Gestures import Gestures
from srcmodules.Actions import Actions
from srcmodules.Objects import Object
from srcmodules.SceneFieldFeatures import SceneFieldFeatures
from srcmodules import BTreeLib

import rclpy
from rclpy.node import Node
import time

class BTreeNode(Node):
    def __init__(self):
        super().__init__("G2IServiceNode")
        self.create_service(BTreeSingleCall, '/btree_onerun', self.BTree_onerun_service_callback)


        self.btree = BTreeLib.BTreeHandler(rosnode=self)

    def BTree_onerun_service_callback(self, request, response):
        '''
        '''
        print(f"Request: {request}")
        response.intent = self.btree.predict(request.gestures, request.scene)
        print(f"BTree response: {response}")
        return response

if __name__ == "__main__":
    Object.all_types = Otypes = ['cup', 'drawer', 'object']
    rclpy.init()
    btree = BTreeNode()

    rclpy.spin(btree)
