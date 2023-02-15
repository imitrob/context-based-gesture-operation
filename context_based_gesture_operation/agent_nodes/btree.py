import sys; sys.path.append("..")

from context_based_gesture_operation.msg import Scene as SceneRos
from context_based_gesture_operation.msg import Gestures as GesturesRos
from context_based_gesture_operation.srv import BTreeSingleCall

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
    
