'''
>>> import importlib
>>> import srcmodules.Scenes; importlib.reload(srcmodules.Scenes)
>>> import sys; sys.path.append("..")
'''
import rclpy
from rclpy.node import Node

from context_based_gesture_operation.srv import G2I
from context_based_gesture_operation.msg import Intent
from context_based_gesture_operation.msg import Scene as SceneRos
from context_based_gesture_operation.msg import Gestures as GesturesRos
import numpy as np
from srcmodules.Scenes import Scene
from srcmodules.Gestures import Gestures
from srcmodules.Actions import Actions
from srcmodules.SceneFieldFeatures import SceneFieldFeatures

'''
>>> sceneros = SceneRos()
>>> sceneros.user = 'Jan'
>>> s = Scene(init='from_ros', import_data=sceneros)
'''

class G2IRosNode(Node):
    def __init__(self, G=Gestures.G, A=Actions.A, init_node=False, type='G2I_callback_direct_mapping'):
        super().__init__("G2IServiceNode")
        self.create_service(G2I, '/g2i', getattr(self,type))

        self.G = Gestures.G = G
        self.A = Actions.A = A

    def G2I_callback_direct_mapping(self, msg):
        ''' Direct mapping approach '''

        s = Scene(init='from_ros', import_data=msg.scene)
        focus_point = msg.scene.focus_point
        g = msg.gestures.probabilities.data

        # consturct the transition matrix
        l = np.max([len(self.G), len(self.A)])
        T = np.diag(np.ones(l))
        # match gestures'n'actions -> however they should be the same length
        g_ext = np.zeros(len(self.A))
        g_ext[0:len(g)] = g

        aid = np.argmax(np.dot(T, g_ext))
        a = self.A[aid]

        #SceneFieldFeatures.eeff__feature(s.object_positions, np.array([0,0,0]))
        nojb = np.argmax(SceneFieldFeatures.eeff__feature(s.object_positions,np.array(focus_point)))

        i = Intent()
        i.target_action = a
        i.target_object = s.O[nojb]
        print(f"intent {i.target_action}, {i.target_object}")
        return G2IResponse(i)

def init():
    global rosnode
    rosnode = None
