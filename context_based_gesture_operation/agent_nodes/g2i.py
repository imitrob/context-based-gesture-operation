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
try:
    import srcmodules
except ModuleNotFoundError:
    import sys; sys.path.append("..")
from srcmodules.Scenes import Scene
from srcmodules.Gestures import Gestures
from srcmodules.Actions import Actions
from srcmodules.SceneFieldFeatures import SceneFieldFeatures
from srcmodules.Objects import Object

from srcmodules.nnwrapper import NNWrapper

try:
    import theano
except:
    theano = None

# TMP
import sys, os; sys.path.append(f"/home/petr/ros2_ws/src/teleop_gesture_toolbox/teleop_gesture_toolbox")
sys.path.append(f"/home/petr/ros2_ws/src/teleop_gesture_toolbox/teleop_gesture_toolbox/leapmotion")
import gesture_classification.gestures_lib as gl; gl.init(silent=True)

'''
>>> sceneros = SceneRos()
>>> sceneros.user = 'Jan'
>>> s = Scene(init='from_ros', import_data=sceneros)
'''

class G2IRosNode(Node):
    def __init__(self, init_node=False, inference_type='NN', load_model='M3v8_D4_1.pkl', ignore_unfeasible=False):
        super().__init__("G2IServiceNode")
        self.create_service(G2I, '/g2i', self.G2I_service_callback)

        self.G = Gestures.G
        self.A = Actions.A

        self.inference_type = inference_type
        if inference_type == 'NN':
            self.sampler = PyMC3_Sample(load_model)
            self.A = self.sampler.nn.args['A']
            self.G = self.sampler.nn.args['G']
            self.Otypes = self.sampler.nn.args['Otypes']
        elif inference_type == '1to1':
            self.sampler = OneToOne_Sample()
        else: # TODO
            self.sampler = MultiDim_Sample()

        self.ignore_unfeasible = ignore_unfeasible
        self.load_model = load_model
        self.scene_def_id = int(load_model.split("_")[0][3:])

    def gl_to_g2i(self, g):
        '''
        Tester:
        import numpy as np
        Gset1 = ['grab', 'pinch', 'point', 'two', 'three', 'four', 'five', 'thumbsup', 'swipe_down', 'swipe_front_right', 'swipe_left', 'swipe_up', 'no_gesture']
        Gset2 = ['swipe_up', 'swipe_left', 'swipe_down', 'swipe_right', 'five', 'grab', 'thumbsup', 'rotate', 'point']
        g = np.zeros(13)
        g[-3] = 1.0
        '''
        Gset1 = gl.gd.Gs
        Gset2 = self.G

        g_g2i = np.zeros(np.array(len(Gset2)))
        for n, g_ in enumerate(Gset2):
            #### ! TMP !
            if g_ == 'swipe_right' and 'swipe_front_right' in Gset1:
                g_g2i[n] = g[Gset1.index('swipe_front_right')]
            if g_ in Gset1:
                g_g2i[n] = g[Gset1.index(g_)]

        g_g2i
        if np.allclose(g_g2i, np.zeros(np.array(len(Gset2))) ): return None

        return g_g2i

    def G2I_service_callback(self, request, response):
        ''' G2I for ROS2 service
        Parameters:
            request.scene (SceneRos)
            request.gestures.probabilities.data (Float[]) Gesture probabilities
        Returns:
            response.intent.target_action (string)
            response.intent.target_object (string)
        '''
        # SceneRos to Python Scene object
        s = Scene(init='from_ros', import_data=request.scene)
        focus_point = request.scene.focus_point
        g_gl = request.gestures.probabilities.data

        g = self.gl_to_g2i(g_gl)
        if g is None:
            print("[ERROR] No gesture in G2I model")
            return response

        ## focus point to target object
        id_obj = np.argmin(np.linalg.norm(np.array(s.object_positions_real) - np.array(focus_point), axis=1))
        name_obj = s.objects[id_obj].name
        id_g = np.argmax(g)

        print(self.G, id_g)

        name_g = self.G[id_g]

        a, o = self.predict_with_scene_gesture_and_target_object(s, name_g, name_obj, scene_def_id=self.scene_def_id)

        #SceneFieldFeatures.eeff__feature(s.object_positions, np.array([0,0,0]))
        nojb = np.argmax(SceneFieldFeatures.eeff__feature(s.object_positions_real,np.array(focus_point)))

        response.intent.target_action = a
        response.intent.target_object = s.O[id_obj]
        print(f"intent {response.intent.target_action}, {response.intent.target_object}")

        return response

    def predict_with_scene_gesture_and_target_object(self,s, gesture, target_object, scene_def_id):
        '''
        Parameters:
            scene - Scene() object - srcmodules.Scenes.Scene
            gesture - String - Gesture name performed
            target_object - String - object name in the scene\
        Returns:
            (target_action, target_object) - (String, String) - Note: target_object is same as on input
        '''

        focus_point = getattr(s, target_object).position_real
        return self.predict_with_scene_gesture_and_focus_point(s, gesture, focus_point, scene_def_id)

    def predict_with_scene_gesture_and_focus_point(self,s, gesture, focus_point, scene_def_id):
        target_object = s.O[s.get_closest_object(focus_point)]

        gestures = np.zeros((len(self.G)))
        gesture_id = self.G.index(gesture)
        gestures[gesture_id] = 1

        user_dep = True
        X = np.zeros([70])

        obs = s.scene_to_observation(type=scene_def_id, focus_point=focus_point)


        if not user_dep:
            ll = len(gestures) + len(obs)
            X[0:ll] = [*gestures, *obs]
            print(f"v2")
        else:
            ll = len(gestures) + 1 + len(obs)
            X[0:ll] = [*gestures, s.u.selected_id, *obs]
            print(f"v3")
        print(X)
        inference_probs = self.sampler.sample(X)[0]

        action_id = np.argmax(inference_probs)
        print("[G2I Result] action_id: ", self.A[action_id])

        possible_actions = Actions.get_possible_actions(s, ignore_location=True)
        possible_actions_for_target_object = [(a) for a in possible_actions if a[1]==target_object]

        if self.ignore_unfeasible:
            return (self.A[action_id], target_object)

        ## Check if possible to do
        print(f"Action probs: {inference_probs}")
        print("possible_actions_for_target_object", possible_actions_for_target_object)
        if [(a) for a in possible_actions_for_target_object if a[0]==self.A[action_id]] == []:
            print("Action is not feasible to do! Try to return second most probable action!")
            inference_probs[action_id] = 0
            action_id_2 = np.argmax(inference_probs)
            if [(a) for a in possible_actions_for_target_object if a[0]==self.A[action_id_2]] == []:
                return ("", "")
            else:
                return (self.A[action_id_2], target_object)
        return (self.A[action_id], target_object)



    def predict_with_list_of_gestures(self, s, gestures, focus_point, scene_def_id):
        ''' Choses only the last gesture performed
        '''
        if len(s.O) == 0:
            print("No objects on the scene -> returning")
            return ("", "")
        ## focus point to target object
        id_obj = np.argmin(np.linalg.norm(np.array(s.object_positions_real) - np.array(focus_point), axis=1))
        target_object = s.objects[id_obj].name

        # Discard no-meaning gestures
        gque = [g[1] for g in gestures]
        while 'nothing_dyn' in gque:
            gque.remove('nothing_dyn')
        # Choose the last performed gesture
        ## TODO: This approach may be improved in the future
        gesture = gque[-1]

        return self.predict_with_scene_gesture_and_target_object(s, gesture, target_object, scene_def_id)



def init():
    global rosnode
    rosnode = None

'''
PyMC3 Sampler
'''
class PyMC3_Sample():
    def __init__(self, load_model):
        self.X_train = None
        self.approx = None
        self.neural_network = None
        self.sample_proba = None
        self._sample_proba = None

        assert theano is not None
        assert load_model != ""
        network_path = "/home/petr/ros2_ws/src/context_based_gesture_operation/context_based_gesture_operation/trained_networks/"
        self.nn = NNWrapper.load_network(network_path, name=load_model)
        self.init(self.nn)


    def sample(self, data):
        return self.sample_proba([data.data],100).mean(0)

    def init(self, nn):
        self.X_train = nn.X_train
        self.approx = nn.approx
        self.neural_network = nn.neural_network
        x = theano.tensor.matrix("X")
        n = theano.tensor.iscalar("n")
        x.tag.test_value = np.empty_like(self.X_train[:10])
        n.tag.test_value = 100
        self._sample_proba = self.approx.sample_node(
            self.neural_network.out.distribution.p, size=n, more_replacements={self.neural_network["nn_input"]: x}
        )
        self.sample_proba = theano.function([x, n], self._sample_proba)



class OneToOne_Sample():
    def __init__(self):
        # consturct the transition matrix
        self.T =    np.array([[ 1,  .0, .0,  .0,  0, .0,   .0,  .0,  .0], #, .0, .0], # move_up
                              [.0,   1, .0,  .0, .0, .0,    0,   0,  .0], #, .0, .0], # move_left
                              [.0,  .0,  1,  .0,  0, .0,   .0,   0,  .0], #, .0, .0], # move_down
                              [.0,  .0, .0,   1,  0, .0,   .0,  .0,  .0], #, .0, .0], # move_right
                              [.0,  .0, .0,  .0,  1, .0,   .0,  .0,  .0], #, .0, .0], # put
                              [.0,  .0, .0,  .0, .0, .0,   .0,  .0,  .0], #,  1, .0], # put_on
                              [.0,  .0, .0,  .0, .0, .0,    0,   1,  .0], #, .0, .0], # pour
                              [ 0,  .0, .0,  .0, .0,  1,    0,  .0,  .0], #, .0, .0], # pick_up
                              [.0,  .0, .0,  .0, .0, .0,    1,  .0,  .0], #, .0, .0], # place
                              [.0,  .0, .0,  .0, .0, .0,   .0,  .0,   1], #, .0, .0], # open
                              [.0,  .0, .0,  .0, .0, .0,   .0,  .0,  .0]]) #, .0,  1]])# close
    def sample(self, data):
        return np.dot(self.T, data.data[0:9])



class MultiDim_Sample():
    ''' TODO
    '''
    def __init__(self, G,A):
        self.G = G
        self.A = A
        # OBJECT == CUP -- empty
        self.T = np.zeros((2,3,9,11))

        self.T[0,0]=np.array([[ 1,  .0, .0,  .0,  0, .0,   .0,  .0,  .0], # move_up
                           [.0,   1, .0,  .0, .0, .0,    0,   0,  .0], # move_left
                           [.0,  .0,  1,  .0,  0, .0,   .0,   0,  .0], # move_down
                           [.0,  .0, .0,   1,  0, .0,   .0,  .0,  .0], # move_right
                           [.0,  .0, .0,  .0, .0, .0,   .0,  .0,  .0], # put
                           [.0,  .0, .0,  .0, .0, .0,   .0,  .0,  .0], # put_on
                           [.0,  .0, .0,  .0, .0, .0,    0,   1,  .0], # pour
                           [ 0,  .0, .0,  .0, .0,  1,    0,  .0,  .0], # pick_up
                           [.0,  .0, .0,  .0, .0, .0,    1,  .0,  .0], # place
                           [.0,  .0, .0,  .0, .0, .0,   .0,  .0,  .0], # open
                           [.0,  .0, .0,  .0, .0, .0,   .0,  .0,  .0]]).T# close

        # OBJECT == drawer -- closed
        # gestures:                     up,left,dwn,rght,fiv,grab,thum,rota,poin
        self.T[0,1]=np.array([[ 0,  .0, .0,  .0,  0, .0,   .0,  .0,  .0], # move_up
                           [.0,   0, .0,  .0,  0, .0,    0,   0,  .0], # move_left
                           [.0,  .0,  0,  .0,  0, .0,   .0,   0,  .0], # move_down
                           [.0,  .0, .0,   0,  0, .0,   .0,  .0,  .0], # move_right
                           [.0,  .0,  0,  .0,  0, .0,   .0,  .0,  .0], # put
                           [.0,  .0,  1,  .0, .0, .0,   .0,  .0,   0], # put_on
                           [.0,  .0, .0,  .0, .0, .0,   .0,  .0,  .0], # pour
                           [.0,  .0, .0,  .0, .0, .0,   .0,  .0,  .0], # pick_up
                           [.0,  .0, .0,  .0, .0, .0,    0,  .0,   1], # place
                           [.0,  .0, .0,  .0, .0,  1,    0,  .0,  .0], # open
                           [.0,  .0, .0,  .0, .0, .0,    0,  .0,  .0]]).T# close
        # OBJECT == object/box --
        # gestures:                     up,left,dwn,rght,fiv,grab,thum,rota,poin
        self.T[0,2]=np.array([[1,  .0, .0,  .0,  0, .0,   .0,  .0,  .0], # move_up
                           [.0,   1, .0,  .0, .0, .0,    0,   0,  .0], # move_left
                           [.0,  .0,  1,  .0,  0, .0,   .0,   0,  .0], # move_down
                           [.0,  .0, .0,   1,  0, .0,   .0,  .0,  .0], # move_right
                           [.0,  .0, .0,  .0,  0,  0,   .0,  .0,  .0], # put
                           [.0,  .0, .0,  .0,  1,  0,   .0,  .0,  .0], # put_on
                           [.0,  .0, .0,  .0, .0, .0,    0,   0,  .0], # pour
                           [ 0,  .0, .0,  .0, .0,  1,    0,  .0,  .0], # pick_up
                           [.0,  .0, .0,  .0, .0, .0,    1,  .0,  .0], # place
                           [.0,  .0, .0,  .0, .0, .0,   .0,  .0,  .0], # open
                           [.0,  .0, .0,  .0, .0, .0,   .0,  .0,  .0]]).T# close

        # OBJECT == CUP -- full
        self.T[1,0]=np.array([[ 1,  .0, .0,  .0,  0, .0,   .0,  .0,  .0], # move_up
                           [.0,   1, .0,  .0, .0, .0,    0,   0,  .0], # move_left
                           [.0,  .0,  1,  .0,  0, .0,   .0,   0,  .0], # move_down
                           [.0,  .0, .0,   1,  0, .0,   .0,  .0,  .0], # move_right
                           [.0,  .0, .0,  .0, .0, .0,   .0,  .0,  .0], # put
                           [.0,  .0, .0,  .0, .0, .0,   .0,  .0,  .0], # put_on
                           [.0,  .0, .0,  .0, .0, .0,    0,   1,  .0], # pour
                           [ 0,  .0, .0,  .0, .0,  1,    0,  .0,  .0], # pick_up
                           [.0,  .0, .0,  .0, .0, .0,    1,  .0,  .0], # place
                           [.0,  .0, .0,  .0, .0, .0,   .0,  .0,  .0], # open
                           [.0,  .0, .0,  .0, .0, .0,   .0,  .0,  .0]]).T# close

        # OBJECT == drawer -- opened
        # gestures:                     up,left,dwn,rght,fiv,grab,thum,rota,poin
        self.T[1,1]=np.array([[ 0,  .0, .0,  .0,  0, .0,   .0,  .0,  .0], # move_up
                           [.0,   0, .0,  .0,  0, .0,    0,   0,  .0], # move_left
                           [.0,  .0,  0,  .0,  0, .0,   .0,   0,  .0], # move_down
                           [.0,  .0, .0,   0,  0, .0,   .0,  .0,  .0], # move_right
                           [.0,  .0,  0,  .0,  0, .0,   .0,  .0,  .0], # put
                           [.0,  .0,  1,  .0, .0, .0,   .0,  .0,   0], # put_on
                           [.0,  .0, .0,  .0, .0, .0,   .0,  .0,  .0], # pour
                           [.0,  .0, .0,  .0, .0, .0,   .0,  .0,  .0], # pick_up
                           [.0,  .0, .0,  .0, .0, .0,    0,  .0,   1], # place
                           [.0,  .0, .0,  .0, .0,  1,    0,  .0,  .0], # open
                           [.0,  .0, .0,  .0, .0, .0,    0,  .0,  .0]]).T# close
        # OBJECT == object/box --
        # gestures:                     up,left,dwn,rght,fiv,grab,thum,rota,poin
        self.T[1,2]=np.array([[1,  .0, .0,  .0,  0, .0,   .0,  .0,  .0], # move_up
                           [.0,   1, .0,  .0, .0, .0,    0,   0,  .0], # move_left
                           [.0,  .0,  1,  .0,  0, .0,   .0,   0,  .0], # move_down
                           [.0,  .0, .0,   1,  0, .0,   .0,  .0,  .0], # move_right
                           [.0,  .0, .0,  .0,  0,  0,   .0,  .0,  .0], # put
                           [.0,  .0, .0,  .0,  1,  0,   .0,  .0,  .0], # put_on
                           [.0,  .0, .0,  .0, .0, .0,    0,   0,  .0], # pour
                           [ 0,  .0, .0,  .0, .0,  1,    0,  .0,  .0], # pick_up
                           [.0,  .0, .0,  .0, .0, .0,    1,  .0,  .0], # place
                           [.0,  .0, .0,  .0, .0, .0,   .0,  .0,  .0], # open
                           [.0,  .0, .0,  .0, .0, .0,   .0,  .0,  .0]]).T# close



    def sample(self, s, gesture, target_object):

        obj = getattr(s, target_object)
        objtype   = obj.type
        objtypeid = Object.all_types.index(objtype)

        if objtypeid == 'drawer':
            state = obj.opened
        elif objtypeid == 'cup':
            state = obj.full
        else:
            state = 0

        return np.dot(self.T[state, objtypeid], gesture)




if __name__ == "__main__":
    Object.all_types = Otypes = ['cup', 'drawer', 'object']
    rclpy.init()
    g2i = G2IRosNode(init_node=False, inference_type='NN', load_model='M3v8_D4_1.pkl', ignore_unfeasible=True)

    rclpy.spin(g2i)
