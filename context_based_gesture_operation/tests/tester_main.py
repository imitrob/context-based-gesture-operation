
# Testing Scene to_ros and import from ros
import sys; sys.path.append("..")

from copy import deepcopy
from teleop_msgs.msg import Scene as SceneRos
from srcmodules.Scenes import Scene as Scene
from srcmodules.Actions import Actions as Actions
import numpy as np

def test_ros_converter():
    for i in range(10000):
        s = Scene(init='drawer,cup,cup,object,object')
        sr = SceneRos()
        sro = s.to_ros(sr)
        sro2 = deepcopy(sro)
        s2 = Scene(init='from_ros', import_data=sro2)

        if s != s2:
            print("!!!", i)
            s.info
            s2.info
            break

    for i in range(10000):
        s = Scene(init='drawer,cup,cup,object,object')
        sr = SceneRos()
        sro = s.to_ros(sr)
        sro2 = deepcopy(sro)
        s2 = Scene(init='from_ros', import_data=sro2)
        s2_dict = s2.to_dict()
        d3_dict = deepcopy(s2_dict)
        s3 = Scene(init='from_dict', import_data=d3_dict)

        if s != s3:
            print("!!!", i)
            s.info
            s2.info
            break


def test_actions():
    if True:
        import sys; sys.path.append("/home/petr/crow-base/src/teleop_gesture_toolbox/teleop_gesture_toolbox")
        from os_and_utils.visualizer_lib import ScenePlot
        import matplotlib
        import matplotlib.pyplot as plt
        #%matplotlib inline
        #matplotlib.use("Qt5Agg")

    ''' 1. '''
    s = Scene(init='', random=False)

    ScenePlot.scene_objects(s)
    ScenePlot.scene_objects(s)

    ''' eef going down '''
    assert np.allclose(s.r.eef_position_real, np.array([ 0.2, -0.3,  0.6]))
    assert np.allclose(s.r.eef_position, np.array([0, 0, 3]))
    assert Actions.do(s, ('move_down', ""))
    ScenePlot.scene_objects(s)
    assert np.allclose(s.r.eef_position_real, np.array([ 0.2, -0.3,  0.4]))
    assert np.allclose(s.r.eef_position, np.array([0, 0, 2]))
    assert Actions.do(s, ('move_down', ""))
    ScenePlot.scene_objects(s)
    assert np.allclose(s.r.eef_position_real, np.array([ 0.2, -0.3,  0.2]))
    assert np.allclose(s.r.eef_position, np.array([0, 0, 1]))
    assert Actions.do(s, ('move_down', ""))
    ScenePlot.scene_objects(s)
    assert np.allclose(s.r.eef_position_real, np.array([ 0.2, -0.3,  0. ]))
    assert np.allclose(s.r.eef_position, np.array([0, 0, 0]))
    assert Actions.do(s, ('move_down', "")) == False
    assert np.allclose(s.r.eef_position_real, np.array([ 0.2, -0.3,  0. ]))
    assert np.allclose(s.r.eef_position, np.array([0, 0, 0]))

    Actions.do(s, ('move_right', ""))
    ScenePlot.scene_objects(s)

    ''' 2. '''
    s = Scene(init='object', random=False)
    ScenePlot.scene_objects(s)

    assert Actions.do(s, ('pick_up', "object"))
    ScenePlot.scene_objects(s)

    assert Actions.do(s, ('place', "object"))
    ScenePlot.scene_objects(s)

    s = Scene(init='object,cup,drawer', random=False)
    ScenePlot.scene_objects(s)
    assert Actions.do(s, ('pick_up', "cup"))
    ScenePlot.scene_objects(s)

    assert Actions.do(s, ('place', "cup"))
    ScenePlot.scene_objects(s)

    ''' Cannot place when not holding anything '''
    assert not Actions.do(s, ('place', "cup"))

    ''' Cannot pick up drawer '''
    assert not Actions.do(s, ('pick_up', "drawer"))


if __name__ == '__main__':
    test_ros_converter()
    test_actions()
