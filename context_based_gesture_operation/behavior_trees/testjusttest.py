import rospy
from context_based_gesture_operation.srv import G2I, G2IResponse
from context_based_gesture_operation.msg import Intent
from context_based_gesture_operation.msg import Gestures
from context_based_gesture_operation.msg import Scene

rospy.init_node("sadiansidona")

try:
    g2i = rospy.ServiceProxy('g2i', G2I)
    response = g2i(gestures=Gestures(), scene=Scene())
except rospy.ServiceException as e:
    print("Service call failed: %s"%e)

response
