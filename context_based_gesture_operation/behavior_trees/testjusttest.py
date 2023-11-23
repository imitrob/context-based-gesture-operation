import rospy
from teleop_msgs.srv import G2I, G2IResponse
from teleop_msgs.msg import Intent
from teleop_msgs.msg import Gestures
from teleop_msgs.msg import Scene

rospy.init_node("sadiansidona")

try:
    g2i = rospy.ServiceProxy('g2i', G2I)
    response = g2i(gestures=Gestures(), scene=Scene())
except rospy.ServiceException as e:
    print("Service call failed: %s"%e)

response
