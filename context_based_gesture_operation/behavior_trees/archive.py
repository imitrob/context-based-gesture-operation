
scan2bb = py_trees_ros.subscribers.EventToBlackboard(
        name="Scan2BB",
        topic_name="/dashboard/scan",
        variable_name="event_scan_button"
    )

battery_check = py_trees.meta.success_is_failure(py_trees.composites.Selector)(name="Battery Emergency")

is_battery_ok = py_trees.blackboard.CheckBlackboardVariable(
    name="Battery Ok?",
    variable_name='battery_low_warning',
    expected_value=False
)
#
flash_led_strip = py_trees_ros.tutorials.behaviours.FlashLedStrip(
    name="Flash Red",
    colour="red")

is_scan_requested = py_trees.blackboard.CheckBlackboardVariable(
    name="Scan?",
    variable_name='event_scan_button',
    expected_value=True
)

is_scan_requested_two = py_trees.meta.success_is_running(py_trees.blackboard.CheckBlackboardVariable)(
    name="Scan?",
    variable_name='event_scan_button',
    expected_value=True
)

scanning = py_trees.composites.Parallel(name="Scanning", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

scan_rotate = py_trees_ros.actions.ActionClient(
    name="Rotate",
    action_namespace="/rotate",
    action_spec=py_trees_msgs.RotateAction,
    action_goal=py_trees_msgs.RotateGoal(),
    override_feedback_message_on_running="rotating"
)


scan_pause = py_trees.timers.Timer("Pause", duration=3.0)

idle = py_trees.behaviours.Running(name="Idle")

class PrinterTesterMessages(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(PrinterTesterMessages, self).__init__(name=name)

    def setup(self, timeout):
        self.publisher = rospy.Publisher("/out_str", std_msgs.String, queue_size=10, latch=True)
        self.feedback_message = "setup"
        return True

    def update(self):
        self.logger.debug("%s.update()" % self.__class__.__name__)
        blackboard = py_trees.blackboard.Blackboard()
        pubstr = str(blackboard.tester_bool)
        self.publisher.publish(std_msgs.String(pubstr))
        self.feedback_message = "flashing {0}".format(pubstr)
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.publisher.publish(std_msgs.String(""))
        self.feedback_message = "cleared"
class TesterToBlackboard(subscribers.ToBlackboard):
    def __init__(self, name):
        super(TesterToBlackboard, self).__init__(name=name,
                             topic_name='/in_str',
                             topic_type=std_msgs.String,
                             blackboard_variables={"tester": None},
                             clearing_policy=py_trees.common.ClearingPolicy.NEVER
                             )
        self.blackboard = py_trees.blackboard.Blackboard()
        self.blackboard.tester = ""
        self.blackboard.tester_bool = False

        self.blackboard.target_action = ""
        self.blackboard.target_object = ""
        self.blackboard.auxilary_parameters = ""

    def update(self):
        self.logger.debug("%s.update()" % self.__class__.__name__)
        status = super(TesterToBlackboard, self).update()
        if status != py_trees.common.Status.RUNNING:
            # we got something
            if self.blackboard.tester.data == 'yo yo':
                self.blackboard.tester_bool = True
                rospy.logwarn_throttle(60, "%s: tester is on!" % self.name)
            else:
                self.blackboard.tester_bool = False

            self.feedback_message = self.blackboard.tester.data
        return status
