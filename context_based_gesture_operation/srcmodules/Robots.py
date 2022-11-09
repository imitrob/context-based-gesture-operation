
import numpy as np

class Robot():
    def __init__(self, eef_position=np.array([0,0,3]), eef_rotation=0, gripper_opened=True, random=True):
        self.eef_position = eef_position
        self.gripper_opened = gripper_opened
        self.eef_rotation = eef_rotation
        self.attached = None

        self.gripper_range = 0.14 # For Panda

    @property
    def gripper_opened_str(self):
        return 'opened' if self.gripper_opened else 'closed'

    @property
    def attached_str(self):
        return self.attached.name if self.attached else '-'

    def __str__(self):
        return f"Robot: {self.eef_position}, {self.gripper_opened_str}, rotation: {self.eef_rotation}, attached: {self.attached_str}"
