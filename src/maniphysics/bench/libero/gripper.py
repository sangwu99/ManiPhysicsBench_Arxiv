import numpy as np
from robosuite.models.grippers.panda_gripper import PandaGripperBase
REGISTERED_NAME = 'ContinuousPandaGripper'

class ContinuousPandaGripper(PandaGripperBase):

    def format_action(self, action):
        assert len(action) == self.dof
        a = float(np.clip(action[0], -1.0, 1.0))
        self.current_action = np.clip(self.current_action + np.array([-1.0, 1.0]) * self.speed * a, -1.0, 1.0)
        return self.current_action

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1
ABSOLUTE_NAME = 'AbsolutePandaGripper'

class AbsolutePandaGripper(PandaGripperBase):

    def format_action(self, action):
        assert len(action) == self.dof
        a = float(np.clip(action[0], -1.0, 1.0))
        self.current_action = np.array([-a, a])
        return self.current_action

    @property
    def dof(self):
        return 1

def register():
    from robosuite.models import grippers as _g
    _g.GRIPPER_MAPPING[REGISTERED_NAME] = ContinuousPandaGripper
    _g.GRIPPER_MAPPING[ABSOLUTE_NAME] = AbsolutePandaGripper
    if not isinstance(_g.ALL_GRIPPERS, type(_g.GRIPPER_MAPPING.keys())):
        _g.ALL_GRIPPERS = _g.GRIPPER_MAPPING.keys()
    return ContinuousPandaGripper
