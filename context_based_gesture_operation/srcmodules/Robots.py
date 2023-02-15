
import numpy as np

class Robot():
    def __init__(self, eef_position=np.array([0,0,3]), eef_rotation=0, gripper_opened=True, random=True):
        self.eef_position = eef_position
        self.gripper_opened = gripper_opened
        self.eef_rotation = eef_rotation
        self.attached = None

        self.gripper_range = 0.14 # For Panda

        self.eef_position_real = SceneUtilities.position_grid_to_real(self.eef_position)

    @property
    def gripper_opened_str(self):
        return 'opened' if self.gripper_opened else 'closed'

    @property
    def attached_str(self):
        return self.attached.name if self.attached else '-'

    def __str__(self):
        return f"Robot: {self.eef_position}, {self.gripper_opened_str}, rotation: {self.eef_rotation}, attached: {self.attached_str}"





class SceneUtilities():
    @staticmethod
    def position_grid_to_real(position, grid_lens=[4,4,4], max_scene_len=0.8):
        ''' Duplicite function in object.py
        (grid)position -> real_position
        '''
        grid_lens = np.array(grid_lens)

        one_tile_lens = max_scene_len/grid_lens
        y_translation = (grid_lens[1]-1)*one_tile_lens[1]/2

        position_scaled = position * one_tile_lens
        position_translated = position_scaled - [-0.2, y_translation, 0.]

        return position_translated

    @staticmethod
    def generate_grid(grid_lens=[4,4,4]):
        assert np.allclose(*grid_lens), "Not Implemented for different scene lens"
        xs, ys, zs = [], [], []
        for i in range(grid_lens[0]):
            x,y,z = SceneUtilities.position_grid_to_real(position=[i,i,i], grid_lens=grid_lens)
            xs.append(x)
            ys.append(y)
            zs.append(z)
        return np.array(xs), np.array(ys), np.array(zs)

    @staticmethod
    def position_real_to_grid(p, out=""):
        '''
        real_position -> (grid)position
        '''
        xs,ys,zs = SceneUtilities.generate_grid()
        x,y,z = p

        x_ = np.argmin(abs(xs-x))
        y_ = np.argmin(abs(ys-y))
        z_ = np.argmin(abs(zs-z))

        close = np.allclose(p, SceneUtilities.position_grid_to_real(position=(x_,y_,z_)), atol=2e-2)

        if out == "with close": # bool close to grid
            return np.array([x_,y_,z_]), close
        return np.array([x_,y_,z_])
